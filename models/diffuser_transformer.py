import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def get_beta_schedule(num_diffusion_timesteps):
    def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        '''
        num_diffusion_timesteps : integer
        alpha_bar : function creates alpha_bar given t
        return : betas
        '''
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i+1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return betas
    return np.array(betas_for_alpha_bar(num_diffusion_timesteps, lambda t: 1-np.sqrt(t + 0.0001)))

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class TransformerDiffusion:
    def __init__(self, model, inv_model, horizon, obs_dim, num_diffusion_timesteps, apply_condition):
        self.apply_condition = apply_condition
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.use_timesteps = list(range(num_diffusion_timesteps))
        betas = get_beta_schedule(num_diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        last_alpha_cumprod = 1.0
        new_betas = []
        self.timestep_map = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1-alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        self.betas = np.array(new_betas, dtype=np.float64)
        assert len(self.betas.shape) == 1
        assert (self.betas > 0).all() and (self.betas <= 1).all()

        self.num_timesteps = int(self.betas.shape[0])

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, )

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.model = model
        self.inv_model = inv_model

    def get_x_start(self, x, std):
        noise = torch.randn_like(x)
        assert noise.shape == x.shape
        return (x + std * noise)

    def apply_conditioning(self, x, conditions):
        for t, val in conditions.items():
            x[:, t, :] = val.clone()
        return x

    def q_sample(self, x_start, t, noise=None):
        '''
        sample from q(x_t | x_0)
        x_start : initial data x_0
        t : diffusion step
        return : noisy sample at timestep t 
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start 
             + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_mean_variance(self, x_start, t):
        '''
        get q(x_t | x_0) distribution
        '''
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def loss(self, x_input, cond, returns=None):
        # x : [batch, horizon, 1+obs_dim] trajectories
        x = x_input[:, :, 1:]

        bsize = x.shape[0]
        ts = torch.randint(0, self.num_diffusion_timesteps, (bsize,), device=x.device).long()
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, 
                                   torch.tensor([0]).to(x.device), x.shape)
        x_start_log_var = 2 * torch.log(std)
        # x : pure batch 
        # x_start : pure batch + std noised
        x_start = self.get_x_start(x, std)
        
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, ts, noise=noise)
        if self.apply_condition:
            x_t = self.apply_conditioning(x_t, cond)

        # rescale timesteps 
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts] 
        new_ts = new_ts.float() * (1000.0 / self.num_timesteps)
        # estimation of x_0 by model
        model_output = self.model(x_t, new_ts)

        terms = {}
        target = x_start
        assert model_output.shape == target.shape == x_start.shape
        if self.apply_condition:
            target = self.apply_conditioning(target, cond)
            model_output = self.apply_conditioning(model_output, cond)
        terms["mse"] = mean_flat((target - model_output) ** 2)
        # at diffusion step 0, we calculate mse with original batch(without std)
        t0_mask = (ts==0)
        target_t0 = x
        if self.apply_condition:
            target_t0 = self.apply_conditioning(x, cond)
        t0_loss = F.mse_loss(target_t0, model_output)
        t0_loss = mean_flat((target_t0 - model_output) ** 2)
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])
        # regulizer? mean at timestep T
        out_mean, _, _ = self.q_mean_variance(x_start, torch.LongTensor([self.num_timesteps-1]).to(x_start.device))
        tT_loss = mean_flat(out_mean ** 2)

        terms["loss"] = terms["mse"] + tT_loss

        # inv loss 
        x_now = x_input[:, :-1, 1:]
        x_next = x_start[:, 1:, :] # already considered last dimension
        x_comb = torch.cat([x_now, x_next], dim=-1)
        x_comb = x_comb.reshape(-1, 2 * self.obs_dim)
        a_now = x_input[:, :-1, :1]
        a_now_flat = a_now.reshape(-1).to(torch.int64)
        pred_a_now = self.inv_model(x_comb)
        inv_loss = F.cross_entropy(pred_a_now, a_now_flat)

        tot_loss = 1/2 * (terms['loss'].mean() + inv_loss)

        info = {}
        info['diffusion loss'] = terms['loss'].mean().item()
        info['inv loss'] = inv_loss.item()

        return tot_loss, info

# ===================== end of training code ============================

    def q_posterior_mean_variance(self, x_start, x_t, t):
        '''
        returns distribution of q(x_{t-1} | x_t, x_0)
        '''
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape)
        )
        posterior_variance = (
            _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        )
        posterior_log_variance_clipped = (
            _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_mean_variance(self, x, t, model, clipped_denoised=False):
        # rescale timesteps 
        map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        new_ts = map_tensor[t] 
        new_ts = new_ts.float() * (1000.0 / self.num_timesteps) 
        #model_output = self.model(x, t)
        model_output = model(x, t)

        if clipped_denoised:
            model_output = model_output.clamp(-1, 1)

        model_variance = _extract_into_tensor(
            self.posterior_variance, t, x.shape
        )
        model_log_variance = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x.shape
        )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=model_output, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == model_output.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": model_output,
        }

    @torch.no_grad()
    def p_sample(self, x, t, model):
        out = self.p_mean_variance(x, t, model)
        noise = torch.randn_like(x)
        # no noise when t ==0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        sample = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                'greedy_mean':out["mean"], 'out':out}
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, model):
        device = next(self.model.parameters()).device
        noise = torch.randn(*shape, device=device)
        noise = self.apply_conditioning(noise, cond)
        x = noise

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.tensor([i] * shape[0], device=device)
            out = self.p_sample(x, t, model)
            x = out['sample']
            x = self.apply_conditioning(x, cond)
        return x

    @torch.no_grad()
    def conditional_sample(self, cond, model):
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.obs_dim)
        return self.p_sample_loop(shape, cond, model)