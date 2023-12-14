from .inverse_model import InverseDynamics
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def linear_beta_schedule(timesteps, start, end, dtype=torch.float32):
    return torch.linspace(start, end, timesteps, dtype=dtype)

def linear_variance_beta_schedule(timesteps, start, end, dtype=torch.float32):
    variance = torch.linspace(start, end, timesteps, dtype=dtype)
    alpha_bar = 1- variance
    betas = []
    betas.append(1-alpha_bar[0])
    for i in range(1, timesteps):
        betas.append(min(1-alpha_bar[i] / alpha_bar[i-1], 0.999))
    return torch.tensor(betas, dtype=dtype)

class WeightedStateLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss ' : weighted_loss}

class WeightedStateL2(WeightedStateLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

# inv dynamcis seperate training 
class GaussianDiffusion(nn.Module):
    def __init__(self, model, embedding_lookup, horizon, obs_dim, action_dim, item_num, n_timesteps=200, 
                 clip_denoised=True, predict_epsilon=True, dropout=0.1, hidden_dim=256,
                 loss_discount=1.0, loss_weights=None, returns_condition=True,
                 condition_guidance_w=1.2, train_only_inv=False, train_only_diff=True):
        super().__init__()
        self.embedding_lookup = embedding_lookup
        self.horizon = horizon # in terms of episode t 
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.item_num = item_num
        self.dropout = dropout
        self.model = model # Unet model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.train_only_inv = train_only_inv
        self.train_only_diff = train_only_diff

        betas = cosine_beta_schedule(n_timesteps) #(n_timesteps,)

        # this terms are for smaller noise 
        self.noise_scale = 0.005
        self.noise_min = 0.001
        self.noise_max = 0.005
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        # choice 1
        betas = linear_variance_beta_schedule(n_timesteps, start, end)
        # choice 2
        #betas = linear_beta_schedule(n_timesteps, start, end)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = WeightedStateL2(loss_weights)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # ------------ trainig ----------------
        # for q_sample
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. -alphas_cumprod))
        # ------------ trainig ----------------

        # ------------ sampling ----------------
        # for predict_start_from_noise 
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # for q_posterior : q(x_t-1 | x_t, x_0)
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))

        # ------------ sampling ----------------
        '''
        self.inv_model = nn.Sequential(
            nn.Linear(2 * self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.item_num)
        )
        '''
        self.inv_model = InverseDynamics(self.obs_dim, self.dropout, embedding_lookup)

    def get_loss_weights(self, discount):
        self.action_weight = 1
        dim_weights = torch.ones(self.obs_dim, dtype=torch.float32)

        # trajectory timestep decay (discount ** t)
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        # [horizon, obs_dim]
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        if self.predict_epsilon:
            loss_weights[0, :] = 0
        return loss_weights
    
    # helper functions
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1,t)
        return out.reshape(b, *((1,) *(len(x_shape)-1)))

    def apply_conditioning(self, x, conditions, action_dim):
        # cond : {0 : obs at timestep 0}
        # x_start : [batch, horizon, obs_dim]
        for t, val in conditions.items():
            # set timestep 0 val
            x[:, t, action_dim:] = val.clone()
        return x

    # These funcs are used for training diffuser # 
    # loss, p_losses, q_sample
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.rand_like(x_start)

        sample = (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        # cond : {0 : obs at timestep 0}
        # x_start : [batch, horizon, obs_dim]
        noise = torch.rand_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self.apply_conditioning(x_noisy, cond, 0)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = self.apply_conditioning(x_recon, cond, 0)
        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns = None):
        # x : [batch, horizon, 1+obs_dim] trajectories
        # cond : {0 : [batch, obs_dim]} condition
        # returns : [batch, 1]

        # TODO : make more options 
        # currently only diffusion trainig
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffusion_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns)
        '''
        # inv dynamic loss
        x_t = x[:, :-1, self.action_dim:]
        x_t_1 = x[:, 1:, self.action_dim:]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * self.obs_dim)
        a_t = x[:, :-1, :self.action_dim]
        #a_t_flat = a_t.view(-1, self.action_dim)
        a_t_flat = a_t.reshape(-1).to(torch.int64)
        pred_a_t = self.inv_model(x_comb_t)
        inv_loss = F.cross_entropy(pred_a_t, a_t_flat)
        loss = (1/2) * (diffusion_loss + inv_loss)
        '''
        loss = diffusion_loss
        return loss , info

    def loss_inv_dyn(self, x, cond, returns=None):
        x_t = x[:, :-1, self.action_dim:]
        x_t_1 = x[:, 1:, self.action_dim:]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * self.obs_dim)
        a_t = x[:, :-1, :self.action_dim]
        #a_t_flat = a_t.view(-1, self.action_dim)
        a_t_flat = a_t.reshape(-1).to(torch.int64)
        pred_a_t = self.inv_model(x_comb_t)
        inv_loss = F.cross_entropy(pred_a_t, a_t_flat)
        return inv_loss

    # end of funcs for training 

    # this functions are sampling   
    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond-epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
    
        model_mean, posterior_variacne, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variacne, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise at t==0
        nonzero_mask = (1- (t==0).float()).reshape(b, *((1,) *(len(x.shape) -1)))
        return model_mean + nonzero_mask*(0.5 * model_log_variance).exp()*noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None):
        device = self.betas.device

        batch_size = shape[0]
        # start from the noise
        x = 0.5*torch.randn(shape, device=device)
        x = self.apply_conditioning(x, cond, 0)

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = self.apply_conditioning(x, cond, 0)
        return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        #device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.obs_dim)
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    # end of funcs for sampling