from copy import deepcopy
import torch
from .ranker import Ranker, AverageMeterSet
from tqdm import tqdm

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def to_device(x, device='cuda'):
	if torch.is_tensor(x):
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	else:
		print(f'Unrecognized type in `to_device`: {type(x)}')

def batch_to_device(batch, device='cuda:0'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

class EMA():
    def __init__(self, beta):
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new 
        else:
            return old * self.beta + (1-self.beta) * new
    
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

class Trainer:
    def __init__(
        self, 
        diffusion_model,
        train_dataset,
        eval_dataset,
        test_dataset,
        ema_decay=0.995,
        train_bsize=32,
        eval_bsize=64,
        lr_diffuser=2e-5,
        lr_inv_mlp=1e-4,
        gradient_accumulate_every=2,
        eval_every = 100,
        test_every = 500, 
        step_start_ema=2000,
        update_ema_every=10,
        device='cuda'
    ):
        self.model = diffusion_model
        self.ema = EMA(beta=ema_decay)        
        self.ema_model = deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.train_bsize = train_bsize
        self.eval_bsize = eval_bsize
        self.gradient_accumulate_every = gradient_accumulate_every
        self.eval_every = eval_every
        self.test_every = test_every
        self.train_dataset = train_dataset  
        self.eval_dataset = eval_dataset  
        self.test_dataset = test_dataset
        self.device = device

        self.train_dataloader = cycle(torch.utils.data.DataLoader(
            self.train_dataset, batch_size=train_bsize, num_workers=0, shuffle=True, pin_memory=True
        ))
        '''
        self.eval_dataloader = cycle(torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=eval_bsize, num_workers=0, shuffle=True, pin_memory=True
        ))
        '''
        self.eval_dataloader = torch.utils.data.DataLoader(
             self.eval_dataset, batch_size=eval_bsize, num_workers=0, shuffle=True, pin_memory=True
        )
        self.test_dataloader = torch.utils.data.DataLoader(
             self.test_dataset, batch_size=eval_bsize, num_workers=0, shuffle=True, pin_memory=True, 
             drop_last = True
        )
        #self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=lr_diffuser)
        self.optimizer_inv = torch.optim.Adam(self.model.inv_model.parameters(), lr=lr_inv_mlp)
        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, n_train_steps=100001):
        import wandb
        wandb.init(project='Reviewdiff')
        self.model = self.model.to(self.device)
        self.ema_model = self.ema_model.to(self.device)
        self.model.train()
        loss_log, loss_inv_log = 0, 0
        for _ in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.train_dataloader)
                batch = batch_to_device(batch, device=self.device)
                # diffusion part
                loss, infos = self.model.loss(*batch)
                loss_log += (loss.item() / self.gradient_accumulate_every)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                # inv dynamics part
                loss_inv = self.model.loss_inv_dyn(*batch)
                loss_inv_log += (loss_inv.item() / self.gradient_accumulate_every)
                loss_inv = loss_inv / self.gradient_accumulate_every
                loss_inv.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.optimizer_inv.step()
            self.optimizer_inv.zero_grad()
            if (self.step +1) % 200 ==0:
                print(f'current loss at {self.step} => diffusion loss: {loss_log}, inv mlp loss : {loss_inv_log}')
            wandb.log({'diffusion loss': loss_log, 'inv mlp loss' : loss_inv_log})
            loss_log, loss_inv_log = 0, 0

            if (self.step+1) % self.update_ema_every == 0 :
                self.step_ema()

            if (self.step+1) % self.eval_every == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_loss_log, eval_loss_inv_log =0, 0
                    for eval_items in tqdm(self.eval_dataloader, desc='evaluating... '):
                        eval_items = batch_to_device(eval_items)
                        # diffusion part
                        loss, infos = self.model.loss(*eval_items)
                        eval_loss_log += loss.item()
                        # inv dynamics part
                        loss_inv = self.model.loss_inv_dyn(*eval_items)
                        eval_loss_inv_log += loss_inv.item()

                    eval_loss_log /= len(self.eval_dataloader)
                    eval_loss_inv_log /= len(self.eval_dataloader)
                    wandb.log({'eval_diffusion loss': eval_loss_log,
                              'eval_inv mlp loss': eval_loss_inv_log})
                    print('------------------------------------------------')
                    print(f'evaluation loss at {self.step} => diffusion loss: {eval_loss_log} inv mlp loss : {eval_loss_inv_log}')
                    print('------------------------------------------------')
                self.model.train()         

            self.test_every = 1
            if (self.step+1) % self.test_every == 0:
                self.ema_model.eval()
                self.model.eval()
                ks = [10]
                ranker = Ranker(ks)
                average_meter_set = AverageMeterSet()
                with torch.no_grad():
                    returns = torch.ones(self.eval_bsize, 1, device=self.device)
                    for test_items in tqdm(self.test_dataloader, desc='testing...'):
                        test_items, labels = batch_to_device(test_items)
                        #samples = self.ema_model.conditional_sample(test_items, returns)
                        samples = self.model.conditional_sample(test_items, returns)
                        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
                        #scores = self.ema_model.inv_model(obs_comb) # [batch, num_items]
                        scores = self.model.inv_model(obs_comb) # [batch, num_items]

                        res = ranker(scores, labels)
                        metrics = {}
                        for i, k in enumerate(ks):
                            metrics["NDCG@%d" % k] = res[2*i]
                            metrics["Recall@%d" % k] = res[2*i+1]
                        metrics["MRR"] = res[-3]
                        metrics["AUC"] = res[-2]
                        
                        for k, v in metrics.items():
                            average_meter_set.update(k, v)
                average_metrics = average_meter_set.averages()
                print(f'Test result: {average_metrics}')
                self.model.train()
                self.ema_model.train()

            self.step += 1