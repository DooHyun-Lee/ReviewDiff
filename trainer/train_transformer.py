from copy import deepcopy
import torch
from .ranker import Ranker, AverageMeterSet
from tqdm import tqdm
import numpy as np
import random

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

class Trainer:
    def __init__(
                self, 
                diffusion_model,
                train_dataset,
                eval_dataset,
                test_dataset,
                ema_rate=0.9999,
                batch_size=64,
                lr=0.0001,
                lr_anneal_steps=400000,
                weight_decay=0.0,
                log_interval=300,
                eval_interval=2000,
                test_interval=5000,
                device = 'cuda',
    ):
        self.model = diffusion_model
        self.ema_rate = ema_rate
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.lr_anneal_steps = lr_anneal_steps
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.test_interval = test_interval
        self.device = device

        self.train_dataloader = cycle(torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.eval_dataloader = torch.utils.data.DataLoader(
             self.eval_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True
        )
        self.test_dataloader = torch.utils.data.DataLoader(
             self.test_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True, 
             drop_last = True
        )
        all_params = list(self.model.model.parameters()) + list(self.model.inv_model.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.lr, weight_decay=self.weight_decay)
        self.step = 0

        # ema for only diffusion model (not inverse dynamics)
        self.ema_model = deepcopy(self.model.model)

    def train(self):
        seed = 42
        torch.manual_seed(seed)
        torch.backends.cudnn.derministic=True
        torch.backends.cudnn.benmark=False
        random.seed(seed)
        np.random.random(seed)
        
        # do I have to send ema_params as well? : yes
        self.model.model = self.model.model.to(self.device)
        self.model.inv_model = self.model.inv_model.to(self.device)
        self.ema_model = self.ema_model.to(self.device)
        self.model.model.train()
        self.model.inv_model.train()

        while self.step < self.lr_anneal_steps:
            self.optimizer.zero_grad()

            batch = next(self.train_dataloader)
            batch = batch_to_device(batch, device=self.device)
            loss, info = self.model.loss(*batch)
            loss.backward()    
            if (self.step+1) % self.log_interval ==0:
                diffusion_loss = info['diffusion loss']
                inv_loss = info['inv loss']
                print(f'current diffusion loss at timestep {self.step}: {diffusion_loss}')
                print(f'current inv loss at timestep {self.step}: {inv_loss}')
                # grad norm check
                sqsum = 0.0
                for p in list(self.model.model.parameters()):
                    sqsum += (p.grad ** 2).sum().item()
                print(f'current grad norm at timestep {self.step}: {sqsum}')

            # lr annealing
            frac_done = (self.step) / self.lr_anneal_steps
            lr = self.lr * (1- frac_done)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.step()
            # update ema model
            for targ, src in zip(self.ema_model.parameters(), self.model.model.parameters()):
                targ.detach().mul_(self.ema_rate).add_(src, alpha=1-self.ema_rate)
            
            if (self.step+1) % self.eval_interval ==0:
                eval_loss_log_diff, eval_loss_log_inv = 0, 0
                with torch.no_grad():
                    for eval_items in tqdm(self.eval_dataloader, desc='evaluating...'):
                        eval_items =batch_to_device(eval_items)
                        loss, info = self.model.loss(*eval_items)
                        eval_loss_log_diff += info['diffusion loss']
                        eval_loss_log_inv += info['inv loss']
                    
                    eval_loss_log_diff /= len(self.eval_dataloader)
                    eval_loss_log_inv /= len(self.eval_dataloader)
                    print('---------------------------------------')
                    print(f'eval diffusion loss at timestep {self.step}: {eval_loss_log_diff}')
                    print(f'eval inv loss at timestep {self.step}: {eval_loss_log_inv}')
                    print(f'current learning at timestep {self.step}: {lr}')
                    print('---------------------------------------')
            
            # TODO : implement testing (sampling part)
            if (self.step + 1) % self.test_interval == 0:
                self.ema_model.eval()
                ks = [10]
                ranker = Ranker(ks)
                average_meter_set = AverageMeterSet()
                with torch.no_grad():
                    for test_items in tqdm(self.test_dataloader, desc='testing...'):
                        test_items, labels = batch_to_device(test_items)
                        samples = self.model.conditional_sample(test_items, self.ema_model)
                        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
                        scores = self.model.inv_model(obs_comb) 
                       
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
                self.ema_model.train()

            self.step += 1