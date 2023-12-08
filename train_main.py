from datasets.sequence_cumul import SequenceDataset
from models.unet import TemporalUnet
from models.diffuser import GaussianDiffusion
from trainer.train import Trainer
import argparse  

parser = argparse.ArgumentParser()
# Unet Configs
parser.add_argument('--horizion', default=8, type=int, help='column length for diffusion')
parser.add_argument('--transition_dim', default=128, type=int, help='obs embedding size') 
parser.add_argument('--cond_dim', default=128, type=int, help='obs embedding size (not used!)') 
parser.add_argument('--dim', default=128, type=int, help='hidden dimension for diffusion model') 
parser.add_argument('--dim_mults', default=(1,2,4,8), type=tuple, help='factor for hidden dimension') 
parser.add_argument('--returns_condition', default=True, help='using condition for diffusion model') 
parser.add_argument('--condition_dropout', default=0.25, type=float) 
parser.add_argument('--kernel_size', default=5, type=int, help='kernel size for conv1d in Unet') # default 5
# Diffusion model Configs
parser.add_argument('--action_dim', default=1, help='we are using item index')  
parser.add_argument('--n_timesteps', default=200, type=int, help='T for diffusion') # default 200
parser.add_argument('--clip_denoised', default=True, help='activate clip during sampling') 
parser.add_argument('--predict_epsilon', default=False, help='Unet will predict epsilon') # default True
parser.add_argument('--hidden_dim', default=256, type=int, help='dim for inv dynamics') 
parser.add_argument('--dropout', default=0.1, type=float, help='dim for inv dynamics') 
parser.add_argument('--loss_discount', default=1.0, type=float, help='discount along horizon in diffusion') # default 1.0
parser.add_argument('--condition_guidance_w', default=1.2, type=float, help='classifier free guidance constant') 
parser.add_argument('--train_only_inv', default=False)
parser.add_argument('--train_only_diff', default=False)
# Datset Configs
parser.add_argument('--meta_path', default='/home/doolee13/ReviewDiff/preprocess/meta_data.json', type=str) # path containing item info
parser.add_argument('--seq_path', default='/home/doolee13/ReviewDiff/preprocess/train_data.json', type=str) # path containing trajectories
parser.add_argument('--save_path', default='/home/doolee13/ReviewDiff/preprocess/embedidngs_cls.json', type=str) # path containing embedded trajectories
parser.add_argument('--model_key', default='bert-tiny', type=str)
parser.add_argument('--horizon', default=8, type=int)
parser.add_argument('--include_returns', default=True)
parser.add_argument('--max_traj_len', default=240, type=int, help='max traj length for each user') 
parser.add_argument('--weight_factor', default=1.05, type=float,help='weight factor for trajectory cummulation') 
parser.add_argument('--discount', default=0.99, type=float, help='discount factor for RL reward') 
# Trainer Configs
parser.add_argument('--ema_decay', default=0.995, type=float, help='exponential moving average constant') 
parser.add_argument('--train_bsize', default=32, type=int)
parser.add_argument('--eval_bsize', default=64, type=int)
parser.add_argument('--lr_diffusion', default=2e-5, type=float) # 2e-5
parser.add_argument('--lr_inv', default=1e-4, type=float) # 1e-4
parser.add_argument('--gradient_accumulate_every', default=2, type=int)
parser.add_argument('--eval_every', default=200, type=int)
parser.add_argument('--test_every', default=2000, type=int)
parser.add_argument('--step_start_ema', default=2000, type=int, help='step to start update using ema copy ') 
parser.add_argument('--update_ema_every', default=10, type=int)
parser.add_argument('--device', default='cuda', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    train_dataset = SequenceDataset(meta_path=args.meta_path, seq_path=args.seq_path,
                                    save_path=args.save_path, model_key=args.model_key,
                                    horizon=args.horizon, include_returns=args.include_returns,
                                    max_traj_len=args.max_traj_len, weight_factor=args.weight_factor,
                                    discount=args.discount, mode='train')
    eval_dataset = SequenceDataset(meta_path=args.meta_path, seq_path=args.seq_path,
                                    save_path=args.save_path, model_key=args.model_key,
                                    horizon=args.horizon, include_returns=args.include_returns,
                                    max_traj_len=args.max_traj_len, weight_factor=args.weight_factor,
                                    discount=args.discount, mode='eval')
    test_dataset = SequenceDataset(meta_path=args.meta_path, seq_path=args.seq_path,
                                    save_path=args.save_path, model_key=args.model_key,
                                    horizon=args.horizon, include_returns=args.include_returns,
                                    max_traj_len=args.max_traj_len, weight_factor=args.weight_factor,
                                    discount=args.discount, mode='test')

    unet_model = TemporalUnet(horizon=args.horizon, transition_dim=args.transition_dim,
                              cond_dim=args.cond_dim, dim=args.dim, dim_mults=args.dim_mults,
                              returns_conditon=args.returns_condition, 
                              condition_dropout=args.condition_dropout, kernel_size=args.kernel_size)
    diffusion_model = GaussianDiffusion(model=unet_model, embedding_lookup=train_dataset.embedding_lookup, horizon=args.horizon, obs_dim=args.transition_dim,
                                        action_dim=args.action_dim, item_num=len(train_dataset.asin_id_dict),
                                        n_timesteps=args.n_timesteps, clip_denoised=args.clip_denoised,
                                        predict_epsilon=args.predict_epsilon, dropout=args.dropout, hidden_dim=args.hidden_dim,
                                        loss_discount=args.loss_discount, returns_condition=args.returns_condition,
                                        condition_guidance_w=args.condition_guidance_w, train_only_inv=args.train_only_inv,
                                        train_only_diff=args.train_only_diff)
    trainer = Trainer(diffusion_model=diffusion_model, train_dataset=train_dataset, 
                      eval_dataset=eval_dataset, test_dataset= test_dataset, ema_decay=args.ema_decay, train_bsize=args.train_bsize,
                      eval_bsize=args.eval_bsize, lr_diffuser=args.lr_diffusion, lr_inv_mlp=args.lr_inv, gradient_accumulate_every=args.gradient_accumulate_every,
                      eval_every=args.eval_every, test_every=args.test_every,
                      step_start_ema=args.step_start_ema, 
                      update_ema_every=args.update_ema_every, device=args.device)
    
    trainer.train()