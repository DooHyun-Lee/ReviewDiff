from datasets.sequence import SequenceDataset
from models.transformernet import Transformernet
from models.diffuser_transformer import TransformerDiffusion
from models.inverse_model import InverseDynamics
from trainer.train_transformer import Trainer
import argparse  

parser = argparse.ArgumentParser()

# Transformer configs
parser.add_argument('--in_channels', default=128, type=int)
parser.add_argument('--model_channels', default=128, type=int)
parser.add_argument('--out_channels', default=128, type=int)
# inv dynamics configs
parser.add_argument('--dropout', default=0.1, type=float)
# Diffusion model configs
parser.add_argument('--horizon', default=19, type=int)
parser.add_argument('--obs_dim', default=128, type=int)
parser.add_argument('--num_diffusion_timesteps', default=2000, type=int)
parser.add_argument('--apply_condition', default=False)
# Datset Configs
parser.add_argument('--meta_path', default='/home/doolee13/ReviewDiff5/preprocess/movielens/ml-1m/meta_data.json', type=str) # path containing item info
parser.add_argument('--seq_path', default='/home/doolee13/ReviewDiff5/preprocess/movielens/ml-1m/train_data.json', type=str) # path containing trajectories
parser.add_argument('--include_returns', default=True)
parser.add_argument('--max_traj_len', default=2500, type=int, help='max traj length for each user') 
parser.add_argument('--discount', default=0.99, type=float, help='discount factor for RL reward') 
# Trainer Configs
parser.add_argument('--ema_rate', default=0.9999, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.00002, type=float)
parser.add_argument('--lr_anneal_steps', default=400000, type=int)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--log_interval', default=300, type=int)
parser.add_argument('--test_interval', default=30000, type=int)
parser.add_argument('--device', default='cuda', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    train_dataset = SequenceDataset(meta_path=args.meta_path, seq_path=args.seq_path,
                                    horizon=args.horizon, include_returns=args.include_returns,
                                    max_traj_len=args.max_traj_len, discount=args.discount, mode='train')
    test_dataset = SequenceDataset(meta_path=args.meta_path, seq_path=args.seq_path,
                                    horizon=args.horizon, include_returns=args.include_returns,
                                    max_traj_len=args.max_traj_len, discount=args.discount, mode='test')

    tranformer_model = Transformernet(in_channels=args.in_channels, model_channels=args.model_channels,
                                      out_channels=args.out_channels)

    diffusion_model = TransformerDiffusion(tranformer_model, train_dataset.embedding_lookup, horizon=args.horizon, obs_dim=args.obs_dim,
                                           num_diffusion_timesteps=args.num_diffusion_timesteps,
                                           apply_condition=args.apply_condition)

    trainer = Trainer(diffusion_model, train_dataset.embedding_lookup, train_dataset, test_dataset, 
                      ema_rate=args.ema_rate, batch_size=args.batch_size, lr=args.lr, 
                      lr_anneal_steps=args.lr_anneal_steps, weight_decay=args.weight_decay,
                      log_interval=args.log_interval, test_interval=args.test_interval, device=args.device)


    trainer.train()