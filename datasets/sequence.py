import torch
import json, pickle
import os 
from collections import namedtuple
from copy import deepcopy
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import random

# type 1 : use sentence 

TestBatch = namedtuple('Batch', 'condition labels')
TrainBatch = namedtuple('Batch', 'trajectories condition')

class ReplayBuffer:
    def __init__(self, num_episodes, max_traj_len, embed_dim=128):
        self._dict = {
            'traj_lens' : np.zeros(num_episodes, dtype=np.int32),
            'state_ids' : np.zeros((num_episodes, max_traj_len, 1), dtype=np.int32), # id converted asins
            'rewards' : np.zeros((num_episodes, max_traj_len, 1)),
            'actions' : np.zeros((num_episodes, max_traj_len, 1), dtype=np.int32) # also ids
        }
        self._count = 0

    def add_traj(self, trajs):
        traj_len = len(trajs)
        for i, traj in enumerate(trajs):
            self._dict['state_ids'][self._count][i] = traj['state_id']
            #self._dict['rewards'][self._count][i] = traj['review']
            #self._dict['actions'][self._count][i] = traj['action']

        self._dict['traj_lens'][self._count] = traj_len
        self._count += 1

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, seq_path, horizon=5, include_returns = True, max_traj_len=245,
                 discount = 0.99, emb_dim=128,  mode = 'train', split_mode = 'bert4rec'):
        super().__init__()
        # dict with key:asin, val: attr
        self.meta_data_dict = json.load(open(meta_path))
        # map asin to integer and vise versa
        self.asin_id_dict = {asin: idx for idx, asin in enumerate(self.meta_data_dict.keys())}
        self.id_asin_dict = {self.asin_id_dict[asin]: asin for asin, id in self.asin_id_dict.items()}
        self.horizon = horizon
        self.max_traj_len = max_traj_len
        self.include_returns = include_returns
        self.discount = discount
        self.discounts = self.discount ** np.arange(max_traj_len)[:, None]
        self.emb_dim = emb_dim
        self.mode = mode
        # choose between bert4rec like method or normal 8:1:1 split
        self.splitmode = split_mode

        self.embedding_lookup = np.zeros((len(self.meta_data_dict), self.emb_dim), dtype=np.float32)
        for k, v in tqdm(self.meta_data_dict.items(), desc='building embedding lookup ...'):
            idx = self.asin_id_dict[k]
            self.embedding_lookup[idx] = np.array(v['embedding'], dtype=np.float32)

        train_data = json.load(open(seq_path))
        trajectories= self.preprocess_data(train_data)
        n_episodes = len(trajectories)

        self.n_episodes = n_episodes
        self.buffer = ReplayBuffer(n_episodes, max_traj_len)
        for trajectory in tqdm(trajectories):
            self.buffer.add_traj(trajectory)

        # -1 traj_lens for making training dataset
        self.indices = self.make_indices(self.buffer._dict['traj_lens'] -1, self.horizon)


    def preprocess_data(self, train_data):
        trajectories = []
        for trajectory in train_data:
            processed_trajectory = [
                {
                    #'state_id' : self.asin_id_dict[product['asin']],
                    'state_id' : self.asin_id_dict[str(product['movie_id'])],
                    #'review' : trajectory[i+1]['review'],
                    #'action' : self.asin_id_dict[trajectory[i+1]['asin']]
                }
                for i, product in enumerate(trajectory)
            ] 
            trajectories.append(processed_trajectory)
        random.shuffle(trajectories)
        return trajectories

    def make_indices(self, traj_lens, horizon):
        indices = []
        for i, path_length in enumerate(traj_lens):
            max_start = path_length -1
            max_start = min(max_start, path_length-horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, states):
        if self.mode == 'train':
            return {0: states[0]}
        elif self.mode == 'test':
            cond_dict = {}
            for i in range(len(states)):
                cond_dict[i] = states[i]
            return cond_dict
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.indices)
        else:
            assert self.mode == 'test'
            return len(self.buffer._dict['traj_lens'])

    def __getitem__(self, idx):
        if self.mode == 'test':
            last_idx = self.buffer._dict['traj_lens'][idx] -1
            label = self.buffer._dict['state_ids'][idx][last_idx]
            # load last 4 states (horizon=5 -1)
            states = self.buffer._dict['state_ids'][idx][last_idx -(self.horizon-1):last_idx].squeeze()
            states_emb = self.embedding_lookup[states]
            
            condition = self.get_conditions(states_emb)
            return TestBatch(condition, label)

        elif self.mode == 'train':
            traj_idx, start, end = self.indices[idx]
            states = self.buffer._dict['state_ids'][traj_idx][start:end].squeeze()
            states_emb = self.embedding_lookup[states]

            condition = self.get_conditions(states_emb)
            return TrainBatch(states_emb, condition)

if __name__ == '__main__':
    meta_path = os.path.join(os.getcwd(), '..', 'preprocess/meta_data.json')
    seq_path = os.path.join(os.getcwd(), '..', 'preprocess/train_data.json')
    Dataset = SequenceDataset(meta_path, seq_path, mode='train')
    testset = SequenceDataset(meta_path, seq_path, mode='test')
    traj, cond = Dataset[0]
    states, cond, label = testset[0]