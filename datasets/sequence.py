import torch
import json, pickle
import os 
from collections import namedtuple
from copy import deepcopy
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel
#from .buffer import ReplayBuffer
import numpy as np

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
TestBatch = namedtuple('Batch', 'trajectories labels')
Batch = namedtuple('Batch', 'trajectories conditions')

class ReplayBuffer:
    def __init__(self, num_episodes, max_traj_len, embed_dim):
        self._dict = {
            'traj_lens' : np.zeros(num_episodes, dtype=np.int32),
            'embeddings' : np.zeros((num_episodes, max_traj_len, embed_dim)),
            'rewards' : np.zeros((num_episodes, max_traj_len, 1)),
            'actions' : np.zeros((num_episodes, max_traj_len, 1), dtype=np.int32)
        }
        self._count = 0

    def add_traj(self, trajs):
        traj_len = len(trajs)
        for i, traj in enumerate(trajs):
            self._dict['embeddings'][self._count][i] = traj['embedding']
            self._dict['rewards'][self._count][i] = traj['review']
            self._dict['actions'][self._count][i] = traj['action']

        self._dict['traj_lens'][self._count] = traj_len
        self._count += 1

    def finalize(self, weight_factor):
        # make obs(embeddings) cummulative 
        for episode_idx, traj_len in enumerate(self._dict['traj_lens']):
            # [max_traj_len, embed_dim]
            embeddings_copy = deepcopy(self._dict['embeddings'][episode_idx])
            for i in range(1, traj_len):
                weights = np.power(weight_factor, np.arange(i+1))
                weights /= np.sum(weights)
                # [embed_dim]
                emb_i = np.zeros(embeddings_copy.shape[1])
                for j, weight in enumerate(weights):
                    emb_i += embeddings_copy[j] * weight
                self._dict['embeddings'][episode_idx][i] = emb_i

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, seq_path, save_path, model_key, horizon, include_returns = True, max_traj_len=240,
                 weight_factor=1.05, discount = 0.99, mode = 'train'):
        super().__init__()
        # dict with key:asin, val: attr
        self.meta_data_dict = json.load(open(meta_path))
        # map asin to integer and vise versa
        self.asin_id_dict = {asin: idx for idx, asin in enumerate(self.meta_data_dict.keys())}
        self.id_asin_dict = {self.asin_id_dict[asin]: id for asin, id in self.asin_id_dict.items()}
        self.horizon = horizon
        self.weight_factor = weight_factor
        self.include_returns = include_returns
        self.discount = discount
        self.discounts = self.discount ** np.arange(max_traj_len)[:, None]
        self.mode = mode
        
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                trajectories = pickle.load(f)
                self.embed_dim = trajectories[0][0]['embedding'].shape[0]
        else:
            self.model, self.tokenizer = self.load_LM(model_key)
            train_data = json.load(open(seq_path))
            trajectories = self.preprocess_data(train_data, self.model, self.tokenizer, save_path)
            
        n_episodes = len(trajectories)
        if self.mode == 'train':
            trajectories = trajectories[:int(n_episodes * 0.8)]
        elif self.mode == 'eval':
            trajectories = trajectories[int(n_episodes * 0.8):int(n_episodes * 0.95)]
        elif self.mode == 'test':
            trajectories = trajectories[int(n_episodes * 0.95):]
        n_episodes = len(trajectories)
        self.buffer = ReplayBuffer(n_episodes, max_traj_len, self.embed_dim)
        for trajectory in tqdm(trajectories):
            self.buffer.add_traj(trajectory)
        # makes obs(embedding) cummulative 
        self.buffer.finalize(weight_factor)

        self.indices = self.make_indices(self.buffer._dict['traj_lens'], self.horizon)

        self.obs_dim = self.embed_dim
        self.action_dim = 1

        # TODO : make normalizor for obs and action? 
    
    def make_indices(self, traj_lens, horizon):
        indices = []
        for i, path_length in enumerate(traj_lens):
            max_start = path_length -1
            max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def load_LM(self, model_key):
        assert model_key in ["bert-tiny", "bert-mini", "bert-small", "bert-medium", "distilbert-base-uncased"]
        if model_key == "distilbert-base-uncased":
            model, tokenizer = DistilBertModel.from_pretrained(model_key), DistilBertTokenizer.from_pretrained(model_key) 
        else:
            model_name = f"prajjwal1/{model_key}"
            model, tokenizer = AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)  
        self.embed_dim = model.config.hidden_size
        return model, tokenizer

    def generate_embeddings(self, model, tokenizer, sentences):
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

    def preprocess_data(self, data, model, tokenizer, save_path):
        with open(save_path, 'wb') as f:
            trajectories = []
            for trajectory in tqdm(data, desc="Encoding trajectories"):
                if len(trajectory) <= self.horizon:
                    continue
                attributes = [product["attribute"] for product in trajectory]
                embeddings = self.generate_embeddings(model, tokenizer, attributes)
                processed_trajectory = [
                    {
                        "embedding" : embeddings[i],
                        "review" : product["review"],
                        "asin" : product["asin"],
                        # integer corresponding to the next item
                        "action" : self.asin_id_dict[trajectory[i+1]["asin"]]
                    }
                    for i, product in enumerate(trajectory[:-1])
                ]
                trajectories.append(processed_trajectory)
            pickle.dump(trajectories, f)
        return trajectories
    
    def get_conditions(self, observations):
        return {0: observations[0]}

    def __len__(self):
        if self.mode == 'test':
            return len(self.buffer._dict['traj_lens'])
        else:
            assert self.mode in ['train', 'eval']
            return len(self.indices)

    def __getitem__(self, idx):
        if self.mode == 'test':
            last_traj_idx = self.buffer._dict['traj_lens'][idx] -1
            traj = self.buffer._dict['embeddings'][idx, last_traj_idx-1, :]
            label = self.buffer._dict['actions'][idx, last_traj_idx]
            batch = TestBatch({0: traj}, label)

        else:
            traj_idx, start, end = self.indices[idx]

            #TODO : implement obs normalization? 
            observations = self.buffer._dict['embeddings'][traj_idx, start:end] # [horizon, obs_dim]
            actions = self.buffer._dict['actions'][traj_idx, start:end] # [horizon, 1]
            '''
            # we will use cummulative states
            obs_copy = deepcopy(observations)
            #TODO : implement this in matrix form
            for i in range(1, self.horizon):
                weights = np.power(self.weight_factor, np.arange(i+1))
                weights /= np.sum(weights)
                obs_i = np.zeros(observations.shape[1])
                for j, weight in enumerate(weights):
                    obs_i += obs_copy[j] * weight
                observations[i] = obs_i
            '''
            conditions = self.get_conditions(observations)
            trajectories = np.concatenate([actions, observations], axis=-1, dtype=np.float32) # [horizon, 1 + obs_dim]       
            if self.include_returns:
                returns_scale = self.buffer._dict['traj_lens'][traj_idx] * 5.0
                rewards = self.buffer._dict['rewards'][traj_idx][start:]
                discounts = self.discounts[:len(rewards)]
                returns = (discounts * rewards).sum()
                returns = np.array([returns/returns_scale], dtype=np.float32)
                batch = RewardBatch(trajectories, conditions, returns)
            else:
                batch = Batch(trajectories, conditions)
        return batch

if __name__ == '__main__':
    meta_path = '/home/doolee13/ReviewDiff/preprocess/meta_data.json'
    seq_path = '/home/doolee13/ReviewDiff/preprocess/train_data.json'
    save_path = '/home/doolee13/ReviewDiff/preprocess/embedidngs.json'
    Dataset = SequenceDataset(meta_path, seq_path, save_path, 'bert-tiny', 8, mode='test')
    cur_data = Dataset[0]