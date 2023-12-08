import torch
import json, pickle
import os 
from collections import namedtuple
from copy import deepcopy
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel
from .normalization import DatasetNormalizer 
import numpy as np

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
TestBatch = namedtuple('Batch', 'trajectories labels')
Batch = namedtuple('Batch', 'trajectories conditions')
VaeBatch = namedtuple('Batch', 'trajectories')

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

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, seq_path, save_path, model_key, horizon, include_returns = True, max_traj_len=240,
                 weight_factor=1.05, discount = 0.99, mode = 'train'):
        super().__init__()
        # dict with key:asin, val: attr
        self.meta_data_dict = json.load(open(meta_path))
        # map asin to integer and vise versa
        self.asin_id_dict = {asin: idx for idx, asin in enumerate(self.meta_data_dict.keys())}
        self.id_asin_dict = {self.asin_id_dict[asin]: asin for asin, id in self.asin_id_dict.items()}
        self.horizon = horizon
        self.max_traj_len = max_traj_len
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
        if self.mode in ['train', 'vae_train']:
            trajectories = trajectories[:int(n_episodes * 0.8)]
        # ignore vae
        elif self.mode in ['eval', 'vae_eval']:
            trajectories = trajectories[int(n_episodes * 0.8):int(n_episodes * 0.95)]
        elif self.mode == 'test':
            trajectories = trajectories[int(n_episodes * 0.95):]
        n_episodes = len(trajectories)
        self.n_episodes = n_episodes
        self.buffer = ReplayBuffer(n_episodes, max_traj_len, self.embed_dim)
        for trajectory in tqdm(trajectories):
            self.buffer.add_traj(trajectory)

        self.indices = self.make_indices(self.buffer._dict['traj_lens'], self.horizon)
        # ignore vae part
        #self.indices_vae = self.make_indices_vae(self.buffer._dict['traj_lens'])

        self.obs_dim = self.embed_dim
        self.action_dim = 1

        self.embedding_lookup = np.zeros((len(self.meta_data_dict), self.embed_dim))
        for k,v in tqdm(self.meta_data_dict.items(), desc='building embedding lookup..'):
            idx = self.asin_id_dict[k]
            self.embedding_lookup[idx] = np.array(v['embedding'])

        # no big difference 
        #self.normalizer = DatasetNormalizer(self.buffer, normalizer='CDFNormalizer', path_lengths=self.buffer._dict['traj_lens'])
        #self.normalize()
        
    def normalize(self, keys=['embeddings']):
        for key in keys:
            array = self.buffer._dict[key].reshape(self.n_episodes*self.max_traj_len, -1)
            normed = self.normalizer(array, key)
            self.buffer._dict[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_traj_len, -1)

    # added for vae
    # ignore vae part 
    def make_indices_vae(self, traj_lens):
        indices = []
        for i, path_length in enumerate(traj_lens):
            # path_length : actual length (idx contains until path_length -1)
            max_start = path_length - 1
            for start in range(max_start):
                indices.append((i, start))
        indices = np.array(indices)
        return indices
    
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

    def generate_embeddings(self, model, tokenizer, sentences, max_len=512, mode="cls"):
        # sentences : ['str', 'str', 'str', ...]
        assert mode in ["last_hidden_state", "average_hidden_state", "cls"]
        pad_token = '[SEP]'
        cls_token = '[CLS]'
        final_embeds = []
        # generate cummulative sentences under max_len
        model = model.to('cuda')
        for i in range(1, len(sentences)+1):
            for j in range(i):
                cur_str = pad_token.join(sentences[j:i])
                cur_str = cls_token + cur_str
                inputs = tokenizer(cur_str, return_tensors="pt", max_length=max_len, truncation=True, add_special_tokens=False)
                inputs = inputs.to('cuda')
                if inputs.input_ids.shape[1] <= max_len:
                    outputs = model(**inputs) # [batch, seq_len, hidden_dim]
                    if mode == "cls":
                        cur_emb = outputs.last_hidden_state[:, 0, :]
                    elif mode == "average_hidden_state":
                        cur_emb = torch.mean(outputs.last_hidden_state, dim=1)
                    elif mode == "last_hidden_state":
                        cur_emb = outputs.last_hidden_state[:, -1, :]
                    cur_emb = cur_emb.detach().cpu().numpy().squeeze()
                    final_embeds.append(cur_emb)
                    break
        return final_embeds


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
                        #"review" : product["review"],
                        'review' : trajectory[i+1]['review'],
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
        elif self.mode in ['vae_train', 'vae_eval']:
            return len(self.indices_vae)
        else:
            assert self.mode in ['train', 'eval']
            return len(self.indices)

    def __getitem__(self, idx):
        if self.mode == 'test':
            last_traj_idx = self.buffer._dict['traj_lens'][idx] -1
            #traj = self.buffer._dict['normed_embeddings'][idx, last_traj_idx-1, :] # modified!
            traj = self.buffer._dict['embeddings'][idx, last_traj_idx-1, :] # modified!
            # TODO : fix this part..?  
            label = self.buffer._dict['actions'][idx, last_traj_idx-1]
            batch = TestBatch({0: traj}, label)

        # ignore vae part 
        elif self.mode in ['vae_train', 'vae_eval']:
            traj_idx, start = self.indices_vae[idx]
            state_t = self.buffer._dict['normed_embeddings'][traj_idx, start] # [obs_dim]
            state_t_1 = self.buffer._dict['normed_embeddings'][traj_idx, start+1] # [obs_dim]
            action_t = self.buffer._dict['actions'][traj_idx, start] # [1]
            # contain (a_t, s_t, s_t_1)
            trajectories = np.concatenate([action_t, state_t, state_t_1], axis=-1, dtype=np.float32)
            batch = VaeBatch(trajectories)

        else:
            traj_idx, start, end = self.indices[idx]
            # modified!
            # observations = self.buffer._dict['normed_embeddings'][traj_idx, start:end] # [horizon, obs_dim]
            observations = self.buffer._dict['embeddings'][traj_idx, start:end] # [horizon, obs_dim]
            actions = self.buffer._dict['actions'][traj_idx, start:end] # [horizon, 1]
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
    save_path = '/home/doolee13/ReviewDiff/preprocess/embedidngs_cls.json'
    Dataset = SequenceDataset(meta_path, seq_path, save_path, 'bert-tiny', 8, mode='train')