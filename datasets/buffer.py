import numpy as np
from copy import deepcopy

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