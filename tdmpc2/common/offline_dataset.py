import pdb

import pickle
import numpy as np
import gym

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class IRLBuffer():
    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device('cuda')
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None,
            strict_length=True,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            prefetch=1,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f'Buffer capacity: {self._capacity:,}')
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum([
                (v.numel()*v.element_size() if not isinstance(v, TensorDict) \
                else sum([x.numel()*x.element_size() for x in v.values()])) \
            for v in tds.values()
        ]) / len(tds)
        total_bytes = bytes_per_step*self._capacity
        print(f'Storage required: {total_bytes/1e9:.2f} GB')
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = 'cuda' if 2.5*total_bytes < mem_free else 'cpu'
        print(f'Using {storage_device.upper()} memory for storage.')
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (arg.to(device, non_blocking=True) \
                    if arg is not None else None for arg in args)

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        obs = td['obs']
        action = td['action'][1:]
        reward = td['reward'][1:].unsqueeze(-1)
        task = td['task'][0] if 'task' in td.keys() else None
        return self._to_device(obs, action, reward, task)

    def add(self, td):
        """Add an episode to the buffer."""
        td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * self._num_eps
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        return self._prepare_batch(td)

    def load_dataset(self, env):
        self.env = env
        with open(self.cfg.dataset_path, "rb") as f:
            dataset = pickle.load(f)

        for idx in range(len(dataset)):
            for step in range(len(dataset[idx][0])):
                if step == 0:
                    tds = [self.to_td(torch.from_numpy(dataset[idx][0][step]))]
                obs = torch.from_numpy(dataset[idx][3][step])
                action = torch.from_numpy(dataset[idx][1][step])
                reward = torch.tensor(dataset[idx][2][step])
                # episode_td = self.to_td(obs, action, reward)
                # pdb.set_trace()
                tds.append(self.to_td(obs, action, reward))
            self.add(torch.cat(tds))


    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
        ), batch_size=(1,))
        return td

    def segment_trajectories(self, observations, actions, rewards, next_observations, terminals, timeouts):
        trajectories = []
        start_idx = 0

        # Iterate through the terminals and timeouts arrays to find episode boundaries
        for i, (done, timeout) in enumerate(zip(terminals, timeouts)):
            if done or timeout:
                trajectory = {
                    'observations': observations[start_idx:i + 1],
                    'actions': actions[start_idx:i + 1],
                    'rewards': rewards[start_idx:i + 1],
                    'next_observations': next_observations[start_idx:i + 1],
                    'terminals': terminals[start_idx:i + 1],
                    'timeouts': timeouts[start_idx:i + 1]
                }
                trajectories.append(trajectory)
                start_idx = i + 1

        # Add the last trajectory if the episode doesn't end with a terminal state or timeout
        if start_idx < len(terminals):
            trajectory = {
                'observations': observations[start_idx:],
                'actions': actions[start_idx:],
                'rewards': rewards[start_idx:],
                'next_observations': next_observations[start_idx:],
                'terminals': terminals[start_idx:],
                'timeouts': timeouts[start_idx:]
            }
            trajectories.append(trajectory)

        return trajectories

