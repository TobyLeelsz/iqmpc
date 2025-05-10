from copy import deepcopy

import numpy as np
import torch
import pdb
import torch.nn as nn

from common import layers, math, init
from torch.autograd import Variable, grad



class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)

		self._Qs_original = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1, dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._Qs_original.params[-2]])
		self._target_Qs = deepcopy(self._Qs_original).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
		
	def to(self, *args, **kwargs):
		"""
		Overriding `to` method to also move additional tensors to device.
		"""
		super().to(*args, **kwargs)
		if self.cfg.multitask:
			self._action_masks = self._action_masks.to(*args, **kwargs)
		self.log_std_min = self.log_std_min.to(*args, **kwargs)
		self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
		return self
	
	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def track_q_grad(self, mode=True):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		This method also enables/disables gradients for task embeddings.
		"""
		for p in self._Qs_original.parameters():
			p.requires_grad_(mode)
		if self.cfg.multitask:
			for p in self._task_emb.parameters():
				p.requires_grad_(mode)

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		with torch.no_grad():
			# for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
			# 	p_target.data.lerp_(p.data, self.cfg.tau)
			for p, p_target in zip(self._Qs_original.parameters(), self._target_Qs.parameters()):
				p_target.data.lerp_(p.data, self.cfg.tau)
	
	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)
		# return obs

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mu)

		if self.cfg.multitask: # Mask out unused action dimensions
			mu = mu * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
		pi = mu + eps * log_std.exp()
		mu, pi, log_pi = math.squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std

	def log_prob_expert(self, z, task, expert_action):
		"""
		Computes the log-probability of an expert action under the policy.
		
		Args:
		- z: The latent variable or state input to the policy.
		- task: The task identifier (used for multitask scenarios).
		- expert_action: The action taken by the expert.
		
		Returns:
		- log_prob: The log-probability of the expert action under the policy.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)

		if self.cfg.multitask: # Mask out unused action dimensions
			mu = mu * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			expert_action = expert_action * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else:
			action_dims = None

		eps = (expert_action - mu)

		# Compute log probability of expert action
		log_prob = math.gaussian_logprob(eps, log_std, size=action_dims)

		# If necessary, you can squash the actions and log_prob
		_, _, log_prob = math.squash(mu, expert_action, log_prob)

		return log_prob

	def Q(self, z, a, task, return_type='min', target=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all', "loss"}

		if self.cfg.multitask:
			z = self.task_emb(z, task)
			
		z = torch.cat([z, a], dim=-1)
		out = (self._target_Qs if target else self._Qs)(z)

		if return_type == 'all':
			return out

		if return_type == "loss":
			# pdb.set_trace()
			return torch.sigmoid(math.two_hot_inv(out, self.cfg))

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
		
		return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2

	def Q_original(self, z, a, task, return_type='min', target=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all', "loss"}

		if self.cfg.multitask:
			z = self.task_emb(z, task)
			
		z = torch.cat([z, a], dim=-1)
		out = (self._target_Qs if target else self._Qs_original)(z)

		if return_type == 'all':
			return out

		if return_type == "loss":
			# pdb.set_trace()
			return out

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		
		return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2

	def grad_pen(self, z1, action1, z2, action2, task, lambda_=1):
		expert_data = torch.cat([z1, action1], 1)
		policy_data = torch.cat([z2, action2], 1)

		alpha = torch.rand(expert_data.size()[0], 1)
		alpha = alpha.expand_as(expert_data).to(expert_data.device)

		interpolated = alpha * expert_data + (1 - alpha) * policy_data
		interpolated = Variable(interpolated, requires_grad=True)

		interpolated_state, interpolated_action = torch.split(
			interpolated, [z1.shape[1], action1.shape[1]], dim=1)
		q = self.Q_original(interpolated_state, interpolated_action, task, "avg")
		ones = torch.ones(q.size()).to(policy_data.device)
		gradient = grad(
			outputs=q,
			inputs=interpolated,
			grad_outputs=[ones, ones],
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
		return grad_pen
