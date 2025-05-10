import pdb

import numpy as np
import wandb
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from iq import iq_loss
import matplotlib.pyplot as plt


class TDMPC2:
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._Qs_original.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		], lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)

		if self.cfg.load_pretrained:
			self.load_pretrained()


	def load_pretrained(self):
		model_state_dict = torch.load("checkpt.pt")
		self.model.load_state_dict(model_state_dict["model"])
		

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.
		
		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.
		
		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.
		
		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		z = self.model.encode(obs, task)
		if self.cfg.mpc:
			a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
		else:
			a = self.model.pi(z, task)[int(not eval_mode)][0]
		return a.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G += discount * reward
			discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
		return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

	def _estimate_value_iq(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			Q = self.model.Q_original(z, actions[t], task, return_type='min')
			_, pi, log_pi, _ = self.model.pi(z, task) 
			z = self.model.next(z, actions[t], task)
			_, pi_next, log_pi_next, _ = self.model.pi(z, task) 
			value_next = self.model.Q_original(z, pi_next, task, return_type='min') - self.cfg.entropy_coef * log_pi_next
			reward = Q - self.discount * (value_next)
			# pdb.set_trace()
			G += discount * (reward - self.cfg.entropy_coef * log_pi)
			discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
		_, _, log_pi_terminal, _ = self.model.pi(z, task) 
		return G + discount * (self.model.Q_original(z, self.model.pi(z, task)[1], task, return_type='avg') - self.cfg.entropy_coef * log_pi_terminal)

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.
		
		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""		
		# Sample policy trajectories
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
	
		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
				.clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value_iq(z, actions, task).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		a, std = actions[0], std[0]
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		return a.clamp_(-1, 1)
		
	def update_pi(self, zs, task, action_expert=None):
		"""
		Update policy using a sequence of latent states.
		
		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs, task)
		qs = self.model.Q_original(zs, pis, task, return_type='min')

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		log_pi_expert = self.model.log_prob_expert(zs[:-1, self.cfg.batch_size:, :], task, action_expert)
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
		bc_reg = -((log_pi_expert).mean(dim=(1,2)) * rho[:-1]).mean()
		pi_loss += self.cfg.reg_coef * bc_reg
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item(), log_pis

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)


	def update_iq(self, buffer, expert_buffer, expert_only=False):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, task = buffer.sample()
		# obs, action, reward, task = expert_buffer.sample()
		obs_expert, action_expert, reward_expert, _ = expert_buffer.sample()

		obs = torch.cat((obs, obs_expert), dim=1)
		action = torch.cat((action, action_expert), dim=1)
		is_expert = torch.cat(
			(torch.zeros_like(reward).to(self.device), torch.ones_like(reward_expert).to(self.device)), dim=1)
		reward = torch.cat((reward, reward_expert), dim=1)

		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
		
		z_truth = self.model.encode(obs, task)

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon + 1, self.cfg.batch_size * 2, self.cfg.latent_dim, device=self.device)
		
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho ** t
			zs[t + 1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q_original(zs[:-1], action, task, return_type='loss')

		# Compute losses
		value_loss = 0
		rewards = 0
		value = self.getV(zs, task)
		with torch.no_grad():
			next_value = self.get_target_V(next_z, task)
		reward_whole = []
		if self.cfg.use_v0:
			v0 = self.getV(zs[0, self.cfg.batch_size:].unsqueeze(0), task).mean()
		else:
			v0 = None
		for t in range(self.cfg.horizon):
			reward_step = None
			
			for q in range(self.cfg.num_q):
				if self.cfg.use_grad_pen:
					grad_pen = self.model.grad_pen(_zs[t][self.cfg.batch_size:, :], action[t][self.cfg.batch_size:, :], _zs[t][:self.cfg.batch_size, :], action[t][:self.cfg.batch_size, :], task)
					iq_loss_value, _, loss_dict = iq_loss(self, qs[q][t], value[t], next_value[t], z_truth[t], gamma=self.discount,  expert_only=expert_only, is_expert=is_expert[t], batch_size=self.cfg.batch_size, v0=v0, grad_pen=grad_pen)
				else:
					iq_loss_value, _, loss_dict = iq_loss(self, qs[q][t], value[t], next_value[t], z_truth[t], gamma=self.discount,  expert_only=expert_only, is_expert=is_expert[t], batch_size=self.cfg.batch_size, v0=v0)
				value_loss += iq_loss_value * self.cfg.rho ** t

			with torch.no_grad():
				Q = self.model.Q_original(z_truth[t], action[t], task, return_type='min')
				_, pi_next, log_pi_next, _ = self.model.pi(next_z[t], task)  # TODO: figure out if we can use log_pi here when doing planning
				reward_pred = Q - self.discount * (self.model.Q_original(next_z[t], pi_next, task, return_type='min') - self.cfg.entropy_coef * log_pi_next) # TODO: check if here can use pi_next or actions[t+1]

			if reward_step is None:
				reward_step = reward_pred
			else:
				reward_step = torch.cat((reward_step, reward_pred), dim=1)
			reward_step = reward_step.mean(dim=1)
			reward_whole.append(reward_step)
		reward_whole = torch.stack(reward_whole)
		reward_corr = pearson_correlation(reward, reward_whole).nan_to_num_(0)
		consistency_loss *= (1 / self.cfg.horizon)
		value_loss *= (1 / (self.cfg.horizon * self.cfg.num_q))
		total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		# Update policy
		pi_loss, log_pis = self.update_pi(zs.detach(), task, action_expert)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
			"Q_value": float(qs.mean().item()),
			"Q_diff": float(qs[:, :, self.cfg.batch_size:, :].mean() - qs[:, :, :self.cfg.batch_size, :].mean()),
			"z_diff": float((zs[:, self.cfg.batch_size:, :] - zs[:, :self.cfg.batch_size, :]).mean())
		}

	def getV(self, z, task):
		_, action, log_pi, _ = self.model.pi(z, task)
		current_Q = self.model.Q_original(z, action, task, return_type='min')
		current_V = current_Q - self.cfg.entropy_coef * log_pi
		# pdb.set_trace()
		return current_V

	def get_target_V(self, z, task):
		_, action, log_pi, _ = self.model.pi(z, task)
		current_Q = self.model.Q_original(z, action, task, return_type='min', target=True)
		current_V = current_Q - self.cfg.entropy_coef * log_pi
		# pdb.set_trace()
		return current_V

	# def getV(self, z, actions, task):
	# 	_, action, log_pi, _ = self.model.pi(z, task)
	# 	current_Q = self.model.Q(z, action, task, return_type='loss')
	# 	current_V = current_Q - self.cfg.entropy_coef * log_pi
	# 	# pdb.set_trace()
	# 	return current_V

	def update_pi_bc(self, expert_buffer):
		"""
		Update policy using a sequence of latent states.
		
		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""

		obs_expert, action_expert, _, task = expert_buffer.sample()

		zs = self.model.encode(obs_expert, task)[:-1]

		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		log_pis = self.model.log_prob_expert(zs, task, action_expert)
		# qs = self.model.Q(zs, pis, task, return_type='min')
		# print("Q value:", qs.mean())
		# print("log pis:", log_pis.mean())
		# self.scale.update(qs[0])
		# qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(zs), device=self.device))
		# pdb.set_trace()
		pi_loss = -((log_pis).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item()


def pearson_correlation(tensor1, tensor2):
	# Flatten the tensors to remove the last dimension
	tensor1 = tensor1.squeeze(-1)  # Shape: [time_horizon, batch_size]
	# tensor2 = tensor2.squeeze(-1)  # Shape: [time_horizon, batch_size]

	# Compute the means of the tensors along the time_horizon dimension
	mean1 = tensor1.mean(dim=0)  # Shape: [batch_size]
	mean2 = tensor2.mean(dim=0)  # Shape: [batch_size]

	# Center the tensors by subtracting their means
	centered1 = tensor1 - mean1  # Shape: [time_horizon, batch_size]
	centered2 = tensor2 - mean2  # Shape: [time_horizon, batch_size]

	# Compute the covariance between the centered tensors
	covariance = (centered1 * centered2).sum(dim=0)  # Shape: [batch_size]

	# Compute the standard deviations of the centered tensors
	std1 = centered1.pow(2).sum(dim=0).sqrt()  # Shape: [batch_size]
	std2 = centered2.pow(2).sum(dim=0).sqrt()  # Shape: [batch_size]

	# Compute the Pearson correlation
	pearson_corr = covariance / (std1 * std2)  # Shape: [batch_size]

	return pearson_corr
