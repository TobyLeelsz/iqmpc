class Trainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, env, agent, buffer, expert_buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.expert_buffer = expert_buffer
		self.buffer = buffer
		self.logger = logger
		print('Architecture:', self.agent.model)
		print("Learnable parameters: {:,}".format(self.agent.model.total_params))

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		raise NotImplementedError

	def train(self):
		"""Train a TD-MPC2 agent."""
		raise NotImplementedError
