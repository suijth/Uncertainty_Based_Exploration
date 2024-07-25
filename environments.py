import numpy as np 
import time

class chain:

	def __init__(self, length, ordered = 1, stocastic = 1, default_state = 1):
		self.length = length
		self.action_size = 2
		self.state_size = 1
		self.dir = np.ones(length+1)
		np.random.seed(int(time.time()))
		if not ordered:
			self.dir = np.random.randint(2, size = length+1)
		print(self.dir)
		self.default_state = default_state
		self.state = default_state
		self.l_reward = 1/100
		self.r_reward = 1
		self.f_reward = 0
		self.b_reward = 0
		self.steps = 0
		self.max_steps = length + 9
		self.var = 0
		self.stocastic = stocastic

	def reset(self):
		self.state = self.default_state
		self.steps = 0
		return self.state

	def step(self, action):
		ran = 0
		if self.stocastic: 
			ran = np.random.rand()
		if self.steps == self.max_steps:
			return self.reset(),0,1,0
		if action == self.dir[self.state] and ran < 0.5:

			if self.state == self.length:
				r = self.r_reward
			else:
				if self.var == 0:
					r = self.f_reward
				else:
					r = np.random.normal(self.f_reward,self.var)
				self.state += 1
			# return self.reset(),0,1


		elif not action == self.dir[self.state]:
			if self.state == 0:
				r = self.l_reward
			else:
				if self.var == 0:
					r = self.b_reward
				else:
					r = np.random.normal(self.b_reward,self.var)
				self.state -= 1
		else:
			return 0,0,0,0

		self.steps += 1
		return self.state, r, 0, 0



