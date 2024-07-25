import model
import environments
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import time

class replay:
	def __init__(self,max_size = 1000):
		self.max_size = max_size
		self.buffer = []

	def add(self, element):
		if len(self.buffer) == self.max_size:
			del self.buffer[0]
		self.buffer.append(element)

	def sample(self, siz):
		indices = np.random.randint(len(self.buffer), size = siz)
		samp = []
		for i in indices:
			samp.append(self.buffer[i])
		return samp

class ddqn:

	def __init__(self, env, state_size ,action_size, low = 0 ,high = 50, exploration = 'episilon'):
		self.env = env
		self.state_size = state_size
		self.action_size = action_size

		self.episilon = 1
		self.decay_steps = 15000 
		self.min_episilon = 0.05
		self.sub_episilon = (self.episilon - self.min_episilon)/self.decay_steps
		self.gamma = 0.99
		self.bins = 51

		self.low = low
		self.high = high 

		self.lr = 0.01
		self.model = model.buildNetwork(self.state_size, self.action_size, bins = self.bins,lr = self.lr)
		self.target_model = model.buildNetwork(self.state_size, self.action_size, bins = self.bins, lr = self.lr)

		self.replay_max_size = 20000
		self.replay = replay(self.replay_max_size)
		
		self.batch_size = 32
		self.sample_size = 32
		self.test = 0
	
		self.update_target = 300
		self.total_episodes = 5000
		self.train_per_episodes = 1
	
		self.values = [self.low]
		self.diff = (self.high - self.low ) /(self.bins-1)
		for i in range(self.bins-1):
			self.values.append(self.values[-1] + self.diff)
		self.values = np.array(self.values)

	def expectation(self, probs):
		# ans = 0
		# for i in range(self.bins):
		# 	ans += probs[i]*self.values[i]
		return np.dot(probs, self.values)

	def processData(self, data):
		returns = []
		inp = []
		next_state =  []
		# print(data[0])
		for i in range(len(data)):
			returns.append(data[i][2])
			inp.append(data[i][0])
			next_state.append(data[i][3])
		###
		# print(next_state)
		z = self.model.predict(np.array(next_state)) 
		z_ = self.target_model.predict(np.array(next_state))

		# optimal_action = []
		# z_concat = np.vstack(z)
		# q = np.sum(np.multiply(z_concat, np.array(self.values)), axis=1)
		# q = q.reshape((len(data), self.action_size), order='F')
		q = np.dot(z, self.values).T
		optimal_action = np.argmax(q, axis=1)
		return_dist = self.model.predict(np.array(inp))
		for i in range(len(data)):
			return_dist[data[i][1]][i] = self.project(data[i][2], z_[optimal_action[i]][i], data[i][4])
		return np.array(inp), return_dist


	def project(self, reward, probabs, done = 0):
		m_prob = np.zeros(self.bins)
		if done:
			Tz = min(self.high, max(self.low, reward))
			bj = (Tz - self.low) / self.diff + 0.000001
			m_l, m_u = math.floor(bj), math.ceil(bj)
			m_prob[int(m_l)] += (m_u - bj)
			m_prob[int(m_u)] += (bj - m_l)
		else:
			for j in range(self.bins):
				Tz = min(self.high, max(self.low, reward + self.gamma * self.values[j]))
				bj = (Tz - self.low) / self.diff
				m_l, m_u = math.floor(bj), math.ceil(bj)
				if m_l == m_u:
					m_prob[int(m_l)] += probabs[j]
				else:
					m_prob[int(m_l)] += probabs[j] * (m_u - bj)
					m_prob[int(m_u)] += probabs[j] * (bj - m_l)
		s = np.sum(m_prob)
		m_prob /= s
		return m_prob

	def train(self):
		samp_size = min(len(self.replay.buffer), self.sample_size)
		data = self.replay.sample(samp_size)
		x,y = self.processData(data)
		his = self.model.fit(x, y, verbose = 0)
		# print()
		return his.history['loss'][0]

	def expectation(self,state):
		z = self.model.predict(np.array([state]))
		z = np.vstack(z)
		returns = np.dot(z,self.values)
		return returns

	def optimal_action(self, state):
		# print(state)
		z = self.model.predict(np.array([state]))
		z = np.vstack(z)
		returns = np.dot(z,self.values)
		return np.argmax(returns)


	def draw_graph(self,state,action = 1):
		probs = self.model.predict(np.array([state]))
		probs = np.vstack(probs)
		plt.plot(self.values, probs[action])
		plt.draw()
		plt.pause(0.001)
		# plt.cla()
	def choose_action(self, state, Type = 'episilon'):
		if Type == 'episilon':
			if np.random.rand() < self.episilon and not self.test:
				action = np.random.randint(self.action_size)
			else:
				action = self.optimal_action(state)
			return action
		else:
			probs = self.model.predict(np.array([state]))
			probs = np.vstack(probs)
			rand = np.random.rand(self.action_size)
			su = np.zeros(self.action_size) 
			returns = np.zeros(self.action_size)
			# plt.plot(self.values, probs[1])
			# plt.draw()
			# plt.pause(0.001)
			# plt.cla()
			# time.sleep(5)
			for i in range(self.action_size):
				for j in range(self.bins):
					su[i]+=probs[i][j]
					if rand[i] < su[i]:
						returns[i] = self.values[j]
						break
			# print()
			return np.argmax(returns)

	def decay_episilon(self):
		if self.episilon > self.min_episilon:
			self.episilon -= self.sub_episilon

	def load_model(self,name = 'cartEpi.h5'):
		self.model = model.load_model(name)
		self.target_model = model.load_model(name)


	def run(self):
		steps = 0
		rewards = []
		plotx = []
		ploty = []
		for i in range(self.total_episodes):
			curr_state = self.env.reset()
			# print(done)
			total_reward = 0
			done = 0
			loss = []
			while not done:
				steps+=1
				action = self.choose_action(curr_state, Type = 'ad')
				# print(action)
				next_state,r,done,_ = self.env.step(action)
				# print(next_state)
				total_reward += r
				# r -= 5	 * abs(next_state[0])
				if done and total_reward < 199:
					r = -15
				self.replay.add((curr_state, action, r, next_state, done))
				curr_state = next_state.copy()
				if len(self.replay.buffer) > self.sample_size and not self.test:
					l = self.train()
					loss.append(l)
					self.decay_episilon()
				if steps%self.update_target == 0:
					self.target_model.set_weights(self.model.get_weights())

			rewards.append(total_reward)
				# if i > 400:
				# 	self.test = 1
			if i%10 == 0:
				print(self.episilon)
				avg = np.sum(rewards[-10:])/10
				plotx.append(i)
				ploty.append(avg)
				# plt.draw()
				# plt.pause(0.00001)
			if i%50 == 0:
				rewards.clear()
				plt.savefig('sampling.png')
			if i%50 == 0:
				plt.cla()
				plt.plot(plotx,ploty)
				plt.draw()
				plt.pause(0.00001)

			# if i%self.train_per_episodes == 0 and i != 0:
			# 	self.train()
			print(i, total_reward, sum(loss)/(len(loss)+1))	
		self.model.save('cartEpi.h5')

def cartpole():
	game = gym.make('CartPole-v0')
	agent = ddqn(game, state_size = 4, action_size = 2, low = -25, high = 25)
	agent.run()
	
def mountaincar():
	game = gym.make('MountainCar-v0')
	agent = ddqn(game, state_size = 2, action_size = 3, low = -200, high = 0)
	agent.run()

def lunar_lander():
	game = gym.make('LunarLander-v2')
	agent = ddqn(game, state_size = 8, action_size = 4, low = -200, high = 100)
	agent.run()	

# cartpole()
game = gym.make('CartPole-v0')
agent = ddqn(game, state_size = 4, action_size = 2, low = 0, high = 200)
# agent.run()
# lunar_lander()