import model
import environments
import chain
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.stats import rv_discrete
from gym.envs.registration import register
import FrozenLake
import os

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

register(
    id='FrozenLakeNotSlippery-v1',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

window = 2
np.random.seed(int(time.time()))


class LHShash:
    def __init__(self, length, numb=100):
        self.l = length
        self.x = np.random.randn(length)
        self.b = 1/np.random.randint(numb)
        self.n = numb

    def val(self, vec):
        return vec[0]
        v = np.dot(vec, self.x) + self.b
        v *= self.n
        return int(max(-window*self.n, min(window*self.n, np.floor(v))) + window*self.n)


class replay:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []

    def add(self, element):
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append(element)

    def add_list(self, my_list):
        for el in my_list:
            self.add(el)

    def sample(self, siz):
        indices = np.random.randint(len(self.buffer), size=siz)
        samp = []
        for i in indices:
            samp.append(self.buffer[i])
        return samp


def argmax_tiebreak(values):
    m = np.max(values)
    return np.random.choice(np.flatnonzero(values == m))


class ddqn:

    def __init__(self, env, state_size, action_size, low=0, high=50, exploration='episilon', discrete=0):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.discrete = discrete

        self.episilon = 0.1
        self.decay_steps = 15000
        self.min_episilon = 0.1
        self.sub_episilon = (
            self.episilon - self.min_episilon)/self.decay_steps
        self.numb_samples = 1
        self.gamma = 0.99
        self.bins = 51

        self.low = low
        self.high = high

        self.lr = 0.001
        self.model = []
        self.target_model = []
        self.p_keep = 1.0
        for i in range(action_size):
            self.model.append(model.buildSingleNetwork(self.state_size, self.action_size,
                              bins=self.bins, lr=self.lr, p_keep=self.p_keep, discrete=self.discrete))
            self.target_model.append(model.buildSingleNetwork(
                self.state_size, self.action_size, bins=self.bins, lr=self.lr, p_keep=self.p_keep, discrete=self.discrete))

        self.replay_max_size = 100000
        self.replay = replay(self.replay_max_size)

        self.batch_size = 32
        self.sample_size = 32
        self.test = 0

        self.update_target = 50
        self.total_episodes = 3000
        self.train_per_episodes = 1
        self.explore = 10
        self.reward_avg_count = 30

        self.values = [self.low]
        self.diff = (self.high - self.low) / (self.bins-1)
        for i in range(self.bins-1):
            self.values.append(self.values[-1] + self.diff)
        self.values = np.array(self.values)

        if discrete:
            l = self.state_size
        else:
            l = 100000
        self.count = np.zeros((l, self.action_size), dtype=int)
        self.hFun = LHShash(self.state_size)

    def processData(self, data):
        returns = []
        inp = [[] for _ in range(self.action_size)]
        next_state = []
        # print(data[0])
        for i in range(len(data)):
            returns.append(data[i][2])
            inp[data[i][1]].append(data[i][0])
            next_state.append(data[i][3])
        ###
        # print(next_state)
        z = [m.predict(np.array(next_state)) for m in self.model]
        z_ = [m.predict(np.array(next_state)) for m in self.target_model]

        # optimal_action = []
        # z_concat = np.vstack(z)
        # q = np.sum(np.multiply(z_concat, np.array(self.values)), axis=1)
        # q = q.reshape((len(data), self.action_size), order='F')
        q = np.dot(z, self.values).T
        optimal_action = np.argmax(q, axis=1)
        return_dist = [[] for _ in range(self.action_size)]

        for i in range(len(data)):
            return_dist[data[i][1]].append(self.project(
                data[i][2], z_[optimal_action[i]][i], data[i][4]))
        return inp, return_dist

    def project(self, reward, probabs, done=0):
        m_prob = np.zeros(self.bins)
        if done:
            Tz = min(self.high, max(self.low, reward))
            bj = (Tz - self.low) / self.diff
            m_l, m_u = math.floor(bj), math.ceil(bj)
            if m_l == m_u:
                m_prob[int(m_l)] = 1
            else:
                m_prob[int(m_l)] += (m_u - bj)
                m_prob[int(m_u)] += (bj - m_l)
        else:
            for j in range(self.bins):
                Tz = min(self.high, max(self.low, reward +
                         self.gamma * self.values[j]))
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

    def train(self, data):
        # print(len(data))
        x, y = self.processData(data)
        l = 0
        for i in range(self.action_size):
            if len(x[i]) != 0:
                l += self.model[i].fit(np.array(x[i]),
                                       np.array(y[i]), verbose=0).history['loss'][0]
        # print()
        return l

    def expectation(self, state):
        z = [m.predict(np.array([state])) for m in self.model]
        z = np.vstack(z)
        returns = np.dot(z, self.values)
        return returns

    def optimal_action(self, state):
        # print(state)
        z = [m.predict(np.array([state])) for m in self.model]
        z = np.vstack(z)
        returns = np.dot(z, self.values)
        return np.argmax(returns)

    def sample(self, probs, numb):
        return np.random.choice(self.values, p=probs, size=numb)

    def draw_graph(self, state, action=0):
        probs = self.model[action].predict(np.array([state]))[0]
        plt.plot(self.values, probs)
        plt.draw()
        plt.pause(0.001)

    def choose_action(self, state, Type='episilon'):
        if Type == 'episilon' or self.test:
            if np.random.rand() < self.episilon and not self.test:
                action = np.random.randint(self.action_size)
            else:
                action = self.optimal_action(state)
            return action
        elif Type == 'duvn':
            probs = [m.predict(np.array([state])) for m in self.model]
            probs = np.vstack(probs)
            returns = np.zeros(self.action_size)
            for i in range(self.action_size):
                out = self.sample(probs[i], self.numb_samples)
                returns[i] = np.max(out)
            return argmax_tiebreak(returns)
        else:
            probs = [m.predict(np.array([state])) for m in self.model]
            probs = np.vstack(probs)
            returns = np.zeros(self.action_size)
            for i in range(self.action_size):
                out = self.sample(probs[i], min(
                    700, self.count[self.hFun.val(state)][i]) + 1)
                returns[i] = np.mean(out)
            return argmax_tiebreak(returns)

    def decay_episilon(self):
        if self.episilon > self.min_episilon:
            self.episilon -= self.sub_episilon

    def load_model(self, name):
        for i in range(self.action_size):
            self.model[i] = model.load_model(name[i])
            self.target_model[i] = model.load_model(name[i])

    def run_fast(self):
        steps = 0
        rewards = []
        plotx = []
        ploty = []
        curr_data = []
        for i in range(self.total_episodes):
            curr_state = self.env.reset()
            if self.discrete:
                curr_state = np.array([curr_state])
            # curr_state = np.array([curr_state])
            # print(done)
            total_reward = 0
            done = 0
            loss = []
            while not done:
                steps += 1
                action = self.choose_action(curr_state, Type='asd')
                next_state, r, done, info = self.env.step(action)
                # print(next_state)
                # if next_state != 15:
                # 	r = 0.2*np.random.randn()
                if self.discrete:
                    next_state = np.array([next_state])
                if len(self.replay.buffer) > self.sample_size and not self.test and i > self.explore:
                    self.count[self.hFun.val(curr_state)][action] += 1
                # next_state = np.array([next_state])
                total_reward += r
                # print(r)
                # r -= 5	 * abs(next_state[0])
                # if done and len(info) == 0:
                # 	r = 0
                # if len(info) == 1 and done:
                # 	done = 0
                # print(curr_state)
                self.replay.add((curr_state, action, r, next_state, done))
                curr_data.append((curr_state, action, r, next_state, done))
                # if len(info) == 1:
                # 	done = 1
                TEMP = curr_state
                curr_state = next_state.copy()

                # if steps%self.update_target == 0:
                # 	for j in range(self.action_size):
                # 		self.target_model[j].set_weights(self.model[j].get_weights())
            if len(self.replay.buffer) > self.sample_size and not self.test and i > self.explore:
                for j in range(self.action_size):
                    self.target_model[j].set_weights(
                        self.model[j].get_weights())
                samp_size = len(curr_data)
                data = self.replay.sample(samp_size)
                curr_data += data
                l = self.train(curr_data)
                loss.append(l)
                curr_data.clear()
                self.decay_episilon()

            rewards.append(total_reward)
            # if i > 400:
            # 	self.test = 1
            if i % self.reward_avg_count == 0:
                # print(self.episilon)
                avg = np.average(rewards[-self.reward_avg_count:])
                plotx.append(i)
                ploty.append(avg)
                rewards.clear()

            if i % 50 == 0:
                plt.cla()
                plt.plot(plotx, ploty)
                plt.savefig('sampling.png')
                # plt.draw()
                # plt.pause(0.00001)

            # if i%self.train_per_episodes == 0 and i != 0:
            # 	self.train()
            print(i, "{:.2f}".format(total_reward),
                  sum(loss)/(len(loss)+1), TEMP[0])
            if i % 2000 == 0:
                for m in self.model:
                    m.save('action' + str(self.model.index(m)) + '.h5')
                self.plotx, self.ploty = plotx, ploty
            for m in self.model:
                m.save('action' + str(self.model.index(m)) + '.h5')
            self.plotx, self.ploty = plotx, ploty

    def run(self):
        steps = 0
        rewards = []
        plotx = []
        ploty = []
        curr_data = []
        for i in range(self.total_episodes):
            curr_state = self.env.reset()
            if self.discrete:
                curr_state = np.array([curr_state])
            # curr_state = np.array([curr_state])
            # print(done)
            total_reward = 0
            done = 0
            loss = []
            while not done:
                steps += 1
                action = self.choose_action(curr_state, Type='adm')
                next_state, r, done, info = self.env.step(action)
                if self.discrete:
                    next_state = np.array([next_state])
                # next_state = np.array([next_state])
                # if not done and next_state != 15:
                # 	r = 0.2*np.random.randn()
                if len(self.replay.buffer) > self.sample_size and not self.test and i > self.explore:
                    self.count[self.hFun.val(curr_state)][action] += 1
                # next_state = np.array([next_state])
                total_reward += r
                # r -= 5	 * abs(next_state[0])
                # if done and len(info) == 0:
                # 	r = 0
                # if len(info) == 1 and done:
                # 	done = 0
                # print(curr_state)
                self.replay.add((curr_state, action, r, next_state, done))
                curr_data.append((curr_state, action, r, next_state, done))
                # if len(info) == 1:
                # 	done = 1
                curr_state = next_state.copy()
            # 	if len(curr_data) == self.sample_size and not self.test and i > self.explore:
            # 		samp_size = len(curr_data)
            # 		data = self.replay.sample(samp_size)
            # 		temp_curr_data = curr_data.copy()
            # 		curr_data+=data
            # 		l = self.train(curr_data)
            # 		self.replay.add_list(temp_curr_data)
            # 		loss.append(l)
            # 		curr_data.clear()
            # 		self.decay_episilon()
            # if i < self.explore:
            # 	self.replay.add_list(curr_data)
            # 	curr_data.clear()
                if len(self.replay.buffer) > self.sample_size and not self.test and i > self.explore:
                    samp_size = len(curr_data)
                    data = self.replay.sample(self.sample_size)
                    # curr_data += data
                    l = self.train(data)
                    loss.append(l)
                    # curr_data.clear()
                    self.decay_episilon()
                if steps % self.update_target == 0:
                    for j in range(self.action_size):
                        self.target_model[j].set_weights(
                            self.model[j].get_weights())

            rewards.append(total_reward)
            # if i > 400:
            # 	self.test = 1
            if i % self.reward_avg_count == 0:
                # print(self.episilon)
                avg = np.average(rewards[-self.reward_avg_count:])
                plotx.append(i)
                ploty.append(avg)
                rewards.clear()

            if i % 50 == 0:
                plt.cla()
                plt.plot(plotx, ploty)
                plt.savefig('sampling.png')
                # plt.draw()
                # plt.pause(0.00001)

            # if i%self.train_per_episodes == 0 and i != 0:
            # 	self.train()
            print(i, total_reward, sum(loss)/(len(loss)+1))
            if i % 2000 == 0:
                for m in self.model:
                    m.save('action' + str(self.model.index(m)) + '.h5')
                self.plotx, self.ploty = plotx, ploty
        for m in self.model:
            m.save('action' + str(self.model.index(m)) + '.h5')
        self.plotx, self.ploty = plotx, ploty


def multiple_runs(name, numb):
    cur_dir = 'results/' + name
    try:
        os.makedirs(cur_dir)
    except:
        pass
    game = FrozenLake.FrozenLakeEnv(map_name="8x8")
    all_y = []
    for i in range(numb):
        agent = ddqn(game, state_size=64, action_size=4,
                     low=-1, high=1, discrete=1)
        agent.run_fast()
        x, y = agent.plotx, agent.ploty
        all_y.append(y)
        with open(cur_dir + '/x_run' + str(i) + '.txt', "w") as outfile:
            outfile.write("\n".join(str(item) for item in x))
        with open(cur_dir + '/y_run' + str(i) + '.txt', "w") as outfile:
            outfile.write("\n".join(str(item) for item in y))
    avgy = np.average(np.array(all_y), axis=0)
    with open(cur_dir + '/avg_y' + '.txt', "w") as outfile:
        outfile.write("\n".join(str(item) for item in avgy))
    std = np.std(all_y, axis=0)
    plt.cla()
    plt.plot(x, avgy, color='royalblue')
    plt.fill_between(x, avgy - std, avgy + std, alpha=0.25, color='royalblue')
    plt.savefig(cur_dir + '/plot.png')


def multiple_runs_chain(name, numb):
    cur_dir = 'results/' + name
    try:
        os.makedirs(cur_dir)
    except:
        pass
    game = chain.Chain(25)
    all_y = []
    for i in range(numb):
        agent = ddqn(game, state_size=25, action_size=2,
                     low=0, high=1, discrete=1)
        agent.run_fast()
        x, y = agent.plotx, agent.ploty
        all_y.append(y)
        with open(cur_dir + '/x_run' + str(i) + '.txt', "w") as outfile:
            outfile.write("\n".join(str(item) for item in x))
        with open(cur_dir + '/y_run' + str(i) + '.txt', "w") as outfile:
            outfile.write("\n".join(str(item) for item in y))
    avgy = np.average(np.array(all_y), axis=0)
    std = np.std(all_y, axis=0)
    with open(cur_dir + '/avg_y' + '.txt', "w") as outfile:
        outfile.write("\n".join(str(item) for item in avgy))
    with open(cur_dir + '/sd_y' + '.txt', "w") as outfile:
        outfile.write("\n".join(str(item) for item in avgy))
    plt.cla()
    plt.plot(x, avgy, color='royalblue')
    plt.fill_between(x, avgy - std, avgy + std, alpha=0.25, color='royalblue')
    plt.savefig(cur_dir + '/plot.png')


def cartpole():
    game = gym.make('CartPole-v0')
    agent = ddqn(game, state_size=4, action_size=2, low=-25, high=25)
    agent.run()


def mountaincar():
    game = gym.make('MountainCar-v0')
    agent = ddqn(game, state_size=2, action_size=3, low=-200, high=0)
    agent.run()


def lunar_lander():
    game = gym.make('LunarLander-v2')
    agent = ddqn(game, state_size=8, action_size=4, low=-200, high=100)
    agent.run()


def frozenlake():
    game = gym.make("FrozenLakeNotSlippery-v0")
    agent = ddqn(game, state_size=16, action_size=4, low=0, high=1, discrete=1)
    agent.run_fast()


def Chain():
    game = chain.Chain(25)
    agent = ddqn(game, state_size=1, action_size=2, low=-1, high=1, discrete=1)
    agent.run_fast()


if __name__ == "__main__":
    game = environments.chain(6, ordered=0, default_state=0)
    agent = ddqn(game, state_size=7, action_size=2, low=0, high=1, discrete=1)

    # game = gym.make("FrozenLakeNotSlippery-v1")
    # # game = FrozenLake.FrozenLakeEnv(map_name="8x8")
    # agent = ddqn(game, state_size = 64, action_size = 4, low = -1, high = 1,discrete = 1)
    # agent.run_fast()

    # multiple_runs_chain('chain_25', 5)
    # multiple_runs('frozen_stochastic', 5)

    # cartpole()
    # game = gym.make('CartPole-v0')
    # agent = ddqn(game, state_size = 4, action_size = 2, low = 0, high = 50)
    # agent.run()
    # lunar_lander()
    # frozenlake()
    # Chain()
