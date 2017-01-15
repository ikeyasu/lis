import random
import threading

import numpy as np

import chainer
import six
from chainer import functions as F

import six.moves.queue as queue


class Agent(chainer.Chain):
    gamma = 0.99
    initial_epsilon = 1
    epsilon_reduction = 0.001
    min_epsilon = 0.01

    def __init__(self, input_size, output_size, hidden_size):
        initial_w = chainer.initializers.HeNormal(0.01)
        super(Agent, self).__init__(
            fc1=F.Linear(input_size, hidden_size, initialW=initial_w),
            fc2=F.Linear(hidden_size, hidden_size, initialW=initial_w),
            fc3=F.Linear(hidden_size, output_size, initialW=initial_w),
        )
        self.epsilon = self.initial_epsilon
        self.output_size = output_size

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h

    def randomize_action(self, action):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        return action

    def reduce_epsilon(self):
        self.epsilon = (self.epsilon - self.min_epsilon) * (1 - self.epsilon_reduction) + self.min_epsilon

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        return np.asarray(state, dtype=np.float32)


class LinearAgent(Agent):
    gamma = 0.9
    initial_epsilon = 1
    min_epsilon = 0.01
    epsilon_reduction = 0.001

    def __init__(self):
        super(LinearAgent, self).__init__(32, 3, 24)

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        # todo
        return np.asarray(state, dtype=np.float32)


class ExperiencePool(object):
    def __init__(self, size, state_shape):
        self.size = size
        self.states = np.zeros(((size,) + state_shape), dtype=np.float32)
        self.actions = np.zeros((size,), dtype=np.int32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.nexts = np.zeros((size,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, done):
        index = self.pos % self.size
        self.states[index, ...] = state
        self.actions[index] = action
        self.rewards[index] = reward
        if done:
            self.nexts[index] = 0
        else:
            self.nexts[index] = 1
        self.pos += 1

    def available_size(self):
        if self.pos > self.size:
            return self.size - 1
        return self.pos - 1

    def __getitem__(self, index):
        if self.pos < self.size:
            offset = 0
        else:
            offset = self.pos % self.size - self.size
        index += offset
        return self.states[index], self.actions[index], self.rewards[index], self.states[index + 1], self.nexts[index]


class LisAgent(threading.Thread):
    state_queue = queue.Queue()
    action = None

    def __init__(self):
        super(LisAgent, self).__init__()

    @staticmethod
    def update(agent, target_agent, optimizer, ex_pool, batch_size):
        available_size = ex_pool.available_size()
        if available_size < batch_size:
            return
        indices = np.random.permutation(available_size)[:batch_size]
        data = [ex_pool[i] for i in indices]
        state, action, reward, next_state, has_next = zip(*data)
        state = np.asarray(state)
        action = np.asarray(action)
        reward = np.asarray(reward)
        next_state = np.asarray(next_state)
        has_next = np.asarray(has_next)

        q = F.select_item(agent(state), action)
        next_action = np.argmax(agent(next_state).data, axis=1)
        y = reward + agent.gamma * has_next * target_agent(next_state).data[
            (six.moves.range(len(next_action))), next_action]
        loss = F.mean_squared_error(q, y)
        agent.cleargrads()
        loss.backward()
        optimizer.update()

    def retrieve_action(self):
        self.state_queue.join()
        return self.action

    def run(self):
        episode_num = 1000
        episode_length = 1000
        pool_size = 2000
        batch_size = 32
        train_num = 1
        use_double_q = False

        update_count = 0
        update_agent_interval = 100

        agent = LinearAgent()

        if use_double_q:
            target_agent = agent.copy()
        else:
            target_agent = agent
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(agent)
        ex_pool = ExperiencePool(pool_size, (32,))

        for episode in six.moves.range(episode_num):
            raw_state, raw_reward, done = self.state_queue.get(True)
            state = agent.normalize_state(raw_state)
            for t in six.moves.range(episode_length):
                action = np.argmax(agent(np.expand_dims(state, 0)).data)
                action = agent.randomize_action(action)
                self.action = action
                self.state_queue.task_done()

                prev_state = state
                raw_state, raw_reward, done = self.state_queue.get(True)
                reward = agent.adjust_reward(raw_state, raw_reward, done)
                state = agent.normalize_state(raw_state)
                ex_pool.add(prev_state, action, reward, done or t == episode_length - 1)
                for i in six.moves.range(train_num):
                    self.update(agent, target_agent, optimizer, ex_pool, batch_size)
                update_count += 1
                agent.reduce_epsilon()
                if use_double_q and update_count % update_agent_interval == 0:
                    target_agent = agent.copy()
                if done:
                    print('Episode {} finished after {} timesteps'.format(episode + 1, t + 1))
                    self.state_queue.task_done()
                    break
            if not done:
                print('Epsode {} completed'.format(episode + 1))
