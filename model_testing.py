import random
import gym
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from gym import wrappers
from pong.preprocessing import Preprocessor
from pong.sequence_of_frames import Sequence
from PIL import Image


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.pass_frames = 16
        self.episodes = 2000
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.01  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.lastk = 4

        self.exp_replay = False
        self.batch_size = 4
        if self.exp_replay == False:
            self.memory = deque(maxlen=4)
        else: self.memory = deque(maxlen=64)

        self.trained_model_path = "trained_models/DQN_first_model_conv2d_32_64_64"
        self.model_path = "models/DQN_model_e_"
        self.save_model_freq = 1
        self.log_path = "logs/dqn_log.txt"
        self.loger_mode = True

        self.model = self._build_model()

        self.frame_sequence = Sequence(80, 80, self.lastk)
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(input_shape=(80,80,self.lastk, ), filters=16, kernel_size=(8,8), strides = 4, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(4,4), strides = 2, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides = 1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def load(self):
        self.model.load_weights(self.trained_model_path)

    def save(self, name):
        self.model.save_weights(name)

    def remember(self, state, action, reward, next_state, done):
        network_current = np.copy(self.frame_sequence.sequence.reshape \
            (1, self.state_size[0], self.state_size[1], self.lastk))

        self.frame_sequence.add_frame(state)

        network_next = np.copy(self.frame_sequence.sequence.reshape \
            (1, self.state_size[0], self.state_size[1], self.lastk))

        self.memory.append((network_current, action, reward, network_next, done))

    def update(self):
        network_current, action, reward, network_next, done = self.memory[0]
        target = reward
        if not done:
            target = (reward + self.gamma *
                      np.amax(self.model.predict(network_next)[0]))
        target_f = self.model.predict(network_current)
        target_f[0][action] = target
        self.model.fit(network_current, target_f, epochs=1, verbose=0)

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for network_current, action, reward, network_next, done in minibatch:
            #data = np.zeros((80, 80, 3), dtype=np.uint8)
            #data[:,:,0] = network_current[0,:, :, 2]
            #data[:,:,1] = network_current[0,:, :, 2]
            #img = Image.fromarray(data, 'RGB')
            #img.show()
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(network_next)[0]))
            target_f = self.model.predict(network_current)
            target_f[0][action] = target
            self.model.fit(network_current, target_f, epochs=1, verbose=0)


def main():
    env = gym.make('Pong-v0')

    observation_size = env.observation_space.shape
    preprocessor = Preprocessor(observation_size)

    state_size = Preprocessor.preprocessed_observation_size
    action_size = int(env.action_space.n / 2)

    agent = DQNAgent(state_size, action_size)
    agent.load()

    for e in range(agent.episodes):
        observation = env.reset()
        state = preprocessor.preprocess_observation(observation)
        episode_len = 0
        reward_sum = 0


        #init sequence with repeated first frame
        for i in range(agent.lastk):
            agent.frame_sequence.add_frame(state)

        #one episode
        while True:
            episode_len+=1
            env.render()

            network_input = agent.frame_sequence.sequence.reshape(1, state_size[0], state_size[1], agent.lastk)
            action = agent.act(network_input)
            en_action = action
            if action == 1: en_action = 3
            next_observation, reward, done, _ = env.step(en_action)
            next_state = preprocessor.preprocess_observation(next_observation)

            state = next_state

            reward_sum += reward

            if done:
                break

if __name__ == '__main__':
    main()