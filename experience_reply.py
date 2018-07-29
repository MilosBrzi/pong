import random
import gym
import numpy as np
#import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from gym import wrappers
from pong.preprocessing import Preprocessor

EPISODES = 100


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = deque(maxlen=2000)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(input_shape=(80,80,4, ), filters=16, kernel_size=(8,8), strides = 4, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(4,4), strides = 2, activation='relu'))
        #model.add(Conv2D(filters=64, kernel_size=(3,3), strides = 1, activation='relu'))
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

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    env = gym.make('Pong-v0')

    observation_size = env.observation_space.shape
    preprocessor = Preprocessor(observation_size)

    state_size = Preprocessor.preprocessed_observation_size
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    batch_size = 32

    for e in range(EPISODES):
        reward_sum = 0
        observation = env.reset()
        state = preprocessor.preprocess_observation(observation)
        state = state.reshape(1,state_size[0], state_size[1],1)
        history = None
        while True:
            env.render()
            action = agent.act(state)
            next_observation, reward, done, _ = env.step(action)
            next_state = preprocessor.preprocess_observation(next_observation)
            next_state = next_state.reshape(1,state_size[0], state_size[1],1)

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            reward_sum += reward

            if done:
                print("episode: {}/{}, score: {}, e: {:.2} history: {}"
                      .format(e, EPISODES, reward_sum, agent.epsilon))
                break

            #target = reward + agent.gamma * np.amax(agent.model.predict(next_state)[0])
            #target_f = agent.model.predict(state)
            #target_f[0][action] = target
            #history = agent.model.fit(state, target_f, epochs=1, verbose=0)

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if len(agent.memory) == agent.memory.maxlen/2:
                agent.replay(batch_size)


if __name__ == '__main__':
    main()