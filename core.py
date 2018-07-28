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
        self.episodes = 2000
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.lastk = 4
        self.batch_size = 8

        self.model_path = "models/DQN_model_e_"
        self.save_model_freq = 50
        self.log_path = "logs/dqn_log.txt"
        self.loger_mode = True

        self.model = self._build_model()
        self.memory = deque(maxlen=512)
        self.frame_sequence = Sequence(80, 80, self.lastk)
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(input_shape=(80,80,self.lastk, ), filters=16, kernel_size=(8,8), strides = 4, activation='relu'))
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
        network_current = np.copy(self.frame_sequence.sequence.reshape \
            (1, self.state_size[0], self.state_size[1], self.lastk))

        self.frame_sequence.add_frame(state)

        network_next = np.copy(self.frame_sequence.sequence.reshape \
            (1, self.state_size[0], self.state_size[1], self.lastk))

        self.memory.append((network_current, action, reward, network_next, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for network_current, action, reward, network_next, done in minibatch:
            #data = np.zeros((80, 80, 3), dtype=np.uint8)
            #data[:,:,0] = network_current[0,:, :, 0]
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
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

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
            #env.render()

            network_input = agent.frame_sequence.sequence.reshape(1, state_size[0], state_size[1], agent.lastk)
            action = agent.act(network_input)
            next_observation, reward, done, _ = env.step(action)
            next_state = preprocessor.preprocess_observation(next_observation)

            agent.remember(state,action,reward, next_state, done)

            state = next_state

            reward_sum += reward

            if done:
                if agent.loger_mode:
                    with open(agent.log_path, "a") as log_file:
                        log_file.write("episode: {}/{}, episodelen: {}, score: {}, e: {:.2}\n"
                        .format(e, agent.episodes, episode_len, reward_sum, agent.epsilon))
                else:
                    print("episode: {}/{}, episodelen: {}, score: {}, e: {:.2}"
                        .format(e, agent.episodes, episode_len, reward_sum, agent.epsilon))

                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                if e % agent.save_model_freq == 0:
                    agent.model.save(agent.model_path+str(agent.save_model_freq/50))

                break

            if len(agent.memory) > agent.batch_size:
                agent.replay(agent.batch_size)



if __name__ == '__main__':
    main()