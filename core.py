import random
import gym
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0001
        self.lastk = 4

        self.exp_replay = True
        self.batch_size = 2
        if self.exp_replay == False:
            self.memory = deque(maxlen=4)
        else: self.memory = deque(maxlen=15000)

        self.target_core_mode = True
        self.q_target_freq = 1

        self.model_path = "models/DQN_model_e_"
        self.save_model_freq = 50
        self.log_path = "logs/dqn_log.txt"
        self.loger_mode = True

        self.act_log_path = "logs/act_log.txt"

        self.model = self._build_model()
        self.fixed_model = self._build_model()

        self.frame_sequence = Sequence(80, 80, self.lastk)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(input_shape=(80,80,self.lastk, ), filters=16, kernel_size=(8,8), strides = 4, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(4,4), strides = 2, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides = 1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    def update_fixed_model(self):
        self.fixed_model.set_weights(self.model.get_weights())

    def act(self, state, log_mode):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        if log_mode==True:
            with open(self.act_log_path, "a") as act_log_file:
                act_log_file.write("actions: {} best_action: {} \n"
                           .format(act_values, np.argmax(act_values[0])))

        return np.argmax(act_values[0])  # returns action_index


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def remember(self, state, action_index, reward, done):
        network_current = np.copy(self.frame_sequence.sequence)

        self.frame_sequence.add_frame(state)

        network_next = np.copy(self.frame_sequence.sequence)

        self.memory.append((network_current, action_index, reward, network_next, done))

    def update(self):
        network_current, action, reward, network_next, done = self.memory[0]
        network_current = network_current.reshape(1, 80, 80, 4)
        network_next = network_next.reshape(1, 80, 80, 4)

        target = reward
        if not done:
            target = (reward + self.gamma *
                      np.amax(self.model.predict(network_next)[0]))
        target_f = self.model.predict(network_current)
        tmp = target_f.copy()
        target_f[0][action] = target
        self.model.fit(network_current, target_f, epochs=1, verbose=0)
        new_predict = self.model.predict(network_current)
        print('cao')

    def replay(self, target_mode):
        minibatch = random.sample(self.memory, self.batch_size)

        network_current_mini_batch = [mb[0] for mb in minibatch]
        network_current_mini_batch = np.asarray(network_current_mini_batch)

        network_next_mini_batch = [mb[3] for mb in minibatch]
        network_next_mini_batch = np.asarray(network_next_mini_batch)

        current_predicted_actions = self.model.predict(network_current_mini_batch)
        if target_mode == True:
            next_predicted_actions = self.fixed_model.predict(network_next_mini_batch)
        else:
            next_predicted_actions = self.model.predict(network_next_mini_batch)

        cnt = 0
        target_mini_batch = []
        for mb in minibatch:
            action_index = mb[1]
            one_reward = mb[2]
            done = mb[4]
            target = one_reward
            if not done:
                target = one_reward + self.gamma * np.amax(next_predicted_actions[cnt])
            current_predicted_actions[cnt][action_index] = target

            cnt += 1

        target_mini_batch = np.asarray(current_predicted_actions)
        self.model.fit(network_current_mini_batch, target_mini_batch, epochs=1, verbose=0)


def main():
    env = gym.make('Pong-v0')

    observation_size = env.observation_space.shape
    preprocessor = Preprocessor(observation_size)

    state_size = Preprocessor.preprocessed_observation_size
    action_size = 3
    log_actions_freq = 400
    agent = DQNAgent(state_size, action_size)
    target_mode = False

    for e in range(agent.episodes):
        observation = env.reset()
        state = preprocessor.preprocess_observation(observation)
        episode_len = 0
        reward_sum = 0

        if agent.target_core_mode == True:
            if(e > 1):
                agent.update_fixed_model()
                target_mode = True

        #init sequence with repeated first frame
        for i in range(agent.lastk):
            agent.frame_sequence.add_frame(state)

        #one episode
        while True:
            env.render()

            #Take one action
            network_input = agent.frame_sequence.sequence.reshape(1, state_size[0], state_size[1], agent.lastk)

            action_index = agent.act(network_input, episode_len % log_actions_freq * (1 - agent.epsilon) == 0)

            #Step into environment
            en_action = action_index
            if action_index == 1: en_action = 3
            next_observation, reward, done, _ = env.step(en_action)
            next_state = preprocessor.preprocess_observation(next_observation)

            agent.remember(state, action_index, reward, done)

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
                    agent.model.save(agent.model_path+str(e/agent.save_model_freq))

                break

            #update
            if e != 0 or episode_len > agent.pass_frames:
                if agent.exp_replay:
                    if len(agent.memory) > agent.batch_size:
                        agent.replay(target_mode)
                else:
                    agent.update()

            episode_len += 1


if __name__ == '__main__':
    main()