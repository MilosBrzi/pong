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
        self.learning_rate = 0.001
        self.lastk = 4


        self.batch_size = 100
        self.memory = deque(maxlen=50000)

        self.model_path = "models/DQN_model_e_"
        self.save_model_freq = 5
        self.log_path = "logs/dqn_log.txt"
        self.loger_mode = True

        self.model = self._build_model()

        self.frame_sequence = Sequence(80, 80, self.lastk)

    def _build_model(self):
        W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
        b_conv1 = tf.Variable(tf.zeros([32]))

        W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
        b_conv2 = tf.Variable(tf.zeros([64]))

        W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
        b_conv3 = tf.Variable(tf.zeros([64]))

        W_fc4 = tf.Variable(tf.zeros([2304, 576]))
        b_fc4 = tf.Variable(tf.zeros([576]))

        W_fc5 = tf.Variable(tf.zeros([576, 3]))
        b_fc5 = tf.Variable(tf.zeros([3]))

        # input for pixel data
        self.inp = tf.placeholder("float", [None, 80, 80, 4])

        # Computes rectified linear unit activation fucntion on  a 2-D convolution given 4-D input and filter tensors. and
        conv1 = tf.nn.relu(tf.nn.conv2d(self.inp, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)

        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)

        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)

        conv3_flat = tf.reshape(conv3, [-1, 2304])

        fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

        self.out = tf.matmul(fc4, W_fc5) + b_fc5

        self.argmax = tf.placeholder("float", [None, 3])
        self.gt = tf.placeholder("float", [None])  # ground truth

        # action
        self.action = tf.reduce_sum(tf.multiply(self.out, self.argmax), reduction_indices=1)
        # cost function we will reduce through backpropagation
        self.cost = tf.reduce_mean(tf.square(self.action - self.gt))
        # optimization fucntion to reduce our minimize our cost function
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)



    def act(self, network_input):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        out_t = self.out.eval(feed_dict = {self.inp : network_input})[0]
        maxIndex = np.argmax(out_t)

        return maxIndex  # returns action

    def remember(self, state, action, reward, next_state, done):
        network_current = np.copy(self.frame_sequence.sequence)

        self.frame_sequence.add_frame(state)

        network_next = np.copy(self.frame_sequence.sequence)

        self.memory.append((network_current, action, reward, network_next, done))


    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)

        network_current = [d[0] for d in minibatch]
        actions = [d[1] for d in minibatch]
        rewards = [d[2] for d in minibatch]
        network_next = [d[3] for d in minibatch]

        out_batch = self.out.eval(feed_dict={self.inp: network_next})
        gt_batch = []
        for i in range(0, len(minibatch)):
            gt_batch.append(rewards[i] + self.gamma * np.max(out_batch[i]))


        # train on that
        self.train_step.run(feed_dict={
            self.gt: gt_batch,
            self.argmax: actions,
            self.inp: network_current
        })


def main():
    env = gym.make('Pong-v0')

    observation_size = env.observation_space.shape
    preprocessor = Preprocessor(observation_size)

    state_size = Preprocessor.preprocessed_observation_size
    action_size = int(env.action_space.n / 2)

    sess = tf.InteractiveSession()

    agent = DQNAgent(state_size, action_size)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())


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
            # returns action 0,1 or 2
            action = agent.act(network_input)

            # only serves as env.step
            en_action = action
            if action == 1: en_action = 3
            next_observation, reward, done, _ = env.step(en_action)

            next_state = preprocessor.preprocess_observation(next_observation)

            action_one_hot = np.zeros([agent.action_size])
            action_one_hot[action] = 1
            agent.remember(state,action_one_hot,reward, next_state, done)

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
                    saver.save(sess, './models/' + 'pong' + '-dqn', global_step=e)

                break

            #update
            if e != 0 or episode_len > agent.pass_frames:
                if len(agent.memory) > agent.batch_size:
                    agent.replay()




if __name__ == '__main__':
    main()