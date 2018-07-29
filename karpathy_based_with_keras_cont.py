import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Reshape
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.convolutional import UpSampling2D, Convolution2D

resume = True
should_render = True

def pong_preprocess_screen(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


def discount_rewards(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def create_model(number_of_inputs, learning_rate, input_dim=80 * 80, model_type=1):
    model = Sequential()
    if model_type == 0:
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = RMSprop(lr=learning_rate)
    elif model_type == 1:
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu', init='he_uniform'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = Adam(lr=learning_rate)
    else:
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
        # model.add(Convolution2D(input_shape=(80, 80, 1,), filters=16, kernel_size=(8, 8), strides=4, activation='relu'))
        # model.add(Convolution2D(filters=32, kernel_size=(4, 4), strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    if resume == True:
        model.load_weights('pong_model_checkpoint_1701.h5')
    return model

def main():
    # Script Parameters
    input_dim = 80 * 80
    gamma = 0.99
    update_frequency = 10
    learning_rate = 0.001

    # Initialize
    env = gym.make("Pong-v0")
    number_of_inputs = 3  # This is incorrect for Pong (?)
    # number_of_inputs = 1
    observation = env.reset()
    prev_x = None
    xs, dlogps, drs, probs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 2430
    train_X = []
    train_y = []

    model = create_model(number_of_inputs, learning_rate)

    while True:
        if should_render:
            env.render()
        # Preprocess, consider the frame difference as features
        cur_x = pong_preprocess_screen(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        prev_x = cur_x
        # Predict probabilities from the Keras model
        aprob = ((model.predict(x.reshape([1, x.shape[0]]), batch_size=1).flatten()))
        # aprob = aprob/np.sum(aprob)
        # Sample action
        # action = np.random.choice(number_of_inputs, 1, p=aprob)
        # Append features and labels for the episode-batch
        xs.append(x)
        probs.append(aprob)
        aprob = aprob / np.sum(aprob)
        action = np.random.choice(number_of_inputs, 1, p=aprob)[0]
        y = np.zeros([number_of_inputs])
        y[action] = 1
        # print action
        dlogps.append(np.array(y).astype('float32') - aprob)
        observation, reward, done, info = env.step(action if action != 1 else 3)
        reward_sum += reward
        drs.append(reward)
        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            discounted_epr = discount_rewards(epr, gamma)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            epdlogp *= discounted_epr
            # Slowly prepare the training batch
            train_X.append(xs)
            train_y.append(epdlogp)
            xs, dlogps, drs = [], [], []
            # Periodically update the model
            if episode_number % update_frequency == 1:
                y_train = probs + learning_rate * np.squeeze(np.vstack(train_y))  # Hacky WIP
                model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)
                # clear batch
                train_X = []
                train_y = []
                probs = []
                if (episode_number % 100 == 1):
                    model.save_weights('pong_model_checkpoint_' + str(episode_number) + '.h5')
            # Reset the current environment nad print the current results
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            with open("log.txt", "a") as logfile:
                logfile.write('(%d) Environment reset imminent. Total Episode Reward: %f. Running Mean: %f' % (
                episode_number, reward_sum, running_reward))
            print('(%d) Environment reset imminent. Total Episode Reward: %f. Running Mean: %f' % (
            episode_number, reward_sum, running_reward))
            reward_sum = 0
            observation = env.reset()
            prev_x = None
        # if reward != 0:
        # print( ('Episode %d Result: ' % episode_number) + ('Defeat!' if reward == -1 else 'VICTORY!') )

if __name__ == '__main__':
    main()