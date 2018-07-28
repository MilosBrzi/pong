# Process is:

# processed_observations = image vector - [6400 x 1] array
# Learning after round has finished:

import gym
import numpy as np
from gym.envs.classic_control import rendering
from PIL import Image

class Preprocessor:

    prev_processed_observation = None
    preprocessed_observation_size = (80,80)

    def __init__(self, input_dim):
        self.input_dimensions = input_dim

    def downsample(self, image):
        # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
        return image[::2, ::2, :]

    def image_to_grayscale(self, image):
        """Convert all color (RGB is the third dimension in the image)"""
        return image[:, :, 0]

    def image_to_black_white(self, image):
        image[image == 144] = 0
        image[image == 109] = 0
        image[image != 0] = 1  # everything else (paddles, ball) just set to 1

        return image

    def preprocess_observation(self, input_observation):
        """ convert the 210x160x3 uint8 frame into a 6400 float vector """

        viewer = rendering.SimpleImageViewer()

        processed_observation = input_observation[35:195] # crop

        processed_observation = self.downsample(processed_observation)

        processed_observation = self.image_to_grayscale(processed_observation)

        processed_observation = self.image_to_black_white(processed_observation)

        # Convert from 80 x 80 matrix to 6400 float vector
        #processed_observation = processed_observation.astype(np.float).ravel()

        # subtract the previous frame from the current one so we are only processing on changes in the game
        if self.prev_processed_observation is not None:
            input_observation = processed_observation - self.prev_processed_observation
        else:
            input_observation = np.zeros(self.preprocessed_observation_size)

        # store the previous frame so we can subtract from it next time
        self.prev_processed_observation = processed_observation

        #data = np.zeros((80,80,3), dtype=np.uint8)
        #data[:,:,0] = input_observation[:,:]
        #img = Image.fromarray(data, 'RGB')
        #img.show()
        return input_observation