from collections import deque
import numpy as np

class Sequence:
    def __init__(self, w, h, size):
        self.w = w
        self.h = h
        self.size = size
        self.frames = deque(maxlen=size)
        self.sequence = np.zeros(shape=(w, h, size))

    def add_frame(self, frame):
        self.frames.append(frame)
        if (len(self.frames) == self.size):
            self.update_sequence()

    def update_sequence(self):
        for index in range(self.size):
            self.sequence[:, :, index] = self.frames[index]
