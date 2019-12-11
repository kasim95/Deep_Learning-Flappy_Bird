from tensorflow.keras.losses import Huber
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import random
import numpy as np
from collections import deque
import keras
import cv2

keras.backend.set_image_data_format('channels_last')

__all__ = [
    'DeepQNetwork'
]


class Memory:
    __slots__ = ['state', 'next_state', 'action_index', 'reward', 'terminated']

    def __init__(self, state, next_frame, action_index, reward, terminated):
        self.state = state
        self.next_state = next_frame
        self.action_index = action_index
        self.reward = reward
        self.terminated = terminated


class DeepQNetwork:
    def __init__(self, num_actions, num_history, frame_stack_depth, batch_size, learning_rate, initial_epsilon=1.0,
                 min_epsilon=0.0001, epsilon_decay=0.95):
        self.actions = num_actions
        self.memory = deque(maxlen=num_history)
        self.gamma = 0.99
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = min_epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.frame_stack_depth = frame_stack_depth

        self.model = self.build()

    def build(self):
        # Tune layers and parameters for faster learning
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='same',
                         input_shape=(80, 80, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.actions, name='actions'))

        # rewards should be clipped to [-1, 1] range for this delta to be correct
        #model.compile(loss=Huber(delta=2), optimizer=RMSprop(learning_rate=self.learning_rate))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate))

        model.summary()

        return model

    def remember(self, state, action, reward, next_frame, done):
        self.memory.append(Memory(state, next_frame, action, reward, done))

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.actions)

        act_values = self.model.predict(state)

        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        targets = []

        for i, memory in enumerate(minibatch):
            target = memory.reward

            if not memory.terminated:
                # using previous state stack and the next frame that resulted, construct future state
                # to predict with
                future_state_stack = memory.state
                future_state_stack = np.roll(future_state_stack, shift=1, axis=0)
                future_state_stack[0] = memory.next_state

                target = memory.reward + self.gamma * np.amax(self.model.predict(future_state_stack)[0])

            # todo: normalize reward to [-1, 1] range?
            target_f = self.model.predict(memory.state)
            target_f[memory.action_index] = target

            minibatch[i] = memory.state
            targets.append(target_f)

            #self.model.fit(memory.state, target_f, epochs=1, batch_size=self.batch_size, verbose=0)

        self.model.fit(minibatch, targets, epochs=1, batch_size=self.batch_size, verbose=0)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self):
        self.model.save_weights("model.h5")
        print("Model saved")

    def load(self):
        self.model.load_weights("model.h5")
        print("Model loaded")

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    @staticmethod
    def preprocess_frame(game, raw_frame_data):
        # todo: move this out of nn so nn can be generalized to other games
        from flappybird import BASEY, FlappyBird

        # todo: convert to grayscale? might cause problems for the NN to recognize the difference
        # between bird and pipe

        # drop any section that the bird has passed, no longer of interest and waste of computation time
        leftmost = int(game.playerx)

        raw_frame_data = raw_frame_data[leftmost:, :]

        # drop ground area, nothing exciting happens there
        raw_frame_data = raw_frame_data[:, 0:int(BASEY)]

        # resize image (needed to keep size and num parameters low)
        # todo: check if square input required for gpu processing
        raw_frame_data = cv2.resize(raw_frame_data, (80, 80))

        # scale to 0-1.0 RGB values, NN seem to like those
        raw_frame_data = np.array(raw_frame_data, dtype='float16')
        raw_frame_data /= 255.0

        return raw_frame_data
