import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import random
import numpy as np
from collections import deque
import keras
keras.backend.set_image_data_format('channels_first')

__all__ = [
    'DeepQNetwork',
    'GAME',
    'ACTIONS',
    'FRAME_PER_ACTION',
]


GAME = 'bird'
ACTIONS = 2
FRAME_PER_ACTION = 1


class DeepQNetwork:
    def __init__(self):
        self.actions = 2
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.0001
        self.batch_size = 32
        self.learning_rate = 1e-6
        self.model = self.build()

    def build(self):
        # Tune layers and parameters for faster learning
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='same', input_shape=(80, 80, 4)))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D())
        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(ACTIONS))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append({'state': state,
                            'action': action,
                            'reward': reward,
                            'next_state': next_state,
                            'done': done})

    def do_action(self, state):
        if random.random() <= self.epsilon:
            print("Random action")
            return random.randrange(self.actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    """
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        inputs = np.zeros((self.batch_size, 80, 80, 4))
        targets = np.zeros((self.batch_size, ACTIONS))
        for i in range(len(minibatch)):
            mem = minibatch[i]
            state = mem['state']
            action = mem['action']
            next_state = mem['next_state']
            reward = mem['reward']
            terminate = mem['done']

            inputs[i:i + 1] = state
            # targets[i] = self.model.predict(state)
            
            targets[i] = action
            action_index_ = np.argmax(action)
            
            if terminate:
                targets[i, action_index_] = reward
            else:
                targets[i, action_index_] = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        self.model.fit(inputs, targets, epochs=10, batch_size=self.batch_size, verbose=0)
    """

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        inputs = []
        targets = np.zeros((self.batch_size, ACTIONS))
        # targets = []
        for i in range(len(minibatch)):
            mem = minibatch[i]
            state = mem['state']
            action = mem['action']
            next_state = mem['next_state']
            reward = mem['reward']
            terminate = mem['done']
            targets[i] = self.model.predict(state)[0]
            q = self.model.predict(next_state)[0]
            action_index = np.argmax(action)
            if terminate:
                targets[i, action_index] = reward
            else:
                targets[i, action_index] = reward + self.gamma * np.amax(q)
            inputs.append(state)

        inputs = np.concatenate(inputs)
        self.model.fit(inputs, targets, epochs=1, batch_size=self.batch_size, verbose=0)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        self.model.save_weights("model.h5")
        print("Model saved")

    def load(self):
        self.model.load_weights("model.h5")
        print("Model loaded")

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    def replay_last_15(self):
        # minibatch = random.sample(self.memory, self.batch_size)
        minibatch = []
        for j in range(15, -1, -1):
            minibatch.append(self.memory[j]) 
        minibatch = np.array(minibatch)
        inputs = np.zeros((self.batch_size, 80, 80, 4))
        targets = np.zeros((self.batch_size, ACTIONS))
        for i in range(len(minibatch)):
            mem = minibatch[i]
            state = mem['state']
            action = mem['action']
            next_state = mem['next_state']
            reward = mem['reward']
            terminate = mem['done']

            inputs[i:i + 1] = state
            # targets[i] = self.model.predict(state)
            targets[i] = action
            action_index_ = np.argmax(action)
            if terminate:
                targets[i, action_index_] = reward
            else:
                targets[i, action_index_] = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        self.model.fit(inputs, targets, epochs=10, batch_size=15, verbose=0)

