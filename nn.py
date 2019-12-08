import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import random
import numpy as np
from collections import deque


__all__ = [
    'DeepQNetwork',
    'GAME',
    'ACTIONS',
    'FRAME_PER_ACTION',
]


GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 1000
EXPLORE = 50000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1


class DeepQNetwork:
    def __init__(self):
        self.input_dim = [None, 80, 80, 4]
        self.actions = 2
        self.memory = deque(maxlen=50000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.input = tf.Variable([-1, ])
        self.epsilon_min = 200000
        self.batch_size = 32
        self.learning_rate = 0.001
        self.model = self.build()

    def build(self):
        # Tune layers and parameters for faster learning
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(1600))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(ACTIONS))
        model.add(Activation('relu'))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append({'state': state,
                            'action': action,
                            'reward': reward,
                            'next_state': next_state,
                            'done': done})

    def do_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        #for state, action, reward, next_state, done in minibatch:
        for i in minibatch:
            state = i['state']
            action = i['action']
            reward = i['reward']
            next_state = i['next_state']
            done = i['done']
            print(state.shape)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            print(np.array(target_f).shape)
            print(action)
            target_f[0][np.argmax(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
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
