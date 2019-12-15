from tensorflow.keras.losses import Huber
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization, Lambda
from keras.optimizers import Adam, SGD, RMSprop
import random
import numpy as np
from collections import deque
import keras as K
import cv2

NUM_INPUTS = 3
IDX_CURRENT_BIRD_Y = 0
IDX_TARGET_GAP_Y = 1
IDX_NEXT_PIPE_EDGE_X = 2


class FrameData:
    __slots__ = ['frame', 'bird_y', 'gap_y', 'next_pipe_x']

    def __init__(self, frame, bird_y, gap_y, next_pipe_x):
        self.frame = frame
        self.bird_y = bird_y
        self.gap_y = gap_y
        self.next_pipe_x = next_pipe_x


class SimpleBrain:
    def __init__(self):
        # inputs: current bird y, target gap y, next pipe edge x

        self.model = Sequential()
        self.model.add(Dense(128, input_dim=NUM_INPUTS, name='inputs'))
        self.model.add(Dropout(rate=0.25))
        #self.model.add(Dense(128, input_shape=(,NUM_INPUTS), name='inputs'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(rate=0.25))
        self.model.add(BatchNormalization())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.25))
        self.model.add(Dense(2, name='outputs', activation='softmax'))

        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        self.model.summary()
        print(self.model.get_layer('inputs').input_shape)

        self.memory = deque(maxlen=10000)

    def should_jump(self, frame, current_bird_y, target_gap_y, next_pipe_x):
        inputs = np.array([current_bird_y, target_gap_y, next_pipe_x])
        inputs = np.reshape(inputs, (1, *inputs.shape))

        act_values = self.model.predict(inputs)

        chosen = np.argmax(act_values[0])

        if chosen == 1:
            print(f'Jump! : {act_values[0]}')
            # print(f'choose action: {act_values[0]}')
            return True
        else:
            print(f'nothing : {act_values[0]}')
            return False

    # def train(self, current_bird_y, target_gap_y, next_pipe_x):
    #     if current_bird_y >= target_gap_y:
    #         truth = [0.0, 1.0]
    #     else:
    #         truth = [1.0, 0.0]
    #
    #     inputs = np.array([current_bird_y, target_gap_y, next_pipe_x])
    #     inputs = np.reshape(inputs, (1, *inputs.shape))
    #
    #     outputs = np.array(truth)
    #     outputs = np.reshape(outputs, (1, *outputs.shape))
    #
    #     result = self.model.fit(x=inputs, y=outputs, verbose=True)

    def learn(self):
        if len(self.memory) < 1000:
            return

        inputs = np.zeros((len(self.memory), NUM_INPUTS))
        y = np.zeros((len(self.memory), 2))

        jump = np.zeros((2,))
        wait = np.zeros_like(jump)

        jump[1] = 1.0
        wait[0] = 1.0

        for i in range(len(self.memory)):
            data = self.memory[i]  # type: FrameData

            inputs[i][IDX_CURRENT_BIRD_Y] = data.bird_y
            inputs[i][IDX_TARGET_GAP_Y] = data.gap_y
            inputs[i][IDX_NEXT_PIPE_EDGE_X] = data.next_pipe_x

            y[i] = jump if data.bird_y >= data.gap_y else wait

        loss = self.model.fit(inputs, y, batch_size=256, epochs=20, verbose=True)
        print(f'loss = {loss}')

    def store_memory(self, frame, bird_y, gap_y, next_pipe_x):
        self.memory.append(FrameData(frame, bird_y, gap_y, next_pipe_x))