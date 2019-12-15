from tensorflow.keras.losses import Huber
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization, Lambda
from keras.optimizers import Adam, SGD, RMSprop
import random
import numpy as np
from collections import deque
import keras as K
import cv2

K.backend.set_image_data_format('channels_last')


class Hydra:
    def __init__(self):
        self.model = Hydra.build()

    @staticmethod
    def build():
        # inputs:
        #    frame
        #
        # outputs:
        #    y position of bird                 bird_ypos
        #    y of center of next pipe gap       gap_center_y
        #    x of nearest pipe edge             next_edge_x

        inputs = Input(shape=(80, 80, 1))

        ypos_head = Hydra.build_ypos_head(inputs)

        model = Model(inputs=[inputs], outputs=[ypos_head])

        losses = {'bird_ypos': 'mse'}

        model.compile(optimizer=Adam(learning_rate=0.01), loss=losses)

        return model

    @staticmethod
    def build_ypos_head(input_frame_stack):
        x = Conv2D(8, (3, 3), padding="same", data_format='channels_last')(input_frame_stack)
        x = Activation("relu")(x)
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        #x = BatchNormalization(axis=-1)(x)
        x = Conv2D(16, (2, 2), padding="same", data_format='channels_last')(x)
        x = Activation("relu")(x)
        # x = Conv2D(32, (3, 3), padding="same", data_format='channels_last')(x)
        # x = Activation("relu")(x)
        #x = BatchNormalization(axis=-1)(x)
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        #x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        #x = Dense(128, activation='relu')(x)
        x = Dense(1, name='bird_ypos', activation='linear')(x)  # regression cnn

        return x

    @staticmethod
    def build_target_ypos_head(input_frame_stack):
        pass

    def predict(self, frame, expected):
        frame = np.reshape(frame, (1, *frame.shape))

        bird_y = self.model.predict(frame)[0] #* 512.0
        #bird_y = np.argmax(bird_y)

        #print(f'max bird_y = {bird_y}')
        print(f'Predicted bird y {bird_y}, expected {expected}, off by {abs(expected - bird_y)}')
        #print(f'Predicted: {bird_y}')

    def learn(self, memory):
        if len(memory) < 1000:
            return

        inputs = np.zeros((len(memory), 80, 80, 1))
        y = np.zeros((len(memory), 1))

        for i in range(len(memory)):
            data = memory[i]

            #inputs[i] = data.frame / 255.0

            y[i] = data.bird_y
            #y[i] = 100.0 #/ 512.0

        loss = self.model.fit(inputs, y, batch_size=8, epochs=20, verbose=True)
        #print(f'loss = {loss}')

    @staticmethod
    def preprocess_frame(raw_frame_data):
        # drop any section that the bird has passed, no longer of interest and waste of computation time
        leftmost = int(288 * 0.2 - 1)

        raw_frame_data = raw_frame_data[leftmost:, :]

        # drop ground area, nothing exciting happens there
        raw_frame_data = raw_frame_data[:, 0:int(512 * .79)]

        # resize image (needed to keep size and num parameters low)
        # todo: check if square input required for gpu processing
        raw_frame_data = cv2.resize(raw_frame_data, (80, 80))

        # convert to grayscale (this is faster using cv2 than np by quite a bit)
        raw_frame_data = cv2.cvtColor(raw_frame_data, cv2.COLOR_RGB2GRAY)

        raw_frame_data = np.reshape(raw_frame_data, (80, 80, 1))

        # don't normalize the pixel value yet ... it'll take at least 2 bytes of space and having as many
        # samples in memory as possible is ideal

        return raw_frame_data
