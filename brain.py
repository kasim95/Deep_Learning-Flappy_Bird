from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import numpy as np
from collections import deque

NUM_INPUTS = 4
IDX_CURRENT_BIRD_Y = 0
IDX_BIRD_Y_VELOCITY = 1
IDX_NEXT_PIPE_EDGE_X = 2
IDX_TARGET_GAP_Y = 3


__all__ = ['FrameData', 'Brain']


class FrameData:
    __slots__ = ['bird_y',  'bird_y_velocity', 'next_pipe_x', 'gap_y']

    def __init__(self, bird_y, gap_y, next_pipe_x, bird_y_velocity):
        self.bird_y = bird_y
        self.gap_y = gap_y
        self.next_pipe_x = next_pipe_x
        self.bird_y_velocity = bird_y_velocity


class Brain:
    def __init__(self, gap_size, bird_width, bird_height, bird_x_velocity, bird_x, pipe_width, bird_y_acc,
                 mem_size=100000, batch_size=1024, epochs=20):
        # inputs: current bird y, target gap y, next pipe edge x
        self.bird_x = bird_x
        self.bird_width, self.bird_height = bird_width, bird_height
        self.bird_x_velocity = bird_x_velocity
        self.bird_y_acc = bird_y_acc
        self.gap_size = gap_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.pipe_width = pipe_width

        self.model = Sequential()
        self.model.add(Dense(256, input_dim=NUM_INPUTS, name='inputs'))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Dense(128, activation='sigmoid'))
        self.model.add(Dropout(rate=0.25))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, activation='sigmoid'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.25))
        self.model.add(Dense(2, name='outputs', activation='softmax'))

        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        self.model.summary()
        print(self.model.get_layer('inputs').input_shape)

        self.memory = deque(maxlen=mem_size)

    def predict(self, frame_data):
        # order matters
        inputs = np.array([
            (frame_data.bird_y - 256.0) / 10.0,
            frame_data.bird_y_velocity,
            frame_data.next_pipe_x / 10.0,
            (frame_data.gap_y - 256.0) / 10.0
        ])
        inputs = np.reshape(inputs, (1, *inputs.shape))

        act_values = self.model.predict(inputs)
        chosen = np.argmax(act_values[0])

        return chosen == 1  # jump?

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

            # normalizing input data lead to substantial improvement
            inputs[i][IDX_CURRENT_BIRD_Y] = (data.bird_y - 256.0) / 10.0
            inputs[i][IDX_BIRD_Y_VELOCITY] = data.bird_y_velocity
            inputs[i][IDX_NEXT_PIPE_EDGE_X] = data.next_pipe_x / 10.0
            inputs[i][IDX_TARGET_GAP_Y] = (data.gap_y - 256.0) / 10.0

            # only need to jump if we won't pass this pipe in time
            if data.next_pipe_x + self.pipe_width + self.bird_x_velocity <= self.bird_x:
                will_hit = False
            else:
                will_hit = data.bird_y + self.bird_height + data.bird_y_velocity + self.bird_y_acc + 1>= \
                    data.gap_y + self.gap_size * 0.5

            y[i] = jump if will_hit else wait

        self.model.fit(inputs, y, batch_size=self.batch_size, epochs=self.epochs, shuffle=True, verbose=True)

    def store_memory(self, frame_data):
        self.memory.append(frame_data)

    def save(self, filename=None):
        self.model.save_weights(filename or "model.h5")
        print("Model saved")

    def load(self, filename=None):
        self.model.load_weights(filename or "model.h5")
        print("Model loaded")
