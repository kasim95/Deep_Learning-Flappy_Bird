from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import numpy as np
from collections import deque

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


class Brain:
    def __init__(self, gap_size, bird_size, mem_size=20000, batch_size=1024, epochs=20):
        # inputs: current bird y, target gap y, next pipe edge x
        self.bird_size = bird_size
        self.gap_size = gap_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = Sequential()
        self.model.add(Dense(128, input_dim=NUM_INPUTS, name='inputs'))
        self.model.add(Dropout(rate=0.25))
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

        self.memory = deque(maxlen=mem_size)

    def should_jump(self, frame, current_bird_y, target_gap_y, next_pipe_x):
        inputs = np.array([current_bird_y, target_gap_y, next_pipe_x])
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

            inputs[i][IDX_CURRENT_BIRD_Y] = data.bird_y
            inputs[i][IDX_TARGET_GAP_Y] = data.gap_y
            inputs[i][IDX_NEXT_PIPE_EDGE_X] = data.next_pipe_x

            y[i] = jump if data.bird_y + self.bird_size >= data.gap_y + self.gap_size * 0.5 else wait

        self.model.fit(inputs, y, batch_size=self.batch_size, epochs=self.epochs, shuffle=True, verbose=True)

    def store_memory(self, frame, bird_y, gap_y, next_pipe_x):
        self.memory.append(FrameData(frame, bird_y, gap_y, next_pipe_x))
