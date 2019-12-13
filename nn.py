from tensorflow.keras.losses import Huber
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization
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
    def __init__(self, input_shape,
                 num_actions,
                 num_history,
                 frame_stack_depth,
                 batch_size,
                 learning_rate,
                 initial_epsilon=1.0,
                 min_epsilon=0.0001, epsilon_decay=0.95, update_network_epochs=500):
        self.actions = num_actions
        self.memory = deque(maxlen=num_history)
        self.gamma = 0.99
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = min_epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.frame_stack_depth = frame_stack_depth
        self.input_shape = input_shape

        self.model = self.build()
        self.target_network = self.build()
        self.update_network_epochs = update_network_epochs
        self.current_epoch = 0

        self.dlayer = self.model.get_layer('dense_layer')

    def build(self):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same',
                         input_shape=self.input_shape))
        #model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256, name='dense_layer'))
        model.add(Activation('relu'))
        model.add(Dense(self.actions, name='q_actions'))

        # rewards should be clipped to [-1, 1] range for this delta to be correct
        # todo: Huber loss
        #odel.compile(loss=Huber(delta=2), optimizer=RMSprop(learning_rate=self.learning_rate))
        #model = Model(inputs=model_input, outputs=model_output)
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate, clipvalue=0.5))

        print(f'input shape: {model.input_shape}')
        model.summary()

        return model

    def remember(self, state, action, reward, next_frame, done, significant=False):
        self.memory.append(Memory(state, next_frame, action, reward, done))

        # if significant and len(self.memory) > 30:
        #     # this sequence was actions was especially important, so to increase the odds
        #     # it is trained, let's duplicate the steps that led to this result
        #     sequence = deque()
        #
        #     for j in range(30):  # recall last 30 frames
        #         sequence.append(self.memory.pop())
        #
        #     for i in range(5):  # duplicate this 4 times (we removed originals and are putting them back now)
        #         for m in sequence:
        #             self.memory.append(m)

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.actions)

        # state = np.reshape(state, (1, *state.shape))
        act_values = self.model.predict(state)

        print(f'choose action: {act_values[0]}')

        return np.argmax(act_values[0])

    def create_minibatch(self):
        minibatch = random.sample(self.memory, self.batch_size)

        x_prediction = np.zeros((self.batch_size, *self.input_shape))
        y_reality = np.zeros((self.batch_size, self.actions))

        for i, memory in enumerate(minibatch):
            state_stack = memory.state
            next_state_stack = self.add_to_frame_stack(memory.next_state, memory.state)

            x_prediction[i] = np.divide(memory.state, 255.0)
            next_state_stack = np.divide(next_state_stack, 255.0)

            y_reality[i] = self.target_network.predict(state_stack)[0]
            best_q = np.max(self.target_network.predict(next_state_stack)[0])

            if memory.terminated:
                #y_reality[i, memory.action_index] = memory.reward
                reward = memory.reward
            else:
                #y_reality[i, memory.action_index] = memory.reward + self.gamma * best_q
                reward = memory.reward + self.gamma * best_q

            # clip rewards
            for j in range(self.actions):
                y_reality[i, j] = max(1, min(reward, -1.0))

        return x_prediction, y_reality

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.current_epoch += 1
        if self.current_epoch % self.update_network_epochs == 0:
            self.target_network.set_weights(self.model.get_weights())
            print('target network updated')

        x, y = self.create_minibatch()

        loss = self.model.train_on_batch(x, y)  # returns loss
        print(f'loss: {loss}')

        #print(self.dlayer.get_weights())

    # def replay(self):
    #     if len(self.memory) < self.batch_size:
    #         return
    #
    #     minibatch = random.sample(self.memory, self.batch_size)
    #
    #     #targets = np.zeros((self.batch_size, self.actions))
    #
    #     #input = []
    #
    #     for i, memory in enumerate(minibatch):
    #         target_reward = memory.reward
    #         #input.append(memory.state)
    #
    #         #minibatch[i] = np.reshape(memory.state, (1, self.frame_stack_depth, 80, 80, 3))
    #         #minibatch[i] = memory.state
    #         #minibatch = np.reshape(memory.state, (1, *memory.state.shape))
    #
    #         if not memory.terminated:
    #             # using previous state stack and the next frame that resulted, construct future state
    #             # to predict with
    #             future_state_stack = memory.state
    #             #future_state_stack = np.roll(future_state_stack, shift=3, axis=2)
    #             #future_state_stack[:, :, 0:3] = memory.next_state
    #             #future_state_stack = np.reshape(future_state_stack, (1, *future_state_stack.shape))
    #
    #             future_state_stack = np.roll(future_state_stack, shift=1, axis=0)
    #             future_state_stack[0] = memory.next_state
    #
    #             prediction = self.model.predict(future_state_stack)[0]
    #             amax = np.amax(prediction)
    #
    #             target_reward = memory.reward + self.gamma * np.amax(prediction)
    #
    #         # todo: normalize reward to [-1, 1] range?
    #         prediction = self.model.predict(memory.state)
    #         target_f = prediction[0]
    #         target_f[memory.action_index] = target_reward
    #
    #         #targets[i] = target_f
    #
    #         target_f = np.array(target_f)
    #         #target_f = np.reshape(target_f, (2,))
    #
    #         self.model.fit(memory.state, target_f, epochs=1, batch_size=self.batch_size, verbose=0)
    #         #self.model.fit(minibatch, target_f, epochs=1, batch_size=self.batch_size, verbose=0)
    #
    #     # targets = np.array(targets)
    #
    #     #minibatch = np.array(minibatch)
    #     #minibatch = np.concatenate(minibatch)
    #     #minibatch = np.concatenate(input)
    #     #self.model.train_on_batch(minibatch, targets)
    #
    #     #self.model.fit(minibatch, targets, epochs=1, verbose=0, batch_size=self.batch_size)
    #
    #     self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self):
        self.model.save_weights("model.h5")
        print("Model saved")

    def load(self):
        self.model.load_weights("model.h5")
        self.target_network.set_weights(self.model.get_weights())

        print("Model loaded")

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    @staticmethod
    def preprocess_frame(game, raw_frame_data):
        # todo: move this out of nn so nn can be generalized to other games
        # todo: don't hardcode input shape so it can be generalized

        from flappybird import BASEY, FlappyBird

        # drop any section that the bird has passed, no longer of interest and waste of computation time
        leftmost = int(game.playerx)

        raw_frame_data = raw_frame_data[leftmost:, :]

        # drop ground area, nothing exciting happens there
        raw_frame_data = raw_frame_data[:, 0:int(BASEY)]

        # resize image (needed to keep size and num parameters low)
        # todo: check if square input required for gpu processing
        raw_frame_data = cv2.resize(raw_frame_data, (80, 80))

        # convert to grayscale (this is faster using cv2 than np by quite a bit)
        raw_frame_data = cv2.cvtColor(raw_frame_data, cv2.COLOR_RGB2GRAY)
        # raw_frame_data = np.reshape(raw_frame_data, (80, 80, 1))
        # raw_frame_data = np.reshape(raw_frame_data, (1, 80, 80, 1))

        # don't normalize the pixel value yet ... it'll take at least 2 bytes of space and having as many
        # samples in memory as possible is ideal

        return raw_frame_data

    def add_to_frame_stack(self, new_frame, frame_stack):
        if frame_stack is None:
            frame_stack = np.stack((new_frame, new_frame, new_frame, new_frame), axis=2)
            #frame_stack = np.squeeze(frame_stack, axis=3)
            frame_stack = np.reshape(frame_stack, (1, *frame_stack.shape))

            return frame_stack

        frame_stack = np.roll(frame_stack, axis=3, shift=1)
        new_frame = np.reshape(new_frame, (*new_frame.shape, 1))

        frame_stack[:, :, :, 0:1] = new_frame

        return frame_stack

    # def preprocess_images(self, images):
    #     if images.shape[0] < 4:
    #         # single image
    #         x_t = images[0]
    #         s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #     else:
    #         # 4 images
    #         xt_list = []
    #         for i in range(images.shape[0]):
    #             x_t = images[i]
    #             xt_list.append(x_t)
    #         s_t = np.stack((xt_list[0], xt_list[1], xt_list[2], xt_list[3]),
    #                        axis=2)
    #     s_t = np.expand_dims(s_t, axis=0)
    #     return s_t