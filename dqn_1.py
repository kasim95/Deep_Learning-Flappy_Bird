import tensorflow as tf
import cv2
from flappybird import *
import random
import numpy as np
from collections import deque


class DQN:
    def __init__(self):
        self.game = 'bird'
        self.actions = 2
        self.gamma = 0.99
        self.epsilon = 0.0001
        self.initial_epsilon = 0.05
        self.final_epsilon = 0.005
        self.frame_per_action = 2
        self.observe = 50000
        self.explore = 1000000     # 2000000 gives epsilon decay rate as 5e-8 for initial_epsilon as 0.1
        self.replay_memory = 50000
        self.save_step = 10000
        self.batch_size = 32
        self.save_dir = "checkpoints/"
        self.memory = deque(maxlen=self.replay_memory)
        self.game_state = FlappyBird()

    def create_network(self):
        # Input Layer
        layer_input = tf.placeholder("float", [None, 80, 80, 4])

        # First Convolutional Layer
        conv1_weights = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.1))
        conv1_bias = tf.Variable(tf.constant(0.01, shape=[32]))
        layer_conv1 = tf.nn.conv2d(layer_input, conv1_weights, strides=[1, 4, 4, 1], padding='SAME') + conv1_bias
        layer_activation1 = tf.nn.relu(layer_conv1)

        # Max Pooling layer
        layer_pool1 = tf.nn.max_pool(layer_activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second Convolutional Layer
        conv2_weights = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1))
        conv2_bias = tf.Variable(tf.constant(0.01, shape=[64]))
        layer_conv2 = tf.nn.conv2d(layer_pool1, conv2_weights, strides=[1, 2, 2, 1], padding='SAME') + conv2_bias
        layer_activation2 = tf.nn.relu(layer_conv2)

        # Third Convolutional Layer
        conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
        conv3_bias = tf.Variable(tf.constant(0.01, shape=[64]))
        layer_conv3 = tf.nn.conv2d(layer_activation2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME') + conv3_bias
        layer_activation3 = tf.nn.relu(layer_conv3)

        # Flatten after Convolution
        layer_flatten = tf.reshape(layer_activation3, [-1, 1600])

        # Dense Layer
        fc1_weights = tf.Variable(tf.truncated_normal([1600, 512], stddev=0.1))
        fc1_bias = tf.Variable(tf.constant(0.01, shape=[512]))
        layer_fc1 = tf.matmul(layer_flatten, fc1_weights) + fc1_bias
        layer_activation4 = tf.nn.relu(layer_fc1)

        # Output Layer
        fc2_weights = tf.Variable(tf.truncated_normal([512, self.actions], stddev=0.1))
        fc2_bias = tf.Variable(tf.constant(0.01, shape=[self.actions]))
        layer_output = tf.matmul(layer_activation4, fc2_weights) + fc2_bias

        return layer_input, layer_output

    def train_network(self, ip, readout, sess):
        # ------------------------------------------
        # glossary
        # t         - timestep
        # x_t       - frame at timestep t
        # s_t       - state at timestep t
        # s_t1      - next state at timestep t
        # a         - action
        # a_t       - action at timestep t
        # readout   - output
        # r_t       - reward at timestep t

        # ------------------------------------------
        # cost function
        a = tf.placeholder("float", [None, self.actions])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # ------------------------------------------
        # get first action
        first_action = np.zeros(self.actions)
        first_action[0] = 1

        # ------------------------------------------
        # apply first action and preprocess frame to get gamestate
        x_t_colored, r_0, terminal = self.game_state.frame_state(first_action)
        x_t = preprocess_image(x_t_colored)
        x_t = np.reshape(x_t, (80, 80))
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        # ------------------------------------------
        # load model
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Model: ", checkpoint.model_checkpoint_path, " loaded")
        else:
            print("Unable to find saved network")

        # ------------------------------------------
        # start training
        self.epsilon = self.initial_epsilon
        t = 0

        # keep track of previous actions
        a_t_previous = np.zeros([20, 2])
        for i in range(len(a_t_previous)):
            a_t_previous[i][0] = 1

        # ------------------------------------------
        while True:
            # ------------------
            # get output action to be applied to game in order to get frame
            readout_t = readout.eval(feed_dict={ip: [s_t]})[0]
            a_t = np.zeros([self.actions])
            action_index = 0
            if t % self.frame_per_action == 0:
                if random.random() <= self.epsilon:
                    print("____RANDOM ACTION_____")
                    action_index = random.randrange(self.actions)
                    # a_t[random.randrange(self.actions)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    # a_t[action_index] = 1

                #counts = (a_t_previous == [0,1]).sum() / 2
                #if action_index == 1 and counts >= 2:
                #   print("____________________________________________FORCED ACTION_____")
                #    action_index = 0

                # set action according to action index
            a_t[action_index] = 1

            # ------------------
            # scale down epsilon
            if self.epsilon > self.final_epsilon and t > self.observe:
                self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

            # ------------------
            # run selected action to get next frame & reward
            # then preprocess frame to get next state
            x_t1_colored, r_t, terminal = self.game_state.frame_state(a_t)
            x_t1 = preprocess_image(x_t1_colored)
            # the frame is stacked as 4 to account for downward acceleration
            # downward acceleration is similar to gravity in real life
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            # ------------------
            # save model
            self.memory.append((s_t, a_t, r_t, s_t1, terminal))

            # ------------------
            # train model on a batch size of 32 from memory
            if t > self.observe:
                minibatch = random.sample(self.memory, self.batch_size)
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]
                y_batch = []
                readout_j1_batch = readout.eval(feed_dict={ip: s_j1_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + self.gamma * np.max(readout_j1_batch[i]))

                # perform gradient step
                train_step.run(feed_dict={
                    y: y_batch,
                    a: a_batch,
                    ip: s_j_batch}
                )

            # ------------------
            # update state and timestep
            s_t = s_t1
            t += 1
            for i in range(len(a_t_previous) - 1, 0, -1):
                a_t_previous[i] = a_t_previous[i-1]
            a_t_previous[0] = a_t

            # ------------------
            # save progress every save_step iterations
            if t % self.save_step == 0:
                saver.save(sess, self.save_dir + self.game + '-dqn', global_step=t)

            # ------------------
            # print info
            if t <= self.observe:
                state = "observe"
            elif self.observe < t <= (self.observe + self.explore):
                state = "explore"
            else:
                state = "train"
            if r_t == 1:
                print("TIMESTEP", t,
                   "/ STATE", state,
                      "/ EPSILON", self.epsilon,
                      "/ ACTION", action_index,
                      "/ REWARD", r_t,
                      "/ Q_MAX %e" % np.max(readout_t), "__YEEEEEEEEHAAAAAAAAAW__"
                      )
            else:
            	print("TIMESTEP", t,
                      "/ STATE", state,
                      "/ EPSILON", self.epsilon,
                      "/ ACTION", action_index,
                      "/ REWARD", r_t,
                      "/ Q_MAX %e" % np.max(readout_t)
                      )
    # ------------------------------------------


def preprocess_image(x_rgb):
    x_1channel = cv2.cvtColor(cv2.resize(x_rgb, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, x_1channel = cv2.threshold(x_1channel, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(x_1channel, (80, 80, 1))


def train():
    sess = tf.InteractiveSession()
    dqn = DQN()
    input_, output_ = dqn.create_network()
    dqn.train_network(input_, output_, sess)


def main():
    train()


if __name__ == '__main__':
    main()
