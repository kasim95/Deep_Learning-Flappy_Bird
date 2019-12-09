from flappybird import *
from nn import *
import cv2
import numpy as np

# TODO: Refine train function
# TODO: write script to run only


def train():
    dqn = DeepQNetwork()
    # dqn.load()
    game = FlappyBird()
    counter = 0
    observe = 10000
    save_timestep = 1000

    first_action = np.zeros(ACTIONS)
    first_action[1] = 1
    image_data, reward, terminate = game.frame_state(first_action)
    image_data = cv2.resize(image_data, (80, 80))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    ret, x = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY)
    x = x / 255.0
    state = np.stack((x, x, x, x), axis=2)
    state = np.reshape(state, (1, 80, 80, 4))
    while True:
        actions = np.zeros(ACTIONS)
        action_index = 0  # by default set action 0 (do nothing) to 1
        if counter % FRAME_PER_ACTION == 0:
            action_index = dqn.do_action(state)
            actions[action_index] = 1
        else:
            actions[action_index] = 1
        image_data_1, reward, terminate = game.frame_state(actions)
        image_data_1 = cv2.resize(image_data_1, (80, 80))
        image_data_1 = cv2.cvtColor(image_data_1, cv2.COLOR_BGR2GRAY)
        ret, x_1 = cv2.threshold(image_data_1, 1, 255, cv2.THRESH_BINARY)
        x_1 = x_1 / 255.0
        x_1 = np.reshape(x_1, (1, 80,80, 1))
        state_1 = np.append(x_1, state[:, :, :, :3], axis=3)
        state_1 = np.reshape(state_1, (1, 80, 80, 4))

        dqn.remember(state, actions, reward, state_1, terminate)
        if counter > observe:
        	#dqn.replay_last_15()
        	dqn.replay()
        	dqn.decay_epsilon()
        # if reward == 1:
        # dqn.replay_last_15()
	    # dqn.save()
        if counter % save_timestep == 0:
            dqn.save()
        print("Timestep: ", counter,
              "/ Epsilon: ", dqn.epsilon,
              "/ Action: ", action_index,
              "/ Reward: ", reward
              )
        counter += 1

def main():
    train()


if __name__ == '__main__':
    main()
