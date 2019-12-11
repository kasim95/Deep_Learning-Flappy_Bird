from flappybird import *
from nn import *
import cv2
import numpy as np

# TODO: Refine train function
# TODO: write script to run only


def train():
    dqn = DeepQNetwork(
        num_actions=2,
        num_history=5000,
        frame_stack_depth=4,
        batch_size=32,
        learning_rate=0.001,
        initial_epsilon=0.05
    )

    try:
        dqn.load()
    except:
        pass

    game = FlappyBird()
    game_counter = 0
    frame_counter = 0
    k_frame = 4  # action every 4 frames
    #save_timestep = 1000

    try:
        while True:
            game_counter += 1
            frame_counter = 0
            dead = False

            # initial game state
            action = 0
            reward = 0
            frame, reward, dead = game.step_next_frame()
            frame = DeepQNetwork.preprocess_frame(game, frame)

            state_stack = np.stack((frame, frame, frame, frame))

            while not dead:
                if frame_counter % k_frame == 0:
                    # determine next action
                    action = dqn.choose_action(state_stack)
                else:
                    # reuse last action
                    pass

                frame, reward, dead = game.step_next_frame(True if action == 1 else False)
                frame = DeepQNetwork.preprocess_frame(game, frame)
                #print(f'Chose action: {action}, reward = {reward}')

                # remember what happened
                dqn.remember(state_stack, action, reward, frame, dead)

                # update state stack (drop oldest frame, append newest frame)
                # note: output layer will have frame_stack entries and we use the first one, so
                # the first one needs to be newest
                state_stack = np.roll(state_stack, shift=1, axis=0)
                state_stack[0] = frame
                frame_counter += 1

                # apply gradient descent step
                dqn.replay()

            print(f'Finished game {game_counter}: score was {game.score}')

            if game_counter % 20 == 0:
                dqn.save()

    except KeyboardInterrupt:
        dqn.save()




            # actions = np.zeros(ACTIONS)
            # action_index = 0  # by default set action 0 (do nothing) to 1
            # if counter % FRAME_PER_ACTION == 0:
            #     action_index = dqn.do_action(state)
            #     actions[action_index] = 1
            # else:
            #     actions[action_index] = 1
            # image_data_1, reward, terminate = game.frame_state(actions)
            # image_data_1 = cv2.resize(image_data_1, (80, 80))
            # image_data_1 = cv2.cvtColor(image_data_1, cv2.COLOR_BGR2GRAY)
            # ret, x_1 = cv2.threshold(image_data_1, 1, 255, cv2.THRESH_BINARY)
            #
            # x_1 = x_1 / 255.0
            # x_1 = np.reshape(x_1, (1, 80, 80, 1))
            # next_state = np.append(x_1, state[:, :, :, :3], axis=3)
            # next_state = np.reshape(next_state, (1, 80, 80, 4))
            #
            # dqn.remember(state, actions, reward, next_state, terminate)
            # state = next_state

            # if counter > observe:
            #     #dqn.replay()
            #     #dqn.decay_epsilon()
            #     dqn.epsilon = max(dqn.epsilon_min,  dqn.epsilon * dqn.epsilon_decay)
            #     #dqn.save()
            #
            #     # if counter % save_timestep == 0:
            #     #     dqn.save()




def main():
    train()


if __name__ == '__main__':
    main()


    # first_action = np.zeros(ACTIONS)
    # first_action[0] = 1
    # image_data, reward, terminate = game.frame_state(first_action)
    # FlappyBird.save_encoded_frame(image_data, 'encoded_original.png')
    #
    # DeepQNetwork.preprocess_frame(game, image_data)


    #image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    #ret, x = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY)


    # image_data = image_data / 255.0
    # image_data2 = np.array(image_data, dtype=np.float16)
    #
    # x = np.ones_like(image_data)
    # state = np.stack((image_data, image_data, image_data, image_data), axis=0)
    # state = np.reshape(state, (4, 80, 80, 3))
#
# print("Timestep: ", counter,
#       "/ Epsilon: ", dqn.epsilon,
#       "/ Action: ", action,
#       "/ Reward: ", reward
#       )
# counter += 1