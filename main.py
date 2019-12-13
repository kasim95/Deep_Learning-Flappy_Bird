from flappybird import *
from nn import *
import cv2
import numpy as np

# TODO: Refine train function
# TODO: write script to run only


def train():
    # s1 = np.ones((5, 5, 3))
    # s2 = np.ones((5, 5, 3)) * 2
    # s3 = np.ones((5, 5, 3)) * 3
    # s4 = np.ones((5, 5, 3)) * 4
    # s5 = np.ones((5, 5, 3)) * 5
    # s6 = np.ones((5, 5, 3)) * 6
    # s7 = np.ones((5, 5, 3)) * 7
    # s8 = np.ones((5, 5, 3)) * 8
    # s9 = np.ones((5, 5, 3)) * 9
    # s10 = np.ones((5, 5, 3)) * 10
    # s11 = np.ones((5, 5, 3)) * 11
    # s12 = np.ones((5, 5, 3)) * 12
    #
    # snew = np.ones_like(s12) * 255
    #
    # #stack = np.stack((s1, s2, s3, s4), axis=0)
    #
    # #state = np.zeros((5, 5, 4))
    # state = np.dstack((s1, s2, s3, s4))
    #
    # # state1 = np.roll(state, shift=1, axis=2)
    # # state2 = np.roll(state, shift=2, axis=2)
    # state3 = np.roll(state, shift=3, axis=2)
    # state3[:, :, 0:3] = snew

    #return

    dqn = DeepQNetwork(
        input_shape=(80, 80, 4),  # channels last
        num_actions=2,
        num_history=15000,
        frame_stack_depth=4,
        batch_size=32,
        learning_rate=0.001,
        initial_epsilon=0.25,
        epsilon_decay=0.98
    )

    try:
        dqn.load()
    except:
        pass  # todo: don't delete old model in case something actually went wrong

    game = FlappyBird()
    game_counter = 0
    frame_counter = 0
    k_frame = 2  # action every k frames
    best_score = 0
    #save_timestep = 1000

    # test frame stacking
    # f1 = np.ones((80, 80))
    # f2 = np.ones_like(f1) * 2
    # f3 = np.ones_like(f1) * 3
    # f4 = np.ones_like(f1) * 4
    # f5 = np.ones_like(f1) * 5
    #
    # stack = dqn.add_to_frame_stack(f1, None)
    # stack = dqn.add_to_frame_stack(f2, stack)
    # stack = dqn.add_to_frame_stack(f3, stack)
    # stack = dqn.add_to_frame_stack(f4, stack)
    # stack = dqn.add_to_frame_stack(f5, stack)

    try:
        while True:
            game_counter += 1
            frame_counter = 0
            dead = False

            # initial game state
            action = 0
            reward = 0
            current_score, best_score = 0, 0
            frame_stack = None

            next_frame, reward, dead = game.step_next_frame()
            next_frame = DeepQNetwork.preprocess_frame(game, next_frame)

            frame_stack = dqn.add_to_frame_stack(next_frame, frame_stack)

            while not dead:
                if frame_counter % k_frame == 0:
                    # determine next action
                    action = dqn.choose_action(frame_stack)
                else:
                    # reuse last action
                    pass

                current_score = game.score
                best_score = max(game.score, best_score)

                next_frame, reward, dead = game.step_next_frame(True if action == 1 else False)
                next_frame = DeepQNetwork.preprocess_frame(game, next_frame)
                #print(f'Chose action: {action}, reward = {reward}')

                # remember what happened
                dqn.remember(frame_stack, action, reward, next_frame, dead)

                # update state stack (drop oldest frame, append newest frame)
                # note: output layer will have frame_stack entries and we use the first one, so
                # the first one needs to be newest
                # state_stack = np.roll(state_stack, shift=1, axis=0)
                # state_stack[0] = frame
                # state_stack = np.roll(state_stack, shift=3, axis=2)
                # state_stack[:, :, 0:3] = frame
                frame_stack = dqn.add_to_frame_stack(next_frame, frame_stack)
                frame_counter += 1

                # apply gradient descent step
                dqn.replay()

            print(f'Finished game {game_counter}: score was {current_score}; best so far is {best_score}')

            if game_counter % 20 == 0:
                dqn.save()

            # if game_counter // 40 > best_score:
            #     dqn.epsilon = 0.05  # add more epsilon to explore

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