from flappybird import *
#from nn import *
import cv2
import numpy as np
from brain import *
#from hydra import *

# TODO: Refine train function
# TODO: write script to run only


def train():
    game = FlappyBird()
    game_counter = 0
    frame_counter = 0
    #k_frame = 2  # action every k frames
    best_score = 0
    #delay_training_until_games_played = 100

    from flappybird import PLAYER_HEIGHT, PIPEGAPSIZE

    simple = Brain(bird_size=PLAYER_HEIGHT, gap_size=PIPEGAPSIZE)

    #hydra = Hydra()

    begin_hydra = False

    try:
        while True:
            game_counter += 1
            frame_counter = 0
            dead = False

            # initial game state
            action = 0
            reward = 0
            current_score = 0
            frame_stack = None  # clear old stack, if any

            next_frame, reward, dead = game.step_next_frame()
            #next_frame = Hydra.preprocess_frame(next_frame)

            #frame_stack = Hydra.add_to_frame_stack(next_frame, None)

            while not dead:
                current_bird_y, target_gap_y, next_pipe_x = game.get_bird_y(), game.get_next_pipe_gap_y(), game.get_next_pipe_gap_x()
                #hydra.predict(next_frame, current_bird_y)
                should_jump = simple.should_jump(next_frame, current_bird_y, target_gap_y, next_pipe_x)
                simple.store_memory(next_frame, current_bird_y, target_gap_y, next_pipe_x)

                next_frame, _, dead = game.step_next_frame(should_jump, False)
                #next_frame = Hydra.preprocess_frame(next_frame)
                #Hydra.add_to_frame_stack(next_frame, frame_stack)

                current_score = game.score if not dead else current_score
                best_score = max(current_score, best_score)
                frame_counter += 1

                if frame_counter % 1000 == 0:
                    simple.learn()

            print(f'Finished game {game_counter}: score was {current_score}; best so far is {best_score}')
            simple.learn()

    except KeyboardInterrupt:
        pass


def main():
    train()


if __name__ == '__main__':
    main()
