import cv2
from flappybird import *
from brain import *
from flappybird import PLAYER_WIDTH, PLAYER_HEIGHT, PIPEGAPSIZE, PIPE_WIDTH, PIPE_VELOCITY, SCREENWIDTH, SCREENHEIGHT


def train(save_video_of_best=False):
    game = FlappyBird()
    game_counter = 0
    frame_counter = 0
    best_score = 0
    brain = Brain(bird_width=PLAYER_WIDTH,
                  bird_height=PLAYER_HEIGHT,
                  bird_x_velocity=PIPE_VELOCITY,
                  bird_y_acc=game.playerAccY,
                  bird_x=game.playerx,
                  gap_size=PIPEGAPSIZE, pipe_width=PIPE_WIDTH,
                  mem_size=1000000)

    try:
        brain.load()
    except:
        pass

    try:
        while True:
            game_counter += 1
            frame_counter = 0
            dead = False
            frames = []

            # initial game state
            action = 0
            reward = 0
            current_score = 0
            video = None
            dead, frame = game.step_next_frame()

            if save_video_of_best:
                frames.append(frame)

            while not dead:
                frame_data = FrameData(
                    bird_y=game.playery,
                    gap_y=game.get_next_pipe_gap_y(),
                    next_pipe_x=game.get_next_pipe_gap_x(),
                    bird_y_velocity=game.playerVelY
                )

                should_jump = brain.predict(frame_data)
                brain.store_memory(frame_data)

                dead, frame = game.step_next_frame(should_jump, False)

                if save_video_of_best:
                    if video is None:
                        if game.score > best_score:
                            print('creating new best video ...')

                            video = cv2.VideoWriter('best.avi', cv2.VideoWriter_fourcc(*'mp4v'),
                                                    30.0, (SCREENWIDTH, SCREENHEIGHT))

                            for f in frames:
                                video.write(cv2.cvtColor(f.transpose([1, 0, 2]), cv2.COLOR_RGB2BGR))

                            frames = []  # stream frames in, because the best agents will play a long time
                        else:
                            frames.append(frame)

                    if video is not None:
                        video.write(cv2.cvtColor(frame.transpose([1, 0, 2]), cv2.COLOR_RGB2BGR))

                current_score = game.score if not dead else current_score

                best_score = max(current_score, best_score)
                frame_counter += 1

                if frame_counter % 1000 == 0:
                    brain.learn()

            print(f'Finished game {game_counter}: score was {current_score}; best so far is {best_score}')
            brain.learn()

            if current_score == best_score and best_score > 100:  # don't save garbage networks
                brain.save()
                brain.save(f'best_{best_score}.h5')

            if video is not None:
                video.release()

            if game_counter % 5 == 0:
                brain.save()

    except KeyboardInterrupt:
        pass


def main():
    train(save_video_of_best=False)


if __name__ == '__main__':
    main()
