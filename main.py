from flappybird import *
from nn import *
import cv2
import numpy as np

# INCOMPLETE
# TODO: write script to train and then run
def main():
    epochs = 1000
    dqn = DeepQNetwork()
    game = FlappyBird()
    log1 = open("logs_" + GAME + "/readout.txt", 'w')
    log2 = open("logs_" + GAME + "/hidden.txt", 'w')

    # setup first action
    first_action = np.zeros(ACTIONS)
    first_action[0] = 1
    for epoch in range(epochs):
        image_data, reward, terminate = game.frame_state(first_action)
        image_data = cv2.cvtColor(cv2.resize(image_data, (80, 80), cv2.COLOR_BGR2GRAY))
        ret, x = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY)
        state = np.stack((x, x, x, x), axis=2)


if __name__ == '__main__':
    main()
