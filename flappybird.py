# ---------------------------------------------------------
# dependencies
import pygame
import numpy as np
import random
from itertools import cycle
from pygame.locals import *

# ---------------------------------------------------------
# module objects
__all__ = [
    'gethitmask',
    'FlappyBird',
]


# ---------------------------------------------------------
# helper function
def gethitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask
# ---------------------------------------------------------


# LOCALS
FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79


# ---------------------------------------------------------
# pygame
pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('AI - Flappy Bird')


# ---------------------------------------------------------
# load sprites
IMAGES = {}
HITMASKS = {}

bird_path = (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
)
bg_path = 'assets/sprites/background-black.png'
base_path = 'assets/sprites/base.png'
pipe_path = 'assets/sprites/pipe-green.png'

IMAGES['base'] = pygame.image.load(base_path).convert_alpha()
IMAGES['background'] = pygame.image.load(bg_path)
IMAGES['player'] = (
    pygame.image.load(bird_path[0]).convert_alpha(),
    pygame.image.load(bird_path[1]).convert_alpha(),
    pygame.image.load(bird_path[2]).convert_alpha(),
)
IMAGES['pipe'] = (
    pygame.transform.flip(pygame.image.load(pipe_path), False, True).convert_alpha(),
    pygame.image.load(pipe_path).convert_alpha(),
)

HITMASKS['pipe'] = (
    gethitmask(IMAGES['pipe'][0]),
    gethitmask(IMAGES['pipe'][1]),
)

HITMASKS['player'] = (
    gethitmask(IMAGES['player'][0]),
    gethitmask(IMAGES['player'][1]),
    gethitmask(IMAGES['player'][2]),
)


# ---------------------------------------------------------
# sprite dimensions
PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_height()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()
BACKGROUND_HEIGHT = IMAGES['background'].get_height()
BASE_WIDTH = IMAGES['base'].get_width()
BASE_HEIGHT = IMAGES['base'].get_height()
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


# ---------------------------------------------------------
# class to keep track of state of game
class FlappyBird:
    def __init__(self):
        self.score = 0
        self.playerIndex = 0
        self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = BASE_WIDTH - BACKGROUND_WIDTH
        newpipe1 = self.getrandompipe()
        newpipe2 = self.getrandompipe()
        self.upperpipes = [
            {
                'x': SCREENWIDTH, 'y': newpipe1[0]['y']
            },
            {
                'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newpipe2[0]['y']
            },
        ]
        self.lowerpipes = [
            {
                'x': SCREENWIDTH, 'y': newpipe1[1]['y']
            },
            {
                'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newpipe1[1]['y']
            },
        ]

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.pipeVelX = -4
        self.playerVelY = 0  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward acceleration
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

    def frame_state(self, ip_actions):
        pygame.event.pump()
        reward = 0.01
        terminate = False

        if sum(ip_actions) != 1:
            raise ValueError('Multiple input actions')

        # ip_actions[0] == 1: do nothing
        # ip_actions[1] == 1: flap the bird
        if ip_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        # add score
        playermidpos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperpipes:
            pipemidpos = pipe['x'] + PIPE_WIDTH / 2
            if pipemidpos <= playermidpos < pipemidpos + 4:
                self.score += 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # moves pipes to left
        for uPipe, lPipe in zip(self.upperpipes, self.lowerpipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperpipes[0]['x'] < 5:
            newpipe = self.getrandompipe()
            self.upperpipes.append(newpipe[0])
            self.lowerpipes.append(newpipe[1])

        # remove first pipe if its out of screen
        if self.upperpipes[0]['x'] < -PIPE_WIDTH:
            self.upperpipes.pop(0)
            self.lowerpipes.pop(0)

        # check if crash here
        player_info = {'x': self.playerx, 'y': self.playery, 'index': self.playerIndex}
        if self.checkcrash(player=player_info, upperpipes=self.upperpipes, lowerpipes=self.lowerpipes):
            terminate = True
            self.__init__()
            reward = -1

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperpipes, self.lowerpipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))

        SCREEN.blit(IMAGES['player'][self.playerIndex], (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        return image_data, reward, terminate

    @staticmethod
    def getrandompipe():
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        # gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
        fixed_gapy = [30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(fixed_gapy) - 1)
        gapy = fixed_gapy[index]
        gapy += int(BASEY * 0.2)
        pipeheight = IMAGES['pipe'][0].get_height()
        pipex = SCREENWIDTH + 10

        return [
            {'x': pipex, 'y': gapy - pipeheight},       # upper pipe
            {'x': pipex, 'y': gapy + PIPEGAPSIZE},      # lower pipe
        ]

    @staticmethod
    def checkcrash(player, upperpipes, lowerpipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']

        # if player crashes into ground
        if player['y'] + PLAYER_HEIGHT >= BASEY - 1:
            return True
        else:
            playerrect = pygame.Rect(player['x'], player['y'],
                                     PLAYER_WIDTH, PLAYER_HEIGHT)

            for upipe, lpipe in zip(upperpipes, lowerpipes):
                # upper and lower pipe rects
                upiperect = pygame.Rect(upipe['x'], upipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
                lpiperect = pygame.Rect(lpipe['x'], lpipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

                # player and upper/lower pipe hitmasks
                phitmask = HITMASKS['player'][pi]
                uhitmask = HITMASKS['pipe'][0]
                lhitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                ucollide = FlappyBird.pixelcollision(playerrect, upiperect, phitmask, uhitmask)
                lcollide = FlappyBird.pixelcollision(playerrect, lpiperect, phitmask, lhitmask)

                if ucollide or lcollide:
                    return True
        return False

    @staticmethod
    def pixelcollision(rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False