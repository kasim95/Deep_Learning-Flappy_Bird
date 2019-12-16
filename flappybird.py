# ---------------------------------------------------------
# dependencies
import pygame
import random
from itertools import cycle
from pygame.locals import *


# ---------------------------------------------------------
# module objects
__all__ = [
    'gethitmask',
    'FlappyBird',
    'PLAYER_HEIGHT',
    'PIPEGAPSIZE',
    'PIPE_WIDTH'
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
#PIPEGAPSIZE = 200
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
#bg_path = 'assets/sprites/background-black.png'
bg_path = 'assets/sprites/background-day.png'
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
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()
BACKGROUND_HEIGHT = IMAGES['background'].get_height()
BASE_WIDTH = IMAGES['base'].get_width()
BASE_HEIGHT = IMAGES['base'].get_height()
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
PIPE_VELOCITY = -4  # this game moves the pipes rather than the player


class Pipe:
    def __init__(self, gap_center_y):
        self.x = SCREENWIDTH + 10
        self.uppery = gap_center_y - PIPEGAPSIZE * 0.5 - PIPE_HEIGHT
        self.lowery = gap_center_y + PIPEGAPSIZE * 0.5
        self.gap_center_y = gap_center_y
        self.upperpipe = IMAGES['pipe'][0]
        self.lowerpipe = IMAGES['pipe'][1]

    def blit(self, screen):
        screen.blit(self.upperpipe, (self.x, self.uppery))
        screen.blit(self.lowerpipe, (self.x, self.lowery))

    @property
    def upipe_rect(self):
        return pygame.Rect(self.x, self.uppery, PIPE_WIDTH, PIPE_HEIGHT)

    @property
    def lpipe_rect(self):
        return pygame.Rect(self.x, self.lowery, PIPE_WIDTH, PIPE_HEIGHT)


class Scoreboard:
    font = None

    def __init__(self):
        Scoreboard.font = Scoreboard.font or pygame.sysfont.SysFont(None, 32, True)

        self.position = (10, 10)
        self.score = 0
        self._update_surface()

    def increase(self):
        self.score += 1
        self._update_surface()

    def _update_surface(self):
        self.image = Scoreboard.font.render(str(self.score), True, pygame.Color('yellow'))

    def blit(self, screen):
        screen.blit(self.image, self.position)


# ---------------------------------------------------------
# class to keep track of state of game
class FlappyBird:
    def __init__(self):
        self.scoreboard = Scoreboard()
        self.playerIndex = 0
        self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = BASE_WIDTH - BACKGROUND_WIDTH
        self.pipes = [self.getrandompipe(), self.getrandompipe()]

        self.pipes[0].x = SCREENWIDTH
        self.pipes[1].x = SCREENWIDTH * 1.5

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.playerVelY = 0  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward acceleration
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

    @property
    def score(self):
        return self.scoreboard.score

    def get_next_pipe(self):
        player_rect = pygame.Rect(self.playerx, self.playery, PLAYER_WIDTH, PLAYER_HEIGHT)

        pipe_width = PIPE_WIDTH

        for pipe in self.pipes:
            if pipe.x + pipe_width < player_rect.left:
                continue  # passed this pipe already, ignore

            next_pipe_location = pipe.x + PIPE_VELOCITY

            if next_pipe_location + pipe_width < player_rect.left:
                continue  # will pass this pipe by next frame, couldn't possibly hit it

            # if we jump every frame, will we hit the pipe?
            # remember jumping is -y and pipe velocity is negative
            frames_to_go = round((pipe.x + pipe_width - player_rect.left) / -PIPE_VELOCITY)
            height_can_increase = frames_to_go * -self.playerFlapAcc

            if player_rect.top - height_can_increase > pipe.gap_center_y - PIPEGAPSIZE * 0.5:
                # can't possibly hit it by jumping. How about falling?
                height_can_fall = self.playerVelY * frames_to_go + 0.5 * self.playerAccY * frames_to_go * frames_to_go

                if player_rect.bottom + height_can_fall < pipe.gap_center_y + PIPEGAPSIZE * 0.5:
                    continue  # couldn't possibly fall in time to hit this pipe

            return pipe

        return self.pipes[-1:][0]  # last pipe I guess, something's probably broken though

    def get_next_pipe_gap_y(self):
        return self.get_next_pipe().gap_center_y

    def get_next_pipe_gap_x(self):
        return self.get_next_pipe().x

    def get_bird_y(self):
        return self.playery + PLAYER_HEIGHT / 2.

    def step_next_frame(self, jump=False, enforce_frame_rate=False):
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
            elif evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE:
                pygame.quit()

        terminate = False

        if jump:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

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
        self.playery += self.playerVelY
        if self.playery < 0:
            terminate = True
            self.__init__()

        # moves pipes to left
        for pipe in self.pipes:
            pipe.x += PIPE_VELOCITY

        # add new pipe when first pipe is about to touch left of screen
        # note: bug in original version: can double-stack pipes if pipe does not move at
        # least this many units per frame, which in original version it does not
        if self.pipes:
            if 0 <= self.pipes[0].x < 5 and self.pipes[-1:][0].x < SCREENWIDTH:
                newpipe = self.getrandompipe()
                self.pipes.append(newpipe)

            if self.pipes[0].x < -PIPE_WIDTH:
                self.pipes.pop(0)

        # check if crash here
        player_info = {'x': self.playerx, 'y': self.playery, 'index': self.playerIndex}
        if self.checkcrash(player=player_info):
            terminate = True
            self.__init__()
        else:
            # add score
            playermidpos = self.playerx + PLAYER_WIDTH / 2

            for pipe in self.pipes:
                pipemidpos = pipe.x + PIPE_WIDTH / 2
                if pipemidpos <= playermidpos < pipemidpos + 4:
                    self.scoreboard.increase()
                    break

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))

        for pipe in self.pipes:
            pipe.blit(SCREEN)

        # # visualize gap
        # gapr = pygame.Rect(self.get_next_pipe().x, self.get_next_pipe_gap_y() - PIPEGAPSIZE * 0.5, PIPE_WIDTH, PIPEGAPSIZE)
        # SCREEN.fill((0, 120, 0), gapr)
        #
        # # visualize target line
        # y = self.get_next_pipe_gap_y() + PIPEGAPSIZE * 0.5 - PLAYER_HEIGHT
        # r = pygame.Rect(self.get_next_pipe_gap_x(), y, PIPE_WIDTH, 4)
        # SCREEN.fill((255, 0, 0), r)
        #
        # # visualize bottom of pipe
        # y = self.get_next_pipe().gap_center_y + PIPEGAPSIZE * 0.5
        # r.y = y
        # SCREEN.fill((255, 0, 255), r)

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        SCREEN.blit(IMAGES['player'][self.playerIndex], (self.playerx, self.playery))

        self.scoreboard.blit(SCREEN)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()

        # todo: have caller do this, else game will always run < FPS
        if enforce_frame_rate:
            FPSCLOCK.tick(FPS)

        return terminate, image_data

    def frame_state_player_only(self):
        SCREEN.blit(IMAGES['background'], (0, 0))
        SCREEN.blit(IMAGES['player'][self.playerIndex], (self.playerx, self.playery))

        #pygame.display.flip()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        return image_data

    @staticmethod
    def save_encoded_frame(frame, name):
        assert frame is not None
        assert name is not None

        num_channels = frame.shape[-1:]

        if type(num_channels) is not tuple:
            num_channels = (num_channels,)

        if num_channels[0] > 3:
            frame = frame[:, :, None].repeat(3, axis=2)
        elif num_channels[0] != 1 and num_channels[0] != 3:
            print('frame must have 1 or 3 color channels')
            raise RuntimeError

        s = pygame.surfarray.make_surface(frame)
        pygame.image.save(s, name)

    def getrandompipe(self):
        MIN_Y = 0.1 * SCREENHEIGHT + PIPEGAPSIZE * 0.5
        MAX_Y = BASEY - 0.1 * SCREENHEIGHT - PIPEGAPSIZE * 0.5

        while 1:
            gapY = random.randrange(0, int(MAX_Y - MIN_Y)) + MIN_Y
            pipex = SCREENWIDTH + 10

            if hasattr(self, 'pipes') and len(self.pipes) > 1:
                # if the gap is too far from the last pipe, it might be literally impossible
                # to manage the transition even with perfect reflexes. So choose something that's technically possible
                last_pipe = self.pipes[-1:][0]

                distance_between_pipes = abs(last_pipe.x - self.pipes[-2:1][0].x)
                frames = round(max(0, distance_between_pipes - PLAYER_WIDTH) / -PIPE_VELOCITY)

                if gapY > last_pipe.gap_center_y:
                    # lower. Note: since y velocity is clamped, we have to solve this iteratively
                    # Also, assume player is in worst possible position to make this

                    fall_distance = 0.0
                    vel = self.playerFlapAcc
                    last_pipe_y = last_pipe.gap_center_y - PIPEGAPSIZE * 0.5 + PLAYER_HEIGHT  # assume player at top of this pipe

                    for _ in range(frames):
                        vel = min(self.playerMaxVelY, vel + self.playerAccY)
                        fall_distance += vel

                    if gapY - last_pipe_y < fall_distance * 0.65:  # fudge factor to account for hitting sides of pipes
                        # it's possible, we'll allow it
                        break
                else:
                    # must ascend. Similar logic as before, except now we can assume the player mashes
                    # jump as fast as possible to maximize their velocity
                    ascend_distance = frames * -self.playerFlapAcc

                    if last_pipe.gap_center_y - ascend_distance * 0.65 < gapY - PIPEGAPSIZE * 0.5:  # player can reach higher than this new pipe
                        break
            else:
                break

        return Pipe(gapY)

    def checkcrash(self, player):
        """returns True if player collders with base or pipes."""
        pi = player['index']

        # if player crashes into ground
        if player['y'] + PLAYER_HEIGHT >= BASEY - 1:
            return True
        else:
            playerrect = pygame.Rect(player['x'], player['y'],
                                     PLAYER_WIDTH, PLAYER_HEIGHT)

            for pipe in self.pipes:
                # upper and lower pipe rects
                upiperect = pipe.upipe_rect
                lpiperect = pipe.lpipe_rect

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
