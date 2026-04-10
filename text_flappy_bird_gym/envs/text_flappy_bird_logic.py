import random
from gymnasium import logger

class FlappyBirdLogic:
  def __init__(self, screen_size, pipe_gap_size = 2):
    # Game Configuration
    self._screen_width = screen_size[0]
    self._screen_height = screen_size[1]
    
    if not screen_size[1]-pipe_gap_size-2<0:
      self._pipe_gap_size = pipe_gap_size
    else:
      logger.error('Pipe gap value too high. Pipe gap cannot exceed the height of the game. At least one element should be placed at each side. (Max Value: Height-2)')
      exit(0)

    # Initial position of player
    self.player_x = int(self._screen_width * 0.3)
    self.player_y = int(self._screen_height / 2)

    # Initial random pipe
    new_pipe = self._get_random_pipe()
    self.upper_pipes = [new_pipe[0]]
    self.lower_pipes = [new_pipe[1]]

    # Game score
    self.score = 0

    # Player's info:
    self.player_alive = True      # player is alive at the beginning of the game
    self.player_vel_y = 0         # player's velocity along Y (default = 0)
    self.player_gravity = 2       # player's downward velocity (default = 2)
    self.player_vel_flap = -1     # player's flap velocity (default = -1)
    self.player_last_action = 0   # player's initial action value (idle)

  def _get_random_pipe(self):
    """ Returns a randomly sampled pipe """
    top_y = random.randrange(1, self._screen_height - self._pipe_gap_size)
    return [{"x":self._screen_width-1, "y":top_y}, {"x":self._screen_width-1, "y":top_y+self._pipe_gap_size}]

  def _check_crash(self):
    # player dies if drops to the ground
    if self.player_y > self._screen_height-1:
      self.player_alive = False
    # player dies if reaches pipe and is not within the range of the pipe gap
    if self.player_x == self.upper_pipes[0]['x'] and self.player_y not in range(self.upper_pipes[0]['y'],(self.upper_pipes[0]['y'] + self._pipe_gap_size)):
      self.player_alive = False

  def update_state(self, action):
    # stores the action
    self.player_last_action = action

    # when first pipe crosses the center of the screen add another one
    if self.upper_pipes[0]['x'] == int(self._screen_width/2)-1:
      new_pipe = self._get_random_pipe()
      self.upper_pipes += [new_pipe[0]]
      self.lower_pipes += [new_pipe[1]]

    # Shift pipes
    for pipe in self.upper_pipes:
      pipe['x'] -= 1
    for pipe in self.lower_pipes:
      pipe['x'] -= 1

    # if action is flap (==1) then change velocity to flap velocity
    if action:
      self.player_vel_y = self.player_vel_flap

    # move player using current velocity
    self.player_y += self.player_vel_y
    self.player_y = int(max(self.player_y, 0))
    self.player_y = int(min(self.player_y, self._screen_height))
    self.player_vel_y += self.player_gravity

    # checks if the player is still alive after this step.
    self._check_crash()

    # if player is alive and firt pipe is behind score a point
    if self.player_alive and self.player_x == self.upper_pipes[0]['x'] + 1:
      self.score += 1

    # remove first pipe if its out of the screen
    if (len(self.upper_pipes) > 0 and self.upper_pipes[0]['x'] < 0):
        self.upper_pipes.pop(0)
        self.lower_pipes.pop(0)

    return self.player_alive