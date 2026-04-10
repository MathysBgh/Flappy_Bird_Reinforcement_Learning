import gymnasium as gym
import numpy as np

from text_flappy_bird_gym.envs.text_flappy_bird_logic import FlappyBirdLogic

class TextFlappyBirdEnvSimple(gym.Env):
  """
  Implementation of a simple Flappy Bird OpenAI Gym environment that yields simple
  numerical information about the game's state as observations and renders each
  step in a text format.

  borrows from: https://github.com/Talendar/flappy-bird-gym

  @ denotes the bird
  [ left border of the screen
  ] right border of the screen
  - top border
  ^ bottom border

  Text Flappy Bird!
  Score: 0
  -----------------
  [     |       | ]
  [    @|       | ]
  [     |         ]
  [               ]
  [               ]
  [               ]
  [             | ]
  [     |       | ]
  [     |       | ]
  [     |       | ]
  [     |       | ]
  ^^^^^^^^^^^^^^^^^
  """

  metadata = {'render_modes': ['human'], "render_fps": 4}

  def __init__(self, 
               height = 15,
               width = 20,
               pipe_gap = 4):
    self._screen_size = (width,height)
    self._pipe_gap = pipe_gap

    self.action_space = gym.spaces.Discrete(2)
    self.action_space_lut = {0:'Idle', 1:'Flap'}

    # These are the max and min values of the observation space
    x_dist_max = self._screen_size[0]-int(self._screen_size[0]*0.3)-1
    x_dist_min = 0
    # The bird can temporarily reach y == height before the episode ends, so the
    # declared observation space needs to cover one extra cell at the bottom.
    y_dist_min = -(self._screen_size[1] - self._pipe_gap - 1 + int(self._pipe_gap // 2))
    y_dist_max = self._screen_size[1] - 1 - int(self._pipe_gap // 2)

    self.observation_space = gym.spaces.Tuple(
      (
        gym.spaces.Discrete(x_dist_max - x_dist_min + 1),
        gym.spaces.Discrete(y_dist_max - y_dist_min + 1, start=y_dist_min),
      )
    )
    self._game = None

  def _get_observation(self):
    """
    The horizontal and vertical distance between the player and the center of the gap is returned as observation
    """
    closest_upcoming_pipe = min([i for i,p in enumerate([pipe['x'] - self._game.player_x for pipe in self._game.upper_pipes]) if p>=0])
    x_dist = self._game.upper_pipes[closest_upcoming_pipe]['x'] - self._game.player_x
    y_dist = self._game.player_y-self._game.upper_pipes[closest_upcoming_pipe]['y']-self._pipe_gap//2

    return (x_dist,y_dist)

  def _get_info(self):
    obs = self._get_observation()
    return {
      "score": self._game.score, 
      "player":[self._game.player_x, self._game.player_y],
      "distance": np.sqrt(obs[0]**2+obs[1]**2),
    }

  def step(self, action):
    """
    Given an action it returns
    * an observations
    * a reward
    * a status report if the game is over
    * an info dictionary
    """

    alive = self._game.update_state(action) # this needs to return True/False if the player is alive
    done = not alive
    obs = self._get_observation()

    reward = 1 # As long as it stays alive the cummulative reward is increased

    info = self._get_info()
    return obs, reward, done, False, info


  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self._game = FlappyBirdLogic(self._screen_size, self._pipe_gap)
    obs = self._get_observation()
    info = self._get_info()
    return obs, info

  def render(self, mode='human'):
    """
    0 : background (' ')
    1 : player ('@')
    2 : pipe ('|')
    3 : dead player ('*')
    4 : left wall ('[')
    5 : right wall (']')
    6 : sky ('-')
    7 : floor ('^')
    """
    lut = {0:' ', 
           1:gym.utils.colorize('@',"yellow"),
           2:gym.utils.colorize('|',"green"),
           3:gym.utils.colorize('*',"red"),
           4:'[',
           5:']',
           6:'-',
           7:'^'}

    r = np.zeros((self._screen_size[0],self._screen_size[1]), dtype='int')

    if self._game.player_alive:
      r[self._game.player_x,self._game.player_y] = 1 # player alive
    else:
      r[self._game.player_x-1,min(self._game.player_y,self._screen_size[1]-1)] = 3 # player dead

    for pipe in self._game.upper_pipes:
      for y in range(pipe['y']):
          r[pipe['x'],y] = 2
    for pipe in self._game.lower_pipes:
      for y in range(pipe['y'], self._screen_size[1]):
          r[pipe['x'],y] = 2
    self._render = r
    r = np.flipud(np.rot90(r,1))
    r = np.pad(r, 1, mode='constant',constant_values=(4,5))
    r[0][:] = 6
    r[-1][:] = 7
    r_str = 'Text Flappy Bird!\nScore: {}\n'.format(self._game.score)
    for i in range(r.shape[0]):
      for j in range(r.shape[1]):
        r_str += lut[r[i,j]]
      r_str += '\n'
    r_str += 'Player Action ({})\n'.format(self.action_space_lut[self._game.player_last_action])

    (dx,dy) = self._get_observation()
    r_str += 'Distance From Pipe (dx={},dy={})\n'.format(dx,dy)

    return r_str

  def close(self):
    pass
