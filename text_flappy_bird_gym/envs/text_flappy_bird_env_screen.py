import gymnasium as gym
import numpy as np

from text_flappy_bird_gym.envs.text_flappy_bird_logic import FlappyBirdLogic

class TextFlappyBirdEnvScreen(gym.Env):
  """
  Implementation of a simple Flappy Bird OpenAI Gym environment that yields the
  game's screen state as observations and renders each step in a text format.

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
               pipe_gap = 2,
               normalize_obs = True):
    self._screen_size = (width,height)
    self._pipe_gap = pipe_gap

    self._normalize_obs = normalize_obs
    self.action_space = gym.spaces.Discrete(2)
    self.action_space_lut = {0:'Idle', 1:'Flap'}
    self.observation_space = gym.spaces.Box(0, 3, [*self._screen_size], dtype=np.int32)
    self._game = None
    self._render = None

  def _get_observation(self):
    """
    The text array is returned as observation
    """
    self.render()
    return self._render

  def _get_info(self):
    obs = self._get_observation()
    info = {"score": self._game.score, 
            "player":[self._game.player_x, self._game.player_y],
            }
    return info

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
    return self._get_observation(), self._get_info()

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
           
    r = np.zeros((self._screen_size[0],self._screen_size[1]), dtype='int32')

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
    r_str += '('+self.action_space_lut[self._game.player_last_action] + ')\n'
    return r_str

  def close(self):
    pass