from gymnasium.envs.registration import register

register(
    id='TextFlappyBird-v0',
    entry_point='text_flappy_bird_gym.envs:TextFlappyBirdEnvSimple'
)

register(
    id='TextFlappyBird-screen-v0',
    entry_point='text_flappy_bird_gym.envs:TextFlappyBirdEnvScreen'
)