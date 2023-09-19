from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import time


env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()

print (env.buttons)
while True:
        action_space = env.action_space
        observation_space = env.observation_space
    
        
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(2)
        done = terminated or truncated

        
        time.sleep(.0166666667)
        if done:
            state = env.reset()
env.close()