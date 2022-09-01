from io import BytesIO
import gym
from modules.Helpers import encodeArray
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions


def runEvaluation(algorithm, env_string, max_step = 100, config ={'channel_type':Channels.DEPTH_ONLY, 'random_init': True}):
  env = gym.make(env_string, config=config)
  obs = env.reset()
  done = False
  step = 0
  frame, frame_shape = encodeArray(obs)
  history = [dict(step=step, frame = frame, frame_shape = frame_shape, action=None, distance=env.getDistance())]
  while not done and step < max_step:
    action = algorithm.compute_action(obs)
    frame, frame_shape = encodeArray(obs)

    
    obs, reward, done, info = env.step(action)
    
    step_data = dict(step=step,  action=action, reward=reward, distance=env.getDistance(),frame = frame, frame_shape = frame_shape,)
    
    history.append(step_data)
    print(step, reward, FourDiscreteMotionActions(action), done, info)
    step += 1

  
  return history