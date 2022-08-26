import ray.rllib.algorithms.ppo as ppo
from timeit import default_timer as timer
import numpy as np
import json
import ray
import math
import os
from lib2to3.pgen2.token import NUMBER
import logging
logging.captureWarnings(True)


def cleanValue(value):
    valueType = type(value)
    if (valueType == dict):
        value = cleanDict(value)
    elif (valueType == list):
        value = cleanList(value)
    elif (valueType) == float:
        value == float(round(value, 5))
    elif (valueType in [str, int, bool, type(None)]):
        pass
    elif (valueType in [np.float32, np.float64]):
        value = float(round(value, 5))
    elif (valueType == np.ndarray):
        value = cleanList(value)
    elif (valueType == type(TRAINER.DEFAULT_CONFIG)):
        value == cleanDict(value)
    else:
        value = str(value)
    return value


def cleanList(dirtyList):
    cleanList = []
    for value in dirtyList:
        cleanList.append(cleanValue(value))
    return cleanList


def cleanDict(dictionary):
    for key, value in zip(dictionary.keys(), dictionary.values()):
        dictionary[key] = cleanValue(value)
    return dictionary

def timeString(time):
  seconds_elapsed = math.floor(time)
  minutes_elapsed = math.floor(seconds_elapsed/60)
  hours_elapsed = math.floor(minutes_elapsed/60)
  days_elapsed = math.floor(hours_elapsed/24)
  return f'{days_elapsed} days  {(hours_elapsed%24):02} hours  {(minutes_elapsed%60):02} mins  {(seconds_elapsed%60):02} seconds'

logging.captureWarnings(False)

print("Starting Ray instance")
ray.init(ignore_reinit_error=True,
         #  logging_level=logging.FATAL, log_to_driver=False
         )

ENV = "LunarLander-v2"
VERSION = "ppo_v1"
TRAINER = ppo
DEFAULT_CONFIG = TRAINER.DEFAULT_CONFIG.copy()
NUMBER_OF_EPOCHS = 5

LAST_CHECKPOINT = 500
EVAL_INTERVAL = 1

checkpoint_root = f'.checkpoints/{ENV}/{VERSION}'

config = TRAINER.PPO.merge_trainer_configs(DEFAULT_CONFIG, {
    'env': ENV,
    'env_config': {
        'continuous': True,
    },
    # 'record_env': True,
    'framework': 'tf',
    'lambda': 0.95,
    'kl_coeff': 0.5,
    'clip_rewards': True,
    'clip_param': 0.1,
    'vf_clip_param': 10.0,
    'entropy_coeff': 0.01,
    'train_batch_size': 5000,
    'rollout_fragment_length': 100,
    'sgd_minibatch_size': 500,
    'num_sgd_iter': 10,
    'num_workers': 10,
    'num_envs_per_worker': 5,
    'batch_mode': 'truncate_episodes',
    'observation_filter': 'NoFilter',
    'model': {
        'vf_share_layers': 'true'
    },
    'num_gpus': 1,
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
})

print(f'Checkpoints Stored at {checkpoint_root}')


print("Starting Trainer")
algorithm = TRAINER.PPO(config=config)
logging.captureWarnings(False)
print("Trainer Started")


if LAST_CHECKPOINT:
    print(f"Loading Checkpoint {LAST_CHECKPOINT}")
    checkpoint_file = f'{checkpoint_root}/checkpoint_{LAST_CHECKPOINT:06}/checkpoint-{LAST_CHECKPOINT}'
    algorithm.restore(checkpoint_file)


start = timer()

for epoch in range(1, NUMBER_OF_EPOCHS+1):

    interim = timer()
    running_time = interim - start
    average_time = running_time/(epoch+1)

    result = algorithm.train()
    os.system('clear')


    print("\n------------------------------------------------------------------")
    print(f"finished epoch: {epoch} | {timeString(running_time)} | average completion: {round(average_time,2)}s")
    current = timer()
    min_reward = result["episode_reward_min"]
    max_reward = result["episode_reward_max"]
    mean_reward = result["episode_reward_mean"]

    print(f'Mean Reward: {mean_reward}\n Min: {min_reward} Max: {max_reward}')
    print(f'completed in {round(current - interim)} seconds')

    time_remaining = (NUMBER_OF_EPOCHS+1 - epoch)*average_time

    print(
        f'Estimated Time Remaining: {timeString(time_remaining)}')
    print("------------------------------------------------------------------")
    if epoch % 10 == 0:
        
        checkpoint = algorithm.save(checkpoint_root)
        with open(f'{checkpoint}/result.json', 'w') as fp:
            json.dump(cleanDict(result), fp,  indent=4)
        
    
    if EVAL_INTERVAL:
        if epoch % EVAL_INTERVAL == 0:
            eval = algorithm.evaluate()
            with open(f'{checkpoint}/eval.json', 'w') as fp:
                json.dump(cleanDict(eval), fp,  indent=4)
                
ray.shutdown()
