import ray.rllib.algorithms.ppo as ppo
from timeit import default_timer as timer
import numpy as np
import json
import ray
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


logging.captureWarnings(False)

print("Starting Ray instance")
ray.init(ignore_reinit_error=True,
        #  logging_level=logging.FATAL, log_to_driver=False
         )

ENV = "BreakoutNoFrameskip-v4"
VERSION = "ppo_v1"
TRAINER = ppo
DEFAULT_CONFIG = TRAINER.DEFAULT_CONFIG.copy()
NUMBER_OF_EPOCHS = 100000

LAST_CHECKPOINT = None


checkpoint_root = f'.checkpoints/{ENV}/{VERSION}'

config = TRAINER.PPO.merge_trainer_configs( DEFAULT_CONFIG, {
      'env': ENV,
      'env_config': {
        # 'render_mode': 'human',
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
        'model':{
            'vf_share_layers': 'true'
        },
        'num_gpus': 1
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
    
    seconds_running = round(running_time%60)
    minutes_running = round(running_time/60)
    hours_running = round(minutes_running/60)
    days_running = round(hours_running/24)
    print("\n------------------------------------------------------------------")
    print(f"finished epoch: {epoch} | running time: {round(days_running)}days {round(hours_running%24)} hrs {round(minutes_running%60)} min {round(running_time%60)}s | average completion: {round(average_time,2)}s")
    current = timer()
    min_reward = result["episode_reward_min"]
    max_reward = result["episode_reward_max"]
    mean_reward = result["episode_reward_mean"]

    print(f'Mean Reward: {mean_reward}\n Min: {min_reward} Max: {max_reward}')
    print(f'completed in {round(current - interim)} seconds')

    time_remaining = (NUMBER_OF_EPOCHS - epoch)*average_time
    
    seconds_remaining = time_remaining%60
    minutes_remaining = round(time_remaining/60)
    hours_remaining = round(minutes_remaining/24)
    days_remaining = round(hours_remaining/24)
    print(
        f'Estimated Time Remaining: {round(days_remaining)} days {round(hours_remaining % 24)} hours {round(minutes_remaining % 60)} min {round(seconds_remaining)}s')
    print("------------------------------------------------------------------")
    if epoch % 10 == 0:
        checkpoint = algorithm.save(checkpoint_root)
        with open(f'{checkpoint}/result.json', 'w') as fp:
            json.dump(cleanDict(result), fp,  indent=4)

ray.shutdown()
