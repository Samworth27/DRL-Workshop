ENV = 'CustomOffWorldDiscreteEnv-v0'
VERSION = "depth_only_v2"
CHECKPOINT = 132


from os import getcwd
import os
import json

import gym
import offworld_gym
import ray

import ray.rllib.algorithms.ppo as ppo
from offworld_gym.envs.common.channels import Channels


from modules.Environment import CustomOffWorldDiscreteEnv
from modules.Helpers import cleanDict, cleanList
from modules.Evaluate import runEvaluation



TRAINER = ppo
# DEFAULT_CONFIG = TRAINER.PPOConfig().framework("tf").rollouts(num_rollout_workers=0)
DEFAULT_CONFIG = TRAINER.DEFAULT_CONFIG.copy()
config = TRAINER.PPO.merge_trainer_configs(DEFAULT_CONFIG, {
    'env': ENV,
    'env_config': {
        'channel_type' : Channels.DEPTH_ONLY
    },
    'framework': 'tf',
    'lambda': 0.95,
    'kl_coeff': 0.5,
    'clip_rewards': True,
    'clip_param': 0.1,
    'vf_clip_param': 10.0,
    'entropy_coeff': 0.01,
    'train_batch_size': 1000,
    'rollout_fragment_length': 10,
    'sgd_minibatch_size': 500,
    'num_sgd_iter': 10,
    'num_workers': 0,
    'num_envs_per_worker': 1,
    'batch_mode': 'truncate_episodes',
    'observation_filter': 'NoFilter',
    'model': {
        'vf_share_layers': 'true'
    },
    'num_gpus': 1,
})

checkpoint_root = f'{getcwd()}/.checkpoints/{ENV}/{VERSION}'

checkpoint_file = f'{checkpoint_root}/checkpoint_{CHECKPOINT:06}/checkpoint-{CHECKPOINT}'

print("Starting Trainer")
algorithm = TRAINER.PPO(config=config)
print("Trainer Started")
print("restoring CHECKPOINT")
algorithm.restore(checkpoint_file)
print("Checkpoint Restored")


eval_results = runEvaluation(algorithm, ENV)

print("Saving Results")
path = f"{getcwd()}/.evaluations/{ENV}/{VERSION}/checkpoint_{CHECKPOINT:06}"
os.makedirs(path, exist_ok=True)
eval_iter = len(os.listdir(path))


with open(f'{path}/eval_{eval_iter}.json', 'w') as fp:
    json.dump(cleanList(eval_results, None), fp,  indent=4)
    
print("shutting down ray")
ray.shutdown()