import ray.rllib.algorithms.dqn as dqn
from timeit import default_timer as timer
import numpy as np
from modules.Train import runTraining
import ray
from os import getcwd
from offworld_gym.envs.common.channels import Channels

print("Starting Ray instance")
ray.shutdown()
ray.init(ignore_reinit_error=True,
        #  logging_level=logging.FATAL, log_to_driver=False
         )

ENV = "CustomOffWorldDockerMonolithDiscreteSim-v0"
VERSION = "dqn_depth_v1"
TRAINER = dqn
NUMBER_OF_EPOCHS = 100

LAST_CHECKPOINT = None
SAVE_INTERVAL = 1
RESULTS_INTERVAL = 1
EVAL_INTERVAL = None

CHECKPOINT_ROOT = f'{getcwd()}/.checkpoints/{ENV}/{VERSION}'


config = TRAINER.DEFAULT_CONFIG.copy()
config['env'] = ENV
config['framework'] = 'tf'
config['env_config']['channel_type'] = Channels.DEPTH_ONLY
config["num_gpus"] = 1
config["num_workers"] = 3
config["num_cpus_per_worker"] = 1
config['target_network_update_freq'] = 8000
config['replay_buffer_config']['capacity'] = 9000
config['log_level'] = "ERROR"
config['n_step'] = 1
config['lr'] =  .00001
config['adam_epsilon'] = 0.00015
config['hiddens'] = [256,256]
config['rollout_fragment_length'] = 4
config['train_batch_size'] = 32
config['exploration_config']['epsilon_timesteps'] = 2000000
config['exploration_config']['final_epsilon'] = 0.01
config['min_sample_timesteps_per_iteration'] = 10000

print(f'Checkpoints Stored at {CHECKPOINT_ROOT}')


print("Starting Trainer")
algorithm = TRAINER.DQN(config=config)
print("Trainer Started")

print(f'Checkpoints Stored at {CHECKPOINT_ROOT}')


runTraining(algorithm, NUMBER_OF_EPOCHS,CHECKPOINT_ROOT, LAST_CHECKPOINT, SAVE_INTERVAL, RESULTS_INTERVAL, EVAL_INTERVAL)

ray.shutdown()
