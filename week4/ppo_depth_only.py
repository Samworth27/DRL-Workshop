import ray.rllib.algorithms.ppo as ppo

from modules.Train import runTraining
import ray

from offworld_gym.envs.common.channels import Channels

print("Starting Ray instance")
ray.init(
    ignore_reinit_error=True,
    # _temp_dir = "./ray_results"
    #  logging_level=logging.FATAL, log_to_driver=False
)

ENV = "CustomOffWorldDockerMonolithDiscreteSim-v0"
VERSION = "depth_only_v2"
TRAINER = ppo
DEFAULT_CONFIG = TRAINER.DEFAULT_CONFIG.copy()
NUMBER_OF_EPOCHS = 100

LAST_CHECKPOINT = 38
RESULTS_INTERVAL = 1
SAVE_INTERVAL = 2
EVAL_INTERVAL = None

checkpoint_root = f'.checkpoints/{ENV}/{VERSION}'

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
    'num_workers': 4,
    'num_envs_per_worker': 1,
    'batch_mode': 'truncate_episodes',
    'observation_filter': 'NoFilter',
    'model': {
        'vf_share_layers': 'true'
    },
    # 'hiddens': [256,256],
    'num_gpus': 1,
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
    'disable_env_checking': True,
})


print(f'Checkpoints Stored at {checkpoint_root}')

print("Starting Trainer")
algorithm = TRAINER.PPO(config=config)
print("Trainer Started")

runTraining(algorithm, NUMBER_OF_EPOCHS,checkpoint_root, LAST_CHECKPOINT, SAVE_INTERVAL, RESULTS_INTERVAL, EVAL_INTERVAL)

ray.shutdown()
