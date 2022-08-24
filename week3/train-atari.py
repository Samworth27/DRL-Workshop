import os
import ray
import gym
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from timeit import default_timer as timer

ray.shutdown()
ray.init(ignore_reinit_error=True)

ENV = "BreakoutNoFrameskip-v4"
TARGET_REWARD = 20
TRAINER = DQNTrainer
# config docs:
# https://docs.ray.io/en/latest/rllib/rllib-training.html
# https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn
config = TRAINER.merge_trainer_configs( DEFAULT_CONFIG, {
      'env': ENV,
      # 'record_env': True,
      'framework': 'tf2',
      'num_workers': 20,
      'num_cpus_per_worker': 0.5,
      'num_gpus': 1,
      'num_gpus_per_worker': 0,
      
      'double_q': False,
      'dueling': False,
      'num_atoms': 1,
      'noisy': False,
      'replay_buffer_config':{
        'type': 'MultiAgentReplayBuffer',
        'capacity': 10000,
      },
      # 'num_steps_sampled_before_learning_starts': 20000,
      'n_step': 1,
      'target_network_update_freq': 8000,
      'lr': .0000625,
      'adam_epsilon': .00015,
      'hiddens': [512],
      'rollout_fragment_length': 4,
      'train_batch_size': 32,
      'exploration_config': {
        'epsilon_timesteps': 200000,
        'final_epsilon': 0.01,
      },
      'env_config': {
        # 'render_mode': 'human',
      },
      'min_sample_timesteps_per_iteration': 10000,
})
last_checkpoint = 173
evaluate = True
experiment_name = 'v1'
chkpt_root = f'.checkpoints/{ENV}/{experiment_name}'
evaluate = False
epochs = 90
# if evaluate:
#   config['env_config'] = config['evaluation_config']['env_config']
#   config['explore'] = False
agent = TRAINER(config=config)

# env = gym.make(ENV, render_mode='human')
# env.reset()

if last_checkpoint != None:
  chkpt_file = f'{chkpt_root}/checkpoint_{last_checkpoint:06}/checkpoint-{last_checkpoint}'
  agent.restore(chkpt_file)
  checkpoint_dir = os.path.dirname(chkpt_file)
else:
  chkpt_file = agent.save(chkpt_root)
  print('*** initial checkpoint', chkpt_file)
  print('config', config)
  checkpoint_dir = os.path.dirname(chkpt_file)
  with open(f'{checkpoint_dir}/config.json', 'w') as fp:
    fp.write(str(config))
if evaluate:
  print('evaluation')
  result = agent.evaluate()
  print('evaluation result', result)
  with open(f'{checkpoint_dir}/evaluation.json', 'w') as fp:
    fp.write(str(result))
else:
  for epoch in range(epochs):
      print(f'--- epoch [{epoch}]')
      start = timer()
      result = agent.train()
      end = timer()
      chkpt_file = agent.save(chkpt_root)
      print(f'*** end of [{epoch}] epoch. Time to complete: {end-start}')
      print(chkpt_file,
              result["episode_reward_min"],
              result["episode_reward_mean"],
              result["episode_reward_max"],
              result["episode_len_mean"],
              )
      # print(result)
      checkpoint_dir = os.path.dirname(chkpt_file)
      with open(f'{checkpoint_dir}/result.json', 'w') as fp:
        fp.write(str(result))
        # json.dump(result, fp,  indent=4)
