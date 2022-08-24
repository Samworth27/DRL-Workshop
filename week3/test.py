#!/usr/bin/env python
# coding: utf-8

# In[20]:


import ray
import json
import ray.rllib.algorithms.apex_dqn as apex_dqn
from timeit import default_timer as timer


# In[21]:


ray.init(ignore_reinit_error=True)


# In[32]:


ENV = "BreakoutNoFrameskip-v4"
VERSION = "apex_dqn_v1"
TRAINER = apex_dqn

checkpoint_root = f'.checkpoints/{ENV}/{str(TRAINER)}/{VERSION}'

config=TRAINER.APEX_DEFAULT_CONFIG.copy()


# In[29]:


config=apex_dqn.APEX_DEFAULT_CONFIG.copy()
config['env'] = ENV
config["num_gpus"]=1
config["num_workers"]=8
config["num_cpus_per_worker"]=0.25
config["num_envs_per_worker"]=1
config['target_network_update_freq'] = 8000
config['replay_buffer_config']['capacity']= 10000

print(config['num_workers'])


# In[31]:


algorithm = apex_dqn.ApexDQN(config=config)


# In[16]:


algorithm._logdir = checkpoint_root
algorithm.logdir


# In[12]:


result = algorithm.train()
print(pretty_print(result))


# In[55]:


checkpoint = algorithm.save(checkpoint_root)
checkpoint
with open(f'{checkpoint}/result.json', 'w') as fp:
    # fp.write(str(result))
    json.dump(str(result), fp,  indent=4)
    
# def cleanDict(dict):  
#   for key, value in zip(dict.keys(),dict.values()):
#     match type(value):
#       case str:
#         pass
    
json.dumps(result)

