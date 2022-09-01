from importlib.metadata import entry_points


try:
    from gym.envs.registration import register
    from modules.Environment import CustomOffWorldDiscreteEnv
    from ray.tune.registry import register_env
    # Real environments

    # OffWorld Monolith Real with Discrete actions
    print("Registering Custom Environment")
    register_env('CustomOffWorldDiscreteEnv-v0', CustomOffWorldDiscreteEnv)
    register(id='CustomOffWorldDiscreteEnv-v0', entry_point='modules.Environment:CustomOffWorldDiscreteEnv')
    
except ImportError:
    print("The 'gym' module isn't installed so not registering envs")
