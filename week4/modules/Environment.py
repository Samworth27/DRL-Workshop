import logging
from scipy.spatial import distance
import numpy as np
from gym import spaces
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.gazebo.image_utils import ImageUtils
from offworld_gym.envs.gazebo.offworld_monolith_env import OffWorldDockerizedMonolithEnv
from ray.rllib.env.env_context import EnvContext

class CustomOffWorldDiscreteEnv(OffWorldDockerizedMonolithEnv):
    """Discrete version of the simulated gym environment that replicates the real OffWor
    .. code:: python

        env = gym.make('OffWorldMonolithDiscreteSim-v0', channel_type=Channels.DEPTHONLY, random_init=True)
        env = gym.make('OffWorldMonolithDiscreteSim-v0', channel_type=Channels.RGB_ONLY, random_init=True)
        env = gym.make('OffWorldMonolithDiscreteSim-v0', channel_type=Channels.RGBD, random_init=True)
    """

    def __init__(self, config):
        print(config)
        channel_type = config['channel_type'] if 'channel_type' in config else Channels.DEPTH_ONLY
        random_init = config['random_init'] if 'random_init' in config else True
        super(CustomOffWorldDiscreteEnv, self).__init__(
            channel_type=channel_type, random_init=random_init)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 255, shape=(
            ImageUtils.IMG_H, ImageUtils.IMG_W, channel_type.value))
        self.logger = logging.getLogger(__name__)
        level = logging.INFO
        self.logger.setLevel(level)

    def _send_action_commands(self, action_type):
        """Sends an action command to the robot.

        Args:
            action_type: FourDiscreteMotionActions instance.
        Returns:
            The real time factor for the move (sim-time elapsed/wall-time elapsed)
        """
        if action_type == FourDiscreteMotionActions.LEFT:
            return self._move_rosbot(0.07, 1.25, 4 * 0.3 * self._STEP_DURATION_SECONDS_IN_SIM)
        elif action_type == FourDiscreteMotionActions.RIGHT:
            return self._move_rosbot(0.07, -1.25, 4 * 0.3 * self._STEP_DURATION_SECONDS_IN_SIM)
        elif action_type == FourDiscreteMotionActions.FORWARD:
            return self._move_rosbot(0.1, 0.0, 2 * 0.3 * self._STEP_DURATION_SECONDS_IN_SIM)
        elif action_type == FourDiscreteMotionActions.BACKWARD:
            return self._move_rosbot(-0.1, 0.0, 2 * 0.3 * self._STEP_DURATION_SECONDS_IN_SIM)

    def step(self, action):
        """Take an action in the environment.

        Args:
            action: An action to be taken in the environment.

        Returns:
            A numpy array with rgb/depth/rgbd encoding of the state observation.
            An integer with reward from the environment.
            A boolean flag which is true when an episode is complete.
            Info containing the ratio of simulation-time / wall-time taken by the step
        """
        self.step_count += 1

        assert action is not None, "Action cannot be None."
        # convert float if it's exactly an integer value, otherwise let it throw an error
        if isinstance(action, (float, np.float32, np.float64)) and float(action).is_integer():
            action = int(action)
        assert isinstance(action, (FourDiscreteMotionActions, int,
                          np.int32, np.int64)), "Action type is not recognized."

        if isinstance(action, (int, np.int32, np.int64)):
            assert action >= 0 and action < 4, "Unrecognized value for the action"
            action = FourDiscreteMotionActions(action)

        self.logger.info("Step: %d" % self.step_count)
        self.logger.info(action)
        real_time_factor_for_move = self._send_action_commands(action)

        self._current_state = self._get_state()
        
        info = {"real_time_factor_for_move": real_time_factor_for_move}
        reward, done = self._calculate_reward(action)

        rosbot_state = self._get_state_vector('rosbot')
        
        dst = distance.euclidean(rosbot_state[0:3], self._monolith_space[0:3])
        
        info['distance'] = self.getDistance()
        
        if done:
            self.step_count = 0
            
    

        return self._current_state, reward, done, info
    
    def getDistance(self):
        rosbot_state = self._get_state_vector('rosbot')
        return distance.euclidean(rosbot_state[0:3], self._monolith_space[0:3])