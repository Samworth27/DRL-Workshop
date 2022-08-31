from sshkeyboard import listen_keyboard

from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import gym
import numpy as np
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.real.real_env import AlgorithmMode, LearningType

# to surpress the warning when running in real env
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


print(
    """
    This example allows you to manually control the OffWorld Gym robot.
    Use the arrow keys ← ↑ → ↓ to issue the commands and [Esc] to exit
    You can monitor the robot via the overhead cameras at https://gym.offworld.ai/cameras
    """)

key_actions = {'up': 2, 'down': 3, 'left': 0, 'right': 1}
key_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

# create the envronment and establish connection
env = gym.make("OffWorldDockerMonolithDiscreteSim-v0",
               channel_type=Channels.RGBD)

state = env.reset()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140],)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(np.array(state[:, :, :3], dtype='int'))
ax2.imshow(np.array(state[:, :, 3]), cmap='gray')
plt.ion()
plt.show()

for i in range(100):

    sample_action = env.action_space.sample()
    print(sample_action)
    state, reward, done, _ = env.step(sample_action)
    rgb = np.array(state[:, :, :3], dtype='int')
    greyscale = rgb2gray(rgb).astype('int')
    depth = np.array(state[:, :, 3])
    combined = np.dstack([(minmax_scale(depth)*255).astype('int'), greyscale,greyscale]).astype('int')
    ax1.imshow(rgb)
    ax2.imshow(depth, cmap='gray')
    ax3.imshow(greyscale, cmap='gray')
    ax4.imshow(combined)
    plt.draw()
    plt.pause(0.001)

    if done:
        state = env.reset()


# listen_keyboard(on_press=press, sequential=True)
