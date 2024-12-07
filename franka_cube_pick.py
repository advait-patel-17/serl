import time

import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs

env = envs.PandaPickCubeGymEnv(action_scale=(0.1, 1))
action_spec = env.action_space

m = env.model
d = env.data

reset = False
KEY_SPACE = 32
def key_callback(keycode):
    if keycode == KEY_SPACE:
        global reset
        reset = True

obs, _ = env.reset()

# go to cube
def travel_above_cube(obs):
    tcp_pos = obs["state"]["panda/tcp_pos"]
    block_pos = obs["state"]["block_pos"]

    block_pos[2] += 0.05 # target 4 cm above the block
    diff = block_pos - tcp_pos
    diff_normalized = diff / np.linalg.norm(diff)  # Unit vector in direction of block
    action = np.zeros(4)
    action[:3] = diff * 0.5
    action[3] = 0
    flag = np.linalg.norm(diff) <= 0.01
    return action, flag

def travel_down_to_cube(obs):
    tcp_pos = obs["state"]["panda/tcp_pos"]
    block_pos = obs["state"]["block_pos"]

    diff = block_pos - tcp_pos
    diff_normalized = diff / np.linalg.norm(diff)  # Unit vector in direction of block
    action = np.zeros(4)
    action[:3] = diff * 0.1
    action[3] = 0
    flag = np.linalg.norm(diff) <= 0.005
    return action, flag

def grab_cube(obs):
    tcp_pos = obs["state"]["panda/tcp_pos"]
    block_pos = obs["state"]["block_pos"]

    diff = block_pos - tcp_pos
    diff_normalized = diff / np.linalg.norm(diff)  # Unit vector in direction of block
    action = np.zeros(4)
    action[:3] = diff * 0.1
    action[3] = 0.2
    gripper_pos = obs["state"]["panda/gripper_pos"]

    flag = gripper_pos > 0.4
    return action, flag

def to_target(obs, target):
    tcp_pos = obs["state"]["panda/tcp_pos"]

    diff = target - tcp_pos
    action = np.zeros(4)
    action[:3] = diff * 0.1
    action[3] = 1.0

    flag = np.linalg.norm(diff) <= 0.005
    return action, flag


z_init = d.sensor("block_pos").data[2]
target = obs["state"]["block_pos"]
z_success = z_init + 0.2
target[2] = z_success

state = 0
# states: above, down, grabbing, up
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    start = time.time()
    while viewer.is_running():
        if reset:
            obs, _ = env.reset()
            reset = False
            state = 0
            z_init = d.sensor("block_pos").data[2]
            target = obs["state"]["block_pos"]
            z_success = z_init + 0.2
            target[2] = z_success
        else:
            step_start = time.time()
            if state == 0:
                action, flag = travel_above_cube(obs)
            elif state == 1:
                action, flag = travel_down_to_cube(obs)
            elif state == 2:
                action, flag = grab_cube(obs)
            elif state == 3:
                start = time.time()
                action, flag = to_target(obs, target)
            obs, _, _, _, _ = env.step(action)
            if flag:
                state += 1
                print("curr state:", state)
                flag = False
                if state == 4:
                    print("FINISHED!")
            
            if state != 4:
                env.step(action)
                viewer.sync()
                time_until_next_step = env.control_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
