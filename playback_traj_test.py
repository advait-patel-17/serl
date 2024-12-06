from franka_sim import envs
from trajectory_recorder import TrajectoryPlayer
import mujoco
env = envs.PandaPickCubeGymEnv(action_scale=(0.1, 1))
action_spec = env.action_space

m = env.model
d = env.data

import time

filename = "trajectory_20241124_235016.json"
player = TrajectoryPlayer(d)
player.load_trajectory(filename)
player.start_playback()

with mujoco.viewer.launch_passive(m, d) as viewer:
    while not player.update():
        step_start = time.time()
        mujoco.mj_step(m, d)
        viewer.sync()
        time_until_next_step = env.control_dt - (time.time() - step_start)
        # TODO figure out why this makes it slower than when recording. time is saved for each frame in 
        # trajectory so probably just use that
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

