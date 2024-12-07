import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from franka_sim import envs

# Constants and setup
FPS = 30  # Recording frequency
REPO_ID = "franka/cube_pick_demo"
TASK_DESC = "Pick up a cube from a table using a Franka robot arm"

def create_dataset(env):
    """Initialize LeRobot dataset with appropriate features"""
    features = {
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),  # 3 for position, 1 for gripper
            "names": None,
        },
        "state.tcp_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": None,
        },
        "state.gripper_pos": {
            "dtype": "float32", 
            "shape": (1,),
            "names": None,
        },
        "state.block_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": None,
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "seed": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
    }
    
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        use_videos=False  # Not recording videos for now
    )
    
    return dataset

def travel_above_cube(obs):
    tcp_pos = obs["state"]["panda/tcp_pos"]
    block_pos = obs["state"]["block_pos"]

    block_pos[2] += 0.05  # target 4 cm above the block
    diff = block_pos - tcp_pos
    diff_normalized = diff / np.linalg.norm(diff)
    action = np.zeros(4)
    action[:3] = diff * 0.5
    action[3] = 0
    flag = np.linalg.norm(diff) <= 0.01
    return action, flag

def travel_down_to_cube(obs):
    tcp_pos = obs["state"]["panda/tcp_pos"]
    block_pos = obs["state"]["block_pos"]

    diff = block_pos - tcp_pos
    diff_normalized = diff / np.linalg.norm(diff)
    action = np.zeros(4)
    action[:3] = diff * 0.1
    action[3] = 0
    flag = np.linalg.norm(diff) <= 0.005
    return action, flag

def grab_cube(obs):
    tcp_pos = obs["state"]["panda/tcp_pos"]
    block_pos = obs["state"]["block_pos"]

    diff = block_pos - tcp_pos
    diff_normalized = diff / np.linalg.norm(diff)
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

def record_episodes(num_episodes=10):
    env = envs.PandaPickCubeGymEnv(action_scale=(0.1, 1))
    dataset = create_dataset(env)
    
    # Define keyboard callback
    reset_episode = False
    stop_recording = False
    def key_callback(keycode):
        nonlocal reset_episode, stop_recording
        if keycode == 32:  # Space key
            reset_episode = True
            print("\nResetting current episode...")
        elif keycode == ord('s'):  # 's' key
            stop_recording = True
            print("\nStopping recording...")
    
    episode = 0
    while episode < num_episodes:
        print(f"Recording episode {episode}")
        
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            # Episode initialization - moved inside the viewer context
            while True:  # This allows for retries of the same episode
                # Reset environment and initialize episode
                obs, _ = env.reset()
                seed = np.random.randint(0, 1e5)
                
                # Initialize episode variables
                z_init = obs["state"]["block_pos"][2]
                target = obs["state"]["block_pos"].copy()
                z_success = z_init + 0.2
                target[2] = z_success
                
                state = 0  # states: above(0), down(1), grabbing(2), up(3)
                episode_start = time.time()
                reset_episode = False  # Reset the flag
                
                while state < 4 and not reset_episode and not stop_recording:
                    step_start = time.time()
                    
                    # Get action based on current state
                    if state == 0:
                        action, flag = travel_above_cube(obs)
                    elif state == 1:
                        action, flag = travel_down_to_cube(obs)
                    elif state == 2:
                        action, flag = grab_cube(obs)
                    elif state == 3:
                        action, flag = to_target(obs, target)
                    
                    # Execute action and get new observation
                    next_obs, reward, terminated, _, info = env.step(action)
                    
                    # Record frame data
                    timestamp = time.time() - episode_start
                    frame = {
                        "timestamp": timestamp,
                        "action": torch.from_numpy(action),
                        "state.tcp_pos": torch.from_numpy(obs["state"]["panda/tcp_pos"]),
                        "state.gripper_pos": torch.from_numpy(np.array([obs["state"]["panda/gripper_pos"]])),
                        "state.block_pos": torch.from_numpy(obs["state"]["block_pos"]),
                        "next.reward": reward,
                        "next.success": state == 3 and flag,  # Success when final state is reached
                        "seed": seed
                    }
                    
                    dataset.add_frame(frame)
                    
                    # Update state if current subtask is complete
                    if flag:
                        state += 1
                        print(f"Completed state {state-1}")
                    
                    obs = next_obs
                    
                    # Maintain consistent timing
                    viewer.sync()
                    time_until_next = 1/FPS - (time.time() - step_start)
                    if time_until_next > 0:
                        time.sleep(time_until_next)
                
                # Handle episode completion or interruption
                if stop_recording:
                    # Clear the current episode's data without saving
                    dataset.clear_episode_buffer()
                    break
                
                if state >= 4:  # Episode completed successfully
                    dataset.save_episode(task=TASK_DESC)
                    episode += 1
                    print(f"Episode {episode} saved successfully")
                    break  # Exit the retry loop and move to next episode
                
                if reset_episode:
                    # Clear the current episode's data and retry
                    dataset.clear_episode_buffer()
                    print("Episode reset, retrying...")
                    continue  # Restart the same episode
            
            if stop_recording:
                break  # Exit the main recording loop
    
    print(f"\nRecording completed with {episode} episodes")
    # Consolidate and save dataset
    dataset.consolidate()
    dataset.push_to_hub(tags=["franka", "manipulation", "pick-and-place"])
    
    return dataset

if __name__ == "__main__":
    record_episodes()