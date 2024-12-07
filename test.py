import time
import numpy as np
import mujoco
import mujoco.viewer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from franka_sim import envs

def visualize_trajectories(episode_indices=None, playback_speed=1.0):
    """
    Visualize recorded trajectories in the MuJoCo environment.
    
    Args:
        episode_indices (list, optional): List of episode indices to visualize. 
                                        If None, visualizes all episodes.
        playback_speed (float): Speed multiplier for playback. Default is 1.0 (real-time).
    """
    # Initialize environment
    env = envs.PandaPickCubeGymEnv(action_scale=(0.1, 1))
    
    # Load dataset from hub
    dataset = LeRobotDataset(repo_id="franka/cube_pick_demo", local_files_only=True)
    
    # Get episode indices to visualize
    if episode_indices is None:
        episode_indices = range(len(dataset.episodes))
    
    # Define keyboard callback for control
    skip_current = False
    stop_visualization = False
    def key_callback(keycode):
        nonlocal skip_current, stop_visualization
        if keycode == 32:  # Space key
            skip_current = True
            print("\nSkipping to next episode...")
        elif keycode == ord('q'):  # 'q' key
            stop_visualization = True
            print("\nStopping visualization...")
    
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        for episode_idx in episode_indices:
            if stop_visualization:
                break
                
            print(f"\nVisualizing episode {episode_idx}")
            episode_data = dataset.episodes[episode_idx]
            
            # Reset environment with the same seed
            seed = episode_data["seed"][0].item()
            env.reset(seed=seed)
            
            skip_current = False
            prev_timestamp = episode_data["timestamp"][0].item()
            
            for frame_idx in range(len(episode_data["timestamp"])):
                if skip_current or stop_visualization:
                    break
                
                # Get action from recorded data
                action = episode_data["action"][frame_idx].numpy()
                
                # Execute action
                obs, reward, terminated, _, info = env.step(action)
                
                # Calculate sleep time based on recorded timestamps and playback speed
                current_timestamp = episode_data["timestamp"][frame_idx].item()
                sleep_time = (current_timestamp - prev_timestamp) / playback_speed
                
                # Update viewer and maintain timing
                viewer.sync()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                prev_timestamp = current_timestamp
                
                # Optional: Print progress
                if frame_idx % 30 == 0:  # Print every second (assuming 30 FPS)
                    success = episode_data["next.success"][frame_idx].item()
                    print(f"Frame {frame_idx}, Success: {success}", end="\r")
            
            print(f"\nCompleted episode {episode_idx}")
    
    print("\nVisualization completed")

if __name__ == "__main__":
    # Example usage
    visualize_trajectories(
        episode_indices=[0, 1, 2],  # Visualize first three episodes
        playback_speed=1.0  # Real-time playback
    )