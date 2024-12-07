from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
dataset = LeRobotDataset(
    repo_id="franka/cube_pick_demo",
    local_files_only=True  # Since we're loading from local cache
)

# Basic information about the dataset
print(f"Total episodes: {dataset.num_episodes}")
print(f"Total frames: {dataset.num_frames}")
print(f"Features available: {list(dataset.features.keys())}")
print(f"Recording FPS: {dataset.fps}")

# Load first episode
first_episode = dataset[0]  # Gets first frame
print("\nFirst frame data:")
for key, value in first_episode.items():
    print(f"{key}: shape {value.shape}")

# Plot trajectory of a single episode
episode_indices = dataset.episode_data_index
start_idx = episode_indices['from'][0].item()
end_idx = episode_indices['to'][0].item()

# Get positions for the first episode
tcp_positions = []
block_positions = []

for i in range(start_idx, end_idx):
    frame = dataset[i]
    tcp_positions.append(frame['state.tcp_pos'].numpy())
    block_positions.append(frame['state.block_pos'].numpy())

tcp_positions = np.array(tcp_positions)
block_positions = np.array(block_positions)

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot TCP trajectory
ax.plot(tcp_positions[:, 0], tcp_positions[:, 1], tcp_positions[:, 2], 
        label='TCP Trajectory', marker='.')

# Plot block position
ax.scatter(block_positions[0, 0], block_positions[0, 1], block_positions[0, 2], 
          color='red', marker='o', s=100, label='Block Initial Position')
ax.scatter(block_positions[-1, 0], block_positions[-1, 1], block_positions[-1, 2], 
          color='green', marker='o', s=100, label='Block Final Position')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Robot TCP and Block Trajectory (First Episode)')
plt.show()

# Print success rate
successes = []
for i in range(dataset.num_frames):
    frame = dataset[i]
    successes.append(frame['next.success'].item())

success_rate = sum(successes) / len(successes)
print(f"\nSuccess rate: {success_rate:.2%}")

# Get episode information
print("\nEpisode information:")
print(dataset.meta.episodes)

# Show task description
print("\nTask description:")
print(dataset.meta.tasks)

# Print some statistics from the dataset
print("\nDataset statistics:")
print(dataset.meta.stats)