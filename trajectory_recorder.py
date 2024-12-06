import numpy as np
import json
from datetime import datetime
import mujoco

class TrajectoryRecorder:
    def __init__(self, data):
        """
        Initialize the trajectory recorder.
        
        Args:
            sim: MuJoCo simulation object
        """
        self.sim_data = data
        self.trajectory = []
        self.recording = False
        self.start_time = None
        
    def start_recording(self):
        """Start recording the trajectory"""
        self.recording = True
        self.start_time = self.sim_data.time
        self.trajectory = []
        
    def stop_recording(self):
        """Stop recording the trajectory"""
        self.recording = False
        
    def record_frame(self):
        """Record the current frame if recording is active"""
        if not self.recording:
            return
            
        frame = {
            'time': self.sim_data.time - self.start_time,
            'qpos': self.sim_data.qpos.copy().tolist(),
            'qvel': self.sim_data.qvel.copy().tolist()
        }
        self.trajectory.append(frame)
        
    def save_trajectory(self, filename=None):
        """
        Save the recorded trajectory to a file.
        
        Args:
            filename: Optional filename, if None generates timestamp-based name
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'trajectory_{timestamp}.json'
            
        trajectory_data = {
            'frames': self.trajectory,
            'metadata': {
                'num_frames': len(self.trajectory),
                'total_time': self.trajectory[-1]['time'] if self.trajectory else 0,
                'recorded_at': datetime.now().isoformat()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        return filename


class TrajectoryPlayer:
    def __init__(self, data):
        """
        Initialize the trajectory player.
        
        Args:
            sim: MuJoCo simulation object
        """
        self.sim_data = data
        self.trajectory = None
        self.current_frame = 0
        self.playing = False
        self.start_time = None
        
    def load_trajectory(self, filename):
        """
        Load a trajectory from a file.
        
        Args:
            filename: Path to the trajectory file
        """
        with open(filename, 'r') as f:
            trajectory_data = json.load(f)
            
        self.trajectory = trajectory_data['frames']
        return trajectory_data['metadata']
        
    def start_playback(self):
        """Start playing the loaded trajectory"""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")
            
        self.playing = True
        self.current_frame = 0
        self.start_time = self.sim_data.time
        
    def stop_playback(self):
        """Stop playing the trajectory"""
        self.playing = False
        
    def update(self):
        """
        Update the simulation state based on the trajectory.
        Returns True if playback is complete.
        """
        if not self.playing or self.trajectory is None:
            return True
            
        current_time = self.sim_data.time - self.start_time
        
        # Find the appropriate frame for the current time
        while (self.current_frame < len(self.trajectory) and 
               self.trajectory[self.current_frame]['time'] <= current_time):
            frame = self.trajectory[self.current_frame]
            self.sim_data.qpos[:] = frame['qpos']
            self.sim_data.qvel[:] = frame['qvel']
            self.current_frame += 1
            
        # Return True if playback is complete
        return self.current_frame >= len(self.trajectory)


# Example usage
def example_usage():
    # Initialize MuJoCo simulation (replace with your model)
    model = mujoco.MjModel.from_xml_path("your_robot.xml")
    data = mujoco.MjData(model)
    
    # Recording example
    recorder = TrajectoryRecorder(data)
    recorder.start_recording()
    
    # Simulation loop for recording
    for _ in range(1000):
        # Your simulation step code here
        mujoco.mj_step(model, data)
        recorder.record_frame()
    
    recorder.stop_recording()
    filename = recorder.save_trajectory()
    print(f"Saved trajectory to {filename}")
    
    # Playback example
    player = TrajectoryPlayer(data)
    metadata = player.load_trajectory(filename)
    print(f"Loaded trajectory with {metadata['num_frames']} frames")
    
    player.start_playback()
    
    # Simulation loop for playback
    while not player.update():
        # You can add visualization code here
        mujoco.mj_step(model, data)

if __name__ == "__main__":
    example_usage()