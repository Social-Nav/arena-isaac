"""RosBag-based data logger for VLN dataset recording (lightweight)."""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class VLNDataLoggerRosbag:
    """Lightweight rosbag command-line based data recorder.
    
    Usage:
        logger = VLNDataLoggerRosbag(
            topics=["/camera/image_raw", "/lidar/points", "/tf"],
            output_dir="collected_data"
        )
        logger.start_recording()
        # ... run simulation ...
        logger.stop_recording()
    """
    
    def __init__(self, 
                 topics: List[str],
                 output_dir: str = "collected_data",
                 initial_delay: float = 1.5):
        """Initialize rosbag recorder.
        
        Args:
            topics: List of topics to record
            output_dir: Output directory
            initial_delay: Delay before recording (seconds)
        """
        self.topics = topics
        self.output_dir = output_dir
        self.initial_delay = initial_delay
        self.episode_idx = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.process = None
        self.bag_path = None
        self._recording = False
        
        sys.stderr.write(f"[INIT] VLNDataLoggerRosbag ready to record {len(topics)} topics\n")

    def start_recording(self):
        """Start background rosbag recording."""
        if self._recording:
            sys.stderr.write("[WARN] Already recording, skipping.\n")
            return
        
        bag_name = f"episode_{self.episode_idx:06d}"
        self.bag_path = os.path.join(self.output_dir, bag_name)
        
        cmd = [
            "ros2", "bag", "record",
            "-s", "mcap",
            "-o", self.bag_path,
        ] + self.topics
        
        sys.stderr.write(f"[START] Recording: {' '.join(cmd)}\n")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            
            time.sleep(self.initial_delay)
            self._recording = True
            sys.stderr.write(f"[OK] Recording started\n")
            
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to start rosbag: {e}\n")
            self.process = None

    def stop_recording(self) -> str:
        """Wait for rosbag process to exit.
        
        Returns:
            Path to saved bag file
        """
        if not self.process:
            sys.stderr.write("[WARN] Rosbag process not started\n")
            return None
        
        sys.stderr.write("[STOP] Waiting for rosbag to exit...\n")
        
        try:
            try:
                self.process.wait(timeout=10.0)
                sys.stderr.write(f"[OK] Rosbag saved to: {self.bag_path}\n")
            except subprocess.TimeoutExpired:
                sys.stderr.write("[WARN] Rosbag timeout, forcing kill\n")
                self.process.kill()
                self.process.wait()
            
            self._recording = False
            return self.bag_path
            
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to wait for process: {e}\n")
            return None

    def next_episode(self):
        """Finish current episode and prepare for next."""
        self.stop_recording()
        self.episode_idx += 1
        self.start_recording()
        sys.stderr.write(f"[NEXT] Starting episode: {self.episode_idx}\n")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        if self.process:
            return self.process.poll() is None
        return False


class VLNDataBufferRosbag:
    """Async data buffer for post-processing MCAP files."""
    
    def __init__(self, mcap_file_path: str, topic_filters: List[str] = None):
        """Initialize buffer.
        
        Args:
            mcap_file_path: Path to MCAP file
            topic_filters: Topics to load (None = load all)
        """
        self.mcap_path = mcap_file_path
        self.topic_filters = topic_filters or []
        self.messages = defaultdict(list)
        self._load_mcap()

    def _load_mcap(self):
        """Load messages from MCAP file."""
        try:
            from mcap.reader import Reader
            
            with open(self.mcap_path, "rb") as f:
                reader = Reader(f)
                
                for schema, channel, message in reader.iter_messages():
                    if self.topic_filters and channel.topic not in self.topic_filters:
                        continue
                    
                    self.messages[channel.topic].append({
                        'timestamp': message.log_time,
                        'data': message.data
                    })
            
            print(f"[OK] Loaded MCAP: {self.mcap_path}")
            for topic, msgs in self.messages.items():
                print(f"     {topic}: {len(msgs)} messages")
                
        except Exception as e:
            print(f"[ERROR] Failed to load MCAP: {e}")

    def get_messages_by_topic(self, topic: str) -> List[Dict]:
        """Get all messages for a topic."""
        return self.messages.get(topic, [])

    def synchronize_by_timestamp(self, time_tolerance_ns: int = 10_000_000) -> List[Dict]:
        """Synchronize messages across topics by timestamp.
        
        Args:
            time_tolerance_ns: Time tolerance in nanoseconds (default 10ms)
            
        Returns:
            Synchronized message list
        """
        all_timestamps = set()
        for msgs in self.messages.values():
            for msg in msgs:
                all_timestamps.add(msg['timestamp'])
        
        all_timestamps = sorted(all_timestamps)
        synchronized = []
        
        for ts in all_timestamps:
            frame = {'timestamp': ts, 'topics': {}}
            
            for topic, msgs in self.messages.items():
                closest = min(
                    msgs,
                    key=lambda m: abs(m['timestamp'] - ts),
                    default=None
                )
                
                if closest and abs(closest['timestamp'] - ts) <= time_tolerance_ns:
                    frame['topics'][topic] = closest
            
            if frame['topics']:
                synchronized.append(frame)
        
        return synchronized


def run_logger(topics: List[str], 
               duration_seconds: Optional[float] = None,
               output_dir: str = "collected_data"):
    """Quick start a rosbag recording.
    
    Args:
        topics: Topics to record
        duration_seconds: Recording duration (None = manual stop)
        output_dir: Output directory
    """
    logger = VLNDataLoggerRosbag(
        topics=topics,
        output_dir=output_dir
    )
    
    logger.start_recording()
    
    if duration_seconds:
        time.sleep(duration_seconds)
        logger.stop_recording()
    else:
        try:
            while logger.is_recording:
                time.sleep(1)
        except KeyboardInterrupt:
            sys.stderr.write("\n[INTERRUPT] User interrupt, stopping recording...\n")
            logger.stop_recording()


if __name__ == "__main__":
    # Example: record LIDAR + TF topics
    topics = [
        "/task_generator_node/jackal/lidar/points",
        "/tf",
        "/tf_static",
    ]
    
    run_logger(topics, duration_seconds=10)
