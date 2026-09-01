import omni.replicator.core as rep
import numpy as np
# === ROBUST AUTO-FIX FOR PANDAS ===
# import sys, subprocess, pkgutil, os, site

# def ensure_pandas_robust():
#     # 1. If it can already be imported, return directly
#     if pkgutil.find_loader('pandas') is not None:
#         return True
    
#     print("[AUTO-FIX] pandas not found, attempting robust install...", file=sys.stderr)
    
#     try:
#         # 2. Get the current interpreter's site-packages path
#         result = subprocess.run(
#             [sys.executable, "-c", "import site; print(site.getsitepackages()[0] if site.getsitepackages() else site.getusersitepackages())"],
#             capture_output=True, text=True, timeout=30
#         )
#         target_path = result.stdout.strip()
#         print(f"[AUTO-FIX] Target install path: {target_path}", file=sys.stderr)
        
#         # 3. Install to the target path (use --target to ensure correct location)
#         install_result = subprocess.run(
#             [sys.executable, "-m", "pip", "install", "--target", target_path, "--quiet", "pandas"],
#             capture_output=True, text=True, timeout=180
#         )
        
#         if install_result.returncode != 0:
#             print(f"[AUTO-FIX] ❌ Install failed: {install_result.stderr[:300]}", file=sys.stderr)
#             return False
        
#         # 4. [KEY] Add the target path to sys.path immediately
#         if target_path and target_path not in sys.path:
#             sys.path.insert(0, target_path)
#             print(f"[AUTO-FIX] Added {target_path} to sys.path", file=sys.stderr)
        
#         # 5. Clear import cache and try re-import
#         if 'pandas' in sys.modules:
#             del sys.modules['pandas']
        
#         # 6. Verify import
#         import pandas as pd
#         print(f"[AUTO-FIX] ✅ SUCCESS: pandas {pd.__version__} loaded from {pd.__file__}", file=sys.stderr)
#         return True
        
#     except Exception as e:
#         print(f"[AUTO-FIX] ❌ Exception: {e}", file=sys.stderr)
#         import traceback
#         traceback.print_exc(file=sys.stderr)
#         return False

# ensure_pandas_robust()
# === END ROBUST AUTO-FIX ===

import os
import re
import sys
import json
import time
import subprocess
import signal
import threading

try:
    from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
except ImportError:
    from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid

try:
    from isaacsim.core.experimental.prims import XformPrim as XFormPrim
except ImportError:
    from omni.isaac.core.prims import XFormPrim

try:
    from isaacsim.core.api.simulation_context import SimulationContext
except ImportError:
    from omni.isaac.core import SimulationContext

from scipy.spatial.transform import Rotation as R
# from mcap_ros2.ros2_decoding import DecoderFactory
# from mcap_ros2.writer import Writer as Ros2Writer

import rclpy
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

try:
    from mcap.writer import Writer
    HAS_MCAP = True
except ImportError:
    HAS_MCAP = False


# ─────────────────────────────────────────────────────────────────────────────────────
# CAPTURE RESOLUTION -- must match the Isaac camera the data is collected from.
#
# Used three times: to size both render products (initial + reset) and as the
# width/height that process_camera_data() turns cameraProjection into pixel
# intrinsics. Those must be the SAME numbers, so keep it a single constant: a
# render product at one size with intrinsics computed at another yields fx/fy/cx/cy
# that are silently wrong for the images actually written.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
# ─────────────────────────────────────────────────────────────────────────────────────

class DataLoggerReplicator:
    """Replicator-based dataset logger using Isaac Sim annotators."""
    
    def __init__(self, camera_prim_path, pedestrian_root_path, lidar_prim_path,
                 output_dir="collected_data", robot_name="jackal",
                 fps=None, world=None):
        """Initialize logger.

        Args:
            camera_prim_path: Camera prim path in USD stage
            pedestrian_root_path: Pedestrian root prim path
            lidar_prim_path: LIDAR prim path
            output_dir: Output directory for saving data (default: collected_data)
            robot_name: Robot model name, used for the MCAP frame_id and topic. Defaults to
                "jackal" only to preserve the old behaviour for callers that do not pass it;
                run_isaacsim always does.
            fps: Capture rate in Hz, in sim time. None (default) captures every rendered
                frame, i.e. the full render rate (60 Hz with World() defaults).

                Capture happens once per world.step(render=True), so the rate is reached
                by keeping 1 frame in N — N integer. Only exact divisors of the render
                rate are therefore achievable: at 60 Hz rendering, 60/30/20/15/12/10 Hz
                are exact, and anything else rounds to the nearest one (25 -> 30, 40 ->
                30). The resolved rate is stored in self.fps and logged at startup; read
                that rather than assuming the requested value was honoured.

                Must match the --fps passed to postprocess/process_raw_to_dataset.py, or
                preview mp4 playback speed will not match sim time.
            world: The --world launch argument (e.g. grscenes_1). When set, output_dir is
                treated as the grscenes root and everything lands in
                    <output_dir>/<world>/{data,meta,videos}/
                with no scenario or timestamp level in between, so re-running the same
                world appends episodes to one dataset instead of forking a new tree.
                When None, falls back to a timestamped session dir under output_dir.
        """
        
        if not is_prim_path_valid(camera_prim_path):
            sys.stderr.write(f"[ERROR] Invalid camera path: {camera_prim_path}\n")
        else:
            sys.stderr.write(f"[OK] Camera path found: {camera_prim_path}\n")

        self.output_dir = output_dir
        self.camera_prim_path = camera_prim_path
        # Derived once so the frame_id and the lidar topic can never disagree with each other.
        self.robot_name = robot_name
        self.base_frame = f"{robot_name}/base_link"

        # Create directory structure keyed on the --world launch argument.
        self.world = world
        if world:
            # <output_dir>/<world>/{data,meta,videos}/ -- output_dir is the grscenes root.
            # No scenario level and no timestamp: one world == one dataset, so a second
            # run of the same world continues numbering into the same three dirs.
            self.scene_root = os.path.join(output_dir, world)
            self.data_dir = os.path.join(self.scene_root, "data")
            self.meta_dir = os.path.join(self.scene_root, "meta")
            self.videos_dir = os.path.join(self.scene_root, "videos")

            old_umask = os.umask(0)
            os.makedirs(self.data_dir, mode=0o777, exist_ok=True)
            os.makedirs(self.meta_dir, mode=0o777, exist_ok=True)
            os.makedirs(self.videos_dir, mode=0o777, exist_ok=True)
            os.umask(old_umask)

            # Episode numbering continues past whatever is already on disk, so a rerun of
            # the same world does not overwrite earlier episodes.
            self.episode_idx = self._next_episode_idx(self.data_dir)

            self.session_dir = self.scene_root
            sys.stderr.write(f"[Logger] World-based structure: {self.scene_root}\n")
            sys.stderr.write(f"  Data   → {self.data_dir}   (episode_XXXXXX.json)\n")
            sys.stderr.write(f"  Videos → {self.videos_dir} (episode_XXXXXX_N.npy)\n")
            sys.stderr.write(f"  Meta   → {self.meta_dir}\n")
            if self.episode_idx:
                sys.stderr.write(
                    f"[Logger] {self.episode_idx} existing episode(s), resuming at "
                    f"episode_{self.episode_idx:06d}.\n")
        else:
            # Legacy structure: timestamped session dir
            import datetime
            beijing_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
            session_name = beijing_time.strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir = os.path.join(output_dir, session_name)
            old_umask = os.umask(0)
            os.makedirs(self.session_dir, mode=0o777, exist_ok=True)
            os.umask(old_umask)
            self.scene_root = self.session_dir
            self.data_dir = self.session_dir  # parquet in episode dirs
            self.videos_dir = self.session_dir  # videos in episode dirs
            self.meta_dir = self.session_dir  # meta in session root
            sys.stderr.write(f"[Logger] Legacy structure: {self.session_dir}\n")
            self.episode_idx = 0

        # NOTE: episode_idx is set by both branches above (the world branch resumes past
        # existing episodes), so it must not be reset here.
        self.param_buffer = []
        self.rgb_frame_buffer = []
        self.depth_frame_buffer = []
        self._stream_chunk_idx = 0
        # Frame index within the current episode, for the flat videos/ naming. Kept
        # separate from _stream_chunk_idx so names stay continuous across chunk flushes.
        self._frame_write_idx = 0
        self._pending_saves = []
        self._skip_frames_after_rebuild = 0  # warmup ticks after a rebuild

        # Requested capture rate -> integer decimation against the render rate.
        # step() runs once per world.step(render=True), so the fastest achievable
        # rate is the render rate itself; fps above that is clamped to it.
        render_hz = self._get_render_hz()
        self.fps = render_hz if fps is None else float(fps)
        self._capture_decimation = max(1, round(render_hz / self.fps))
        self.fps = render_hz / self._capture_decimation  # actual, after rounding
        # Counts every step() call that reaches the gate; the modulo test is against
        # this, so the capture phase stays stable across episodes.
        self._step_call_count = 0
        sys.stderr.write(
            f"[Logger] Capture {self.fps:.1f} Hz "
            f"(every {self._capture_decimation} of {render_hz:.0f} Hz rendering).\n")

        # Create render product for camera
        self.camera_rp = rep.create.render_product(camera_prim_path, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
        
        # Register annotators
        self.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.cam_params_annot = rep.AnnotatorRegistry.get_annotator("camera_params")

        # Attach camera data
        self.rgb_annot.attach(self.camera_rp)
        self.depth_annot.attach(self.camera_rp)
        self.cam_params_annot.attach(self.camera_rp)

        # # Register LIDAR Annotator
        '''self.lidar_annot = None
        self.lidar_rp = None
        try:
            self.lidar_annot = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacComputeRTXLidarPointCloud")
            if is_prim_path_valid(lidar_prim_path):
                self.lidar_rp = rep.create.render_product(lidar_prim_path, (1, 1))
                self.lidar_annot.attach(self.lidar_rp)
                sys.stderr.write(f"✅ LIDAR Annotator initialized: {lidar_prim_path}\n")
            else:
                sys.stderr.write(f"⚠️ LIDAR path does not exist: {lidar_prim_path}, skipping point cloud capture\n")
                self.lidar_annot = None
        except Exception as e:
            sys.stderr.write(f"⚠️ LIDAR Annotator initialization failed: {e}\n")
            self.lidar_annot = None'''

        # Initialize pedestrian list
        self.pedestrian_prims = []
        self.pedestrian_root_path = pedestrian_root_path
        
        # MCAP
        '''
        self._lidar_writer = None
        self._lidar_file = None
        self._lidar_channel_id = None
        self._pc2_schema_id = None
        self._lidar_message_count = 0
        self._init_lidar_writer()'''

        # json
        '''
        output_path = os.path.join(self.output_dir, f"lidar_episode_{self.episode_idx:06d}.mcap")
        self.f = open(output_path, "wb")
        self.writer = Writer(self.f)
        self.writer.start() 

        
        self.schema_id = self.writer.register_schema(
            name="pointcloud_json",
            encoding="jsonschema",
            data=json.dumps({
                "type": "object",
                "properties": {
                    "points": {"type": "array", "items": {"type": "array", "minItems": 3, "maxItems": 3}},
                    "frame_id": {"type": "string"}
                }
            }).encode()
        )

        self.channel_id = self.writer.register_channel(
            topic="/lidar/json_points",
            message_encoding="json",
            schema_id=self.schema_id,
        )'''
        
        # rosbag
        '''self.lidar_topic = "/task_generator_node/jackal/lidar/points"
        self.record_process = None
        self._start_rosbag_record()'''

    def json_write_points(self, points, sim_time, frame_id=None):
        """Write point cloud data to JSON format. frame_id defaults to this robot's base frame."""
        points_list = points.tolist()

        data = {
            "points": points_list,
            "frame_id": frame_id if frame_id is not None else self.base_frame,
            "timestamp": sim_time
        }
        
        ns_time = int(sim_time * 1e9)
        self.writer.add_message(
            channel_id=self.channel_id,
            log_time=ns_time,
            data=json.dumps(data).encode("utf-8"),
            publish_time=ns_time
        )

    def json_close(self):
        """Close MCAP file in JSON mode."""
        self.writer.finish()
        self.f.close()
        sys.stderr.write(f"[OK] MCAP (JSON mode) finalized\n")

    def _init_lidar_writer(self):
        """Initialize MCAP file and setup point cloud channel."""
        if not HAS_MCAP:
            sys.stderr.write("[WARN] mcap library not installed, point cloud save unavailable\n")
            return
        try:
            lidar_output_path = os.path.join(self.output_dir, f"lidar_episode_{self.episode_idx:06d}.mcap")
            os.makedirs(os.path.dirname(lidar_output_path) or ".", exist_ok=True)
            
            self._lidar_file = open(lidar_output_path, "wb")
            self._lidar_ros_writer = Ros2Writer(self._lidar_file)

            self._lidar_channel_id = self._lidar_ros_writer.register_msg_channel(
                    topic=f"/{self.robot_name}/lidar_points",
                    msg_type="sensor_msgs/msg/PointCloud2",
                    frame_id=self.base_frame
                )
            self._lidar_writer = self._lidar_ros_writer
            # self._lidar_writer = Writer(self._lidar_file)
            # self._lidar_writer.start()
            # Register PointCloud2 schema
            # self._pc2_schema_id = self._lidar_writer.register_schema(
            #     name="sensor_msgs/msg/PointCloud2",
            #     encoding="ros2msg",
            #     data=b"",
            # )
            # self._lidar_channel_id = self._lidar_writer.register_channel(
            #     schema_id=self._pc2_schema_id,
            #     topic="/jackal/lidar_points",
            #     message_encoding="cdr",
            # )

            sys.stderr.write(f"✅ MCAP LIDAR Writer initialized successfully: {lidar_output_path}\n")
            self._lidar_message_count = 0
        except Exception as e:
            sys.stderr.write(f"[WARN] MCAP LIDAR Writer init failed: {e}\n")
            self._lidar_writer = None

    '''def _lidar_write_point_cloud(self, sim_time, points):
        if self._lidar_writer is None:
            return
        try:
            ns_time = int(sim_time * 1e9)
            if points is None or points.size == 0:
                return
            if isinstance(points, np.ndarray):
                pc_data = points.astype(np.float32).tobytes()
            else:
                pc_data = np.array(points, dtype=np.float32).tobytes()
            
            self._lidar_writer.add_message(
                channel_id=self._lidar_channel_id,
                log_time=ns_time,
                publish_time=ns_time,
                data=pc_data,
            )
            self._lidar_message_count = getattr(self, '_lidar_message_count', 0) + 1
        except Exception as e:
            sys.stderr.write(f"⚠️ LIDAR point cloud write failed: {e}\n")'''

    def _lidar_write_point_cloud(self, sim_time, points):
        if self._lidar_writer is None:
            return

        try:
            if points is None or points.size == 0:
                return
            
            points_f32 = points.astype(np.float32)
            
            header = Header()
            header.frame_id = self.base_frame
            
            seconds = int(sim_time)
            nanoseconds = int((sim_time - seconds) * 1e9)
            header.stamp.sec = seconds
            header.stamp.nanosec = nanoseconds

            msg = pc2.create_cloud_xyz32(header, points_f32)

            serialized_msg = serialize_message(msg)

            ns_time = int(sim_time * 1e9)
            # Must match the topic registered in register_msg_channel above, hence the same
            # derivation rather than a second literal.
            self._lidar_ros_writer.write_message(
            topic=f"/{self.robot_name}/lidar_points",
            message=msg,
            log_time=ns_time,
            publish_time=ns_time
            )
            # self._lidar_writer.add_message(
            #     channel_id=self._lidar_channel_id,
            #     log_time=ns_time,
            #     publish_time=ns_time,
            #     data=serialized_msg,
            # )
            
            self._lidar_message_count = getattr(self, '_lidar_message_count', 0) + 1
            if self._lidar_message_count % 10 == 0:
                print(f"Successfully wrote {self._lidar_message_count} point cloud frames to MCAP")
        except Exception as e:
            sys.stderr.write(f"⚠️ LIDAR Writing Failed: {e}\n")

    def _lidar_close(self):
        if hasattr(self, "_lidar_ros_writer") and self._lidar_ros_writer:
            try:
                msg_count = getattr(self, '_lidar_message_count', 0)
                self._lidar_ros_writer.finish()
                self._lidar_file.close()
                self._lidar_ros_writer = None
                self._lidar_file = None
                sys.stderr.write(f"[Save] LIDAR: wrote {msg_count} point cloud messages\n")
            except Exception as e:
                sys.stderr.write(f"[Save] Failed to close LIDAR file: {e}\n")

    def save_points_as_ply(self, points, filename):
        """
        Save numpy array (N, 3) as a PLY file
        """
        if points is None or len(points) == 0:
            sys.stderr.write(f"⚠️ Warning: {filename} data is empty, skipping save.\n")
            return

        points = points.astype(np.float32)
        
        header = f"""ply
        format ascii 1.0
        element vertex {len(points)}
        property float x
        property float y
        property float z
        end_header
        """

        with open(filename, 'w') as f:
            f.write(header)
            np.savetxt(f, points, fmt='%f %f %f')

        sys.stderr.write(f"✅ Save pointcloud to: {filename}\n")

    def _start_rosbag_record(self):
            """Start ros2 bag recording in the background"""
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Define a unique bag folder name
            bag_name = f"episode_{int(time.time())}"
            self.bag_path = os.path.join(self.output_dir, bag_name)
            
            # Construct ROS 2 recording command
            # -s mcap: save in mcap format
            # -o <path>: output path
            cmd = [
                "ros2", "bag", "record",
                "-s", "mcap",
                "-o", self.bag_path,
                self.lidar_topic,
                "/tf", 
                "/tf_static"
            ]
            
            sys.stderr.write(f"🚀 [Logger] Starting background recording: {' '.join(cmd)}\n")
            
            # Start background process with Popen
            self.record_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL, # hide normal output
                stderr=subprocess.PIPE     # keep errors for debugging
            )
            
            # Prevent frame drops
            # ros2 bag startup and topic discovery takes about 0.5~1s
            # If we don't wait, the simulator starts immediately and the first frames will be lost
            time.sleep(1.5)
            sys.stderr.write(f"✅ [Logger] Recording process ready, start collecting data!\n")

    def _rosbag_close(self):
        """Safely stop recording process (equivalent to Ctrl+C)"""
        if self.record_process and self.record_process.poll() is None:
            print(f"🛑 [Logger] Preparing to finalize MCAP file...")
            
            # Send SIGINT to child (equivalent to Ctrl+C)
            # After SIGINT, rosbag calls finish() to ensure index integrity
            self.record_process.send_signal(signal.SIGINT)
            
            try:
                # Wait for process to exit safely, up to 5 seconds
                self.record_process.wait(timeout=5.0)
                print(f"💾 [Logger] MCAP file safely saved to: {self.bag_path}")
            except subprocess.TimeoutExpired:
                print("⚠️ [Logger] Recording process unresponsive, force terminating!")
                self.record_process.kill()

    def find_bone_root_recursive(self, prim):
        """Recursively traverse children to find path containing 'RL_BoneRoot'"""
        if "RL_BoneRoot" in prim.GetName():
            return prim.GetPath().pathString
        for child in prim.GetChildren():
            res = self.find_bone_root_recursive(child)
            if res:
                return res
        return None

    def initialize_pedestrians(self):
        """
        After the environment is ready, scan and bind all pedestrian motion nodes
        """
        pedestrian_root_prim = get_prim_at_path(self.pedestrian_root_path)
        if not pedestrian_root_prim:
            # Environment not fully loaded; return and retry next step
            sys.stderr.write(f"\n🧋 Pedestrians not fully loaded\n")
            return

        # Get Pedestrian_0, Pedestrian_1...
        child_paths = [c.GetPath().pathString for c in pedestrian_root_prim.GetChildren()]
        if not child_paths:
            return
            
        child_paths.sort()
        temp_list = []
        
        for ped_path in child_paths:
            if ped_path.endswith("_lidar_proxy"):
                continue
            # Recursively find node with animated motion
            bone_path = self.find_bone_root_recursive(get_prim_at_path(ped_path))
            
            if bone_path:
                # Only wrap in XFormPrim if the path exists
                temp_list.append(XFormPrim(bone_path))
                sys.stderr.write(f"\n✅ Successfully bound dynamic node: {bone_path}\n")
            else:
                # If bone root not found, bind to Pedestrian_x to keep data flowing
                temp_list.append(XFormPrim(ped_path))
                sys.stderr.write(f"\n❌ Using root node as fallback: {ped_path}\n")

        self.pedestrian_prims = temp_list

    def process_camera_data(self,params, width, height):
        """
        Parse, clean, and normalize camera parameters
        :param params: params dict returned by replicator
        :param width: Image width (pixels)
        :param height: Image height (pixels)
        :return: (cleaned Pose list, cleaned Intrinsics list)
        """
        
        # -----------------------------
        # 1. Process pose
        # -----------------------------
        # Isaac Sim returns World-to-Camera (View Matrix)
        # and it is row-major
        view_matrix = params['cameraViewTransform'].reshape(4, 4)
        
        # Invert to get Camera-to-World (Pose Matrix)
        # Still row-major
        pose_matrix_row_major = np.linalg.inv(view_matrix)
        
        # [KEY] Transpose to column-major
        # So the last column is translation [x, y, z, 1]
        pose_matrix_col_major = pose_matrix_row_major.T
        
        # OpenGL/USD -> ROS optical convention. Isaac cameras look down -Z with +Y up; ROS
        # (REP-103) uses +Z forward with +Y down. Without this the extrinsics disagree with
        # observation.peds_state, which comes from get_world_poses() and is already ROS-compatible
        # USD world -- so the two could not be composed even though both are "world frame".
        #
        # RIGHT-multiply: the 3x3 columns are the camera's own axes, and we are relabelling those
        # axes, not moving the world. Left-multiplying would rotate the world instead and put the
        # camera in the wrong place. Translation is untouched either way, which is the sanity check.
        pose_matrix_col_major = pose_matrix_col_major @ np.diag([1.0, -1.0, -1.0, 1.0])

        # [Clean] keep 6 decimals (enough for navigation precision, turns 1e-19 to 0)
        # This maps 0.9999999 -> 1.0, 1.2e-19 -> 0.0
        pose_clean = np.round(pose_matrix_col_major, decimals=6)
        
        # -----------------------------
        # 2. Process intrinsics
        # -----------------------------
        # Get 4x4 projection matrix
        proj_4x4 = params['cameraProjection'].reshape(4, 4)

        # [0,0] is the normalized focal length for X
        # [1,1] is the normalized focal length for Y
        fx = proj_4x4[0, 0] * width / 2.0
        fy = proj_4x4[1, 1] * height / 2.0

        # Principal point FROM the projection matrix, not assumed to be the image centre.
        # It only equals the centre when the camera has no aperture offset; set
        # horizontal/vertical_aperture_offset on the USD camera and the centre assumption puts
        # cx/cy tens of pixels off, which silently breaks any reprojection while the image still
        # looks fine. fx/fy are on the diagonal so they are unaffected by this.
        #
        # The offsets live off-diagonal, and which slot depends on whether Isaac hands us the
        # matrix row- or column-major (the pose above is row-major, but that is not guaranteed to
        # hold for cameraProjection). Reading whichever slot is populated makes this correct under
        # either convention instead of silently reading a structural zero.
        off_x = proj_4x4[2, 0] if proj_4x4[2, 0] != 0.0 else proj_4x4[0, 2]
        off_y = proj_4x4[2, 1] if proj_4x4[2, 1] != 0.0 else proj_4x4[1, 2]
        # NDC -> pixels. y flips because NDC is +up while pixel rows go +down.
        cx = (1.0 + off_x) * width / 2.0
        cy = (1.0 - off_y) * height / 2.0

        # Build standard 3x3 intrinsics K
        K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ])
        
        # [Intrinsics also keep 6 decimals]
        K_clean = np.round(K, decimals=6)
        
        # camera_sim_time = SimulationContext.get_instance().current_time

        return pose_clean.flatten().tolist(), K_clean.flatten().tolist()

    def process_depth_for_png(self, depth_data, max_dist=65.535):
        """Convert float32 planar depth to uint16 millimetres for lossless PNG.

        Uses distance_to_image_plane annotator: the z-coordinate in camera frame
        (perpendicular distance from the camera's XY plane), NOT the Euclidean
        distance from camera origin to 3D point. The distinction matters for:

        1. Reprojection (pinhole camera model):
              X = (u - cx) * Z / fx
              Y = (v - cy) * Z / fy
           requires Z = planar depth, not radial distance.

        2. Occlusion testing (e.g. gen_pixel_goal_labels.py line 211):
              if np.linalg.norm(p_cam[:3]) > measured + margin:
           When both sides are planar depth Z, the comparison is Z_target vs Z_pixel.
           At image edges, radial distance r exceeds planar depth Z significantly:

           | Location      | r/Z ratio | Example: Z=5m -> r= |
           |---------------|-----------|---------------------|
           | Image center  | 1.00      | 5.0 m (negligible)  |
           | Edge midpoint | 1.41      | 7.1 m (+41%)        |
           | Corner        | 1.52      | 7.6 m (+52%)        |

           The margin parameter (default 0.5 m) tolerates this mismatch near center,
           but extreme corners can still trigger false occlusions if radial distance
           was used. If occlusion failures cluster at image edges, either widen
           margin or verify the annotator is distance_to_image_plane, not
           distance_to_camera.

        Args:
            depth_data: (H, W) float32, planar depth in metres from distance_to_image_plane.
            max_dist: Clipping ceiling in metres. 65.535 = uint16 max at 1 mm scale.

        Returns:
            (H, W) uint16, millimetres. 0 = invalid (NaN/inf in input), not zero distance.
            Downstream must check `depth[v, u] <= 0` to skip invalid pixels.
        """
        depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=max_dist, neginf=0.0)
        depth_data = np.clip(depth_data, 0, max_dist)
        depth_mm = (depth_data * 1000.0).astype(np.uint16)
        return depth_mm  # (H, W) single channel

    def get_pedestrian_state(self):
        """
        Return all pedestrians' 4x4 matrices. If not initialized, try dynamic init.
        """
        # Current sim time
        # sim_time = SimulationContext.get_instance().current_time

        if not self.pedestrian_prims:
            self.initialize_pedestrians()
        
        # If still not found (no pedestrians), return placeholder to avoid Parquet errors
        if not self.pedestrian_prims:
            return {"none": [0.0] * 16}
            # return {"none": [0.0] * 16}, sim_time
        
        curr_peds_dict = {}
        for i, xform_prim in enumerate(self.pedestrian_prims):
            label = f"p_{i+1}"
            
            # Get pose (new XformPrim batched API: get_world_poses returns wp.array)
            positions, orientations = xform_prim.get_world_poses()
            position = np.asarray(positions.numpy()[0])
            orientation = np.asarray(orientations.numpy()[0])
            
            # Quaternion [x, y, z, w] to rotation matrix
            r = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
            rotation_matrix = r.as_matrix()
            
            # Build 4x4 matrix
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = position
            
            # Keep 6-decimal precision and flatten to list
            T_flat = np.round(T, decimals=6).flatten().tolist()
            
            # Store in dict
            curr_peds_dict[label] = T_flat
            
        return curr_peds_dict
    
    @staticmethod
    def _get_render_hz(default: float = 60.0) -> float:
        """Render rate in Hz, read from the live SimulationContext.

        Read rather than hardcoded so the capture rate stays correct if rendering_dt is
        ever changed. Falls back to `default` if the context is not up yet or the stage
        is not open, in which case get_rendering_dt raises.
        """
        try:
            dt = SimulationContext.instance().get_rendering_dt()
            if dt and dt > 0.0:
                return 1.0 / dt
            sys.stderr.write(f"[Logger] rendering_dt={dt!r}, assuming {default:.0f} Hz.\n")
        except Exception as e:
            sys.stderr.write(f"[Logger] Could not read rendering_dt ({e}), assuming {default:.0f} Hz.\n")
        return default

    def reset_render_product(self, warmup_frames: int = 60):
        """Rebuild render product and annotators after robot respawn.

        Call this from _on_task_reset (after the simulation tick that spawns
        the robot), not from inside step(). Isaac needs several rendered frames
        before get_data() returns valid data on a newly created render product.
        """
        try:
            for annot in (self.rgb_annot, self.depth_annot, self.cam_params_annot):
                try:
                    annot.detach(self.camera_rp)
                except Exception:
                    pass
            try:
                self.camera_rp.destroy()
            except Exception:
                pass
            self.camera_rp = rep.create.render_product(self.camera_prim_path, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
            self.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
            self.depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            self.cam_params_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
            self.rgb_annot.attach(self.camera_rp)
            self.depth_annot.attach(self.camera_rp)
            self.cam_params_annot.attach(self.camera_rp)
            self._skip_frames_after_rebuild = warmup_frames
            sys.stderr.write(f"[Logger] Render product rebuilt, skipping {warmup_frames} warmup frames.\n")
        except Exception as e:
            sys.stderr.write(f"[Logger] reset_render_product failed: {e}\n")

    def step(self, step_idx, language_instruction="navigate"):
        """
        Call this after env.step()
        """

        # Warmup skip after annotator rebuild — give the renderer time to settle.
        # Counted in rendered frames (every step() call), before decimation, since
        # it is the renderer that needs to settle, not the dataset.
        if self._skip_frames_after_rebuild > 0:
            self._skip_frames_after_rebuild -= 1
            if self._skip_frames_after_rebuild == 0:
                sys.stderr.write(f"[Logger] Warmup complete, resuming data capture.\n")
            return

        # Capture decimation: with _capture_decimation == 1 this is always on-phase,
        # so the default path is unchanged.
        on_phase = (self._step_call_count % self._capture_decimation) == 0
        self._step_call_count += 1
        if not on_phase:
            return

        # Pre-init variables to avoid UnboundLocalError
        rgb = None
        params = None
        curr_ped_pos = None
        # Same constant that sized the render product above -- see CAPTURE_WIDTH.
        width = CAPTURE_WIDTH
        height = CAPTURE_HEIGHT

        # Try to get camera data
        try:
            rgb = self.rgb_annot.get_data()
            params = self.cam_params_annot.get_data()
            depth = self.depth_annot.get_data()
        except Exception as e:
            sys.stderr.write(f"[Logger] Camera get_data failed: {e}\n")
            return

        # Try to get pedestrian state — failures here should not block camera capture
        try:
            curr_ped_pos = self.get_pedestrian_state()
        except Exception as e:
            sys.stderr.write(f"[Logger] Pedestrian state failed (will reinit next frame): {e}\n")
            self.pedestrian_prims = []  # force re-scan next step
            curr_ped_pos = {"none": [0.0] * 16}

        # Check data integrity
        if rgb is None or depth is None or params is None:
            sys.stderr.write(f"Data missing: RGB={rgb is None}, Params={params is None}\n")
            sys.stderr.flush()
            return
        
        if curr_ped_pos is None: 
            sys.stderr.write(f"Pedestrian data missing.\n")
            sys.stderr.flush()

        # Process RGB (remove alpha channel)
        if rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        # Process depth: quantise to uint16 millimetres (1 mm precision, 0-65.5 m range).
        # The None check above does not cover an empty array, and depth_processed is
        # appended unconditionally below — so bail on the whole frame rather than let
        # it reach the append with depth_processed unbound. Skipping the frame (not
        # just the depth) is what keeps rgb/depth/param buffers index-aligned; losing
        # one of the three would silently desync frame_index from the images.
        if depth.size == 0:
            sys.stderr.write("[Logger] Empty depth array, skipping frame.\n")
            return
        depth_processed = self.process_depth_for_png(depth, max_dist=65.535)

        try:
            # params was already fetched above; re-calling get_data() here was a second
            # annotator round-trip per frame for the same data.
            pose_list, intrinsics_list = self.process_camera_data(
                params,
                width,
                height
            )

            # self.buffer.append({
            #     "episode_index": self.episode_idx,
            #     "frame_index": step_idx,
            #     # "timestamp": step_idx * (1.0/30.0), # Assume 30 FPS
            #     # "instruction": language_instruction,
            # })

            self.rgb_frame_buffer.append(rgb)
            self.depth_frame_buffer.append(depth_processed)
            self.param_buffer.append({
                # Position in this episode's buffer, NOT the caller's step_idx: that one
                # counts every step() call including the ones decimation and warmup drop,
                # which would leave gaps (0, 2, 4, ...) here. param_buffer is appended
                # once per captured frame and only reset per episode (the mid-episode
                # chunk flush clears the image buffers, not this one), so its length is
                # the captured-frame count and needs no separate counter.
                "frame_index": len(self.param_buffer),
                "observation.camera_intrin": intrinsics_list,
                "observation.camera_state": pose_list,
                "observation.peds_state": curr_ped_pos,
            })
            
            # Save MCAP
            '''if lidar_points is not None:
                try:
                    if step_idx == 0 or step_idx % 10 == 0:
                        if isinstance(lidar_points, np.ndarray):
                            sys.stderr.write(f"[DEBUG] Preparing to write LIDAR data: shape={lidar_points.shape}, dtype={lidar_points.dtype}, size={lidar_points.size}\n")
                    
                    sim_time = SimulationContext.instance().current_time
                    self._lidar_write_point_cloud(sim_time, lidar_points)
                except Exception as lidar_save_err:
                    sys.stderr.write(f"⚠️ Failed to save LIDAR point cloud: {lidar_save_err}\n")'''

            # #Save JSON
            # sim_time = SimulationContext.instance().current_time
            # self.json_write_points(points, sim_time, frame_id="jackal/base_link")

            # === Confirm buffer increased ===
            if len(self.param_buffer) % 50 == 0:
                sys.stderr.write(f"✓ Captured frames: {len(self.param_buffer)}\n")
                sys.stderr.flush()

            # === Stream flush: write image chunks to disk to bound memory usage ===
            if len(self.rgb_frame_buffer) >= 500:
                self._flush_stream_chunk()

        except Exception as e:
            sys.stderr.write(f"Buffer data error: {e}\n")

        return

    def _flush_stream_chunk(self):
        """Flush current image buffers to a numbered chunk file on disk, freeing memory."""
        if not self.rgb_frame_buffer:
            return

        ep_idx = self.episode_idx
        chunk_idx = self._stream_chunk_idx

        rgb_frames = self.rgb_frame_buffer
        depth_frames = self.depth_frame_buffer

        self.rgb_frame_buffer = []
        self.depth_frame_buffer = []
        self._stream_chunk_idx += 1

        if self.world:
            # Flat layout: one npy per frame, directly under videos/, no subdirs.
            #   videos/episode_XXXXXX_N.npy   -- N is the frame index within the episode
            # Each file holds both modalities in one dict, because the filename carries no
            # rgb/depth discriminator. Postprocess splits it into rgb PNG + depth PNG.
            # Frame numbering is continuous across chunk flushes (see _frame_write_idx).
            out_dir = self.videos_dir
            start_idx = self._frame_write_idx
            self._frame_write_idx += len(rgb_frames)

            sys.stderr.write(
                f"[Flush] ep={ep_idx:06d} frames {start_idx}..{start_idx + len(rgb_frames) - 1}\n")
            sys.stderr.flush()

            def _write():
                try:
                    for i, (rgb, depth) in enumerate(zip(rgb_frames, depth_frames)):
                        np.save(
                            os.path.join(out_dir, f"episode_{ep_idx:06d}_{start_idx + i}.npy"),
                            {"rgb": rgb, "depth": depth},
                            allow_pickle=True,
                        )
                    sys.stderr.write(
                        f"[Flush] ep {ep_idx:06d}: {len(rgb_frames)} frame npy saved.\n")
                except Exception as e:
                    sys.stderr.write(f"[Flush] ep {ep_idx:06d} frame write failed: {e}\n")
                sys.stderr.flush()
        else:
            # Legacy layout: chunked npy under per-episode rgb_videos/ + depth_videos/.
            ep_dir = os.path.join(self.videos_dir, f"episode_{ep_idx:02d}")
            old_umask = os.umask(0)
            os.makedirs(os.path.join(ep_dir, "rgb_videos"),   mode=0o777, exist_ok=True)
            os.makedirs(os.path.join(ep_dir, "depth_videos"), mode=0o777, exist_ok=True)
            os.umask(old_umask)

            rgb_path   = os.path.join(ep_dir, "rgb_videos",   f"chunk_{chunk_idx:04d}.npy")
            depth_path = os.path.join(ep_dir, "depth_videos", f"chunk_{chunk_idx:04d}.npy")

            sys.stderr.write(f"[Flush] ep={ep_idx:02d} chunk={chunk_idx:04d} ({len(rgb_frames)} frames)\n")
            sys.stderr.flush()

            def _write():
                try:
                    np.save(rgb_path,   np.stack(rgb_frames))
                    np.save(depth_path, np.stack(depth_frames))
                    sys.stderr.write(f"[Flush] chunk {ep_idx:02d}/{chunk_idx:04d} saved.\n")
                except Exception as e:
                    sys.stderr.write(f"[Flush] chunk {ep_idx:02d}/{chunk_idx:04d} failed: {e}\n")
                sys.stderr.flush()

        t = threading.Thread(target=_write, daemon=True, name=f"flush-ep{ep_idx}-ch{chunk_idx}")
        self._pending_saves.append(t)
        t.start()

    @staticmethod
    def _next_episode_idx(data_dir):
        """Highest episode_XXXXXX.json in data_dir, plus one (0 if none).

        Lets a second run of the same --world append instead of clobbering, since the
        world dir has no timestamp level to keep runs apart.
        """
        try:
            names = os.listdir(data_dir)
        except OSError:
            return 0
        indices = [
            int(m.group(1))
            for m in (re.match(r"episode_(\d+)\.json$", n) for n in names)
            if m
        ]
        return max(indices) + 1 if indices else 0

    def save_episode(self):
        """
        Call at the end of an episode.
        Flushes any remaining image frames as a final chunk, then saves the JSON params.
        Image data is written in bounded chunks (500 frames each) to prevent OOM.
        """
        sys.stderr.write(f"\n[Save] Save triggered. Chunk buffer size: {len(self.param_buffer)}\n")
        sys.stderr.flush()
        if not self.param_buffer:
            sys.stderr.write("[Save] Warning: buffer empty, skipping write.\n")
            sys.stderr.flush()
            return

        # Flush remaining image frames as final chunk
        self._flush_stream_chunk()

        param_buf   = self.param_buffer
        episode_idx = self.episode_idx
        n_chunks    = self._stream_chunk_idx  # total chunks written for this episode

        self.episode_idx += 1
        self.param_buffer = []
        self._stream_chunk_idx = 0

        if self.world:
            # One json per episode, flat under data/. Postprocess turns each of these into
            # the matching data/chunk-XXX/episode_XXXXXX.parquet.
            self._frame_write_idx = 0  # next episode restarts frame numbering
            json_path = os.path.join(self.data_dir, f"episode_{episode_idx:06d}.json")
        else:
            ep_data_dir = os.path.join(self.data_dir, f"episode_{episode_idx:02d}")
            old_umask = os.umask(0)
            os.makedirs(ep_data_dir, mode=0o777, exist_ok=True)
            os.umask(old_umask)
            json_path = os.path.join(ep_data_dir, "params.json")

        def _write():
            try:
                with open(json_path, "w") as f:
                    json.dump(param_buf, f)
                sys.stderr.write(f"[Save] Episode {episode_idx:02d} saved ({len(param_buf)} frames, {n_chunks} image chunks).\n")
            except Exception as e:
                sys.stderr.write(f"[Save] Episode {episode_idx:02d} JSON failed: {e}\n")
            sys.stderr.flush()

        t = threading.Thread(target=_write, daemon=True, name=f"save-ep{episode_idx}")
        self._pending_saves.append(t)
        t.start()
        sys.stderr.write(f"[Save] Background save started for episode {episode_idx:02d} ({n_chunks} chunks).\n")
        sys.stderr.flush()

    def discard_episode(self):
        """Discard current episode on abort/cancel or mid-task reset.

        Waits for any in-flight chunk-flush threads, deletes all written files
        under the episode directory, clears in-memory buffers. Does NOT advance
        episode_idx so the next episode reuses the same slot number cleanly.
        """
        ep_idx = self.episode_idx
        ep_dir = os.path.join(self.session_dir, f"episode_{ep_idx:02d}")

        # Clear in-memory buffers immediately
        self.param_buffer = []
        self.rgb_frame_buffer = []
        self.depth_frame_buffer = []

        # Wait for in-flight flush threads belonging to this episode
        prefix = f"flush-ep{ep_idx}-"
        ep_threads = [t for t in self._pending_saves if t.name.startswith(prefix)]
        for t in ep_threads:
            t.join(timeout=10.0)
        self._pending_saves = [t for t in self._pending_saves if t.is_alive()]

        deleted = 0
        if self.world:
            # World layout: frames are flat under videos/ as episode_XXXXXX_N.npy and the
            # params json is flat under data/. The legacy per-episode subdirs below do not
            # exist here, so without this branch a discarded episode left every frame it
            # had already flushed on disk -- orphans that postprocess would later trip on.
            frame_pat = re.compile(rf"episode_{ep_idx:06d}_\d+\.npy$")
            try:
                for fname in os.listdir(self.videos_dir):
                    if frame_pat.match(fname):
                        try:
                            os.remove(os.path.join(self.videos_dir, fname))
                            deleted += 1
                        except Exception as e:
                            sys.stderr.write(f"[Discard] Failed to delete {fname}: {e}\n")
            except OSError as e:
                sys.stderr.write(f"[Discard] Could not list {self.videos_dir}: {e}\n")
            # A json for this slot only exists if a PREVIOUS attempt at this episode index
            # got as far as save_episode(); the current attempt has not written one yet
            # (params are held in memory until then). Remove it so the retry starts clean.
            if os.path.exists(json_path := os.path.join(
                    self.data_dir, f"episode_{ep_idx:06d}.json")):
                try:
                    os.remove(json_path)
                    deleted += 1
                except Exception as e:
                    sys.stderr.write(f"[Discard] Failed to delete {json_path}: {e}\n")
            self._frame_write_idx = 0   # next attempt restarts frame numbering
        else:
            # Legacy layout: everything lives under the per-episode dir.
            for subdir in ("rgb_videos", "depth_videos", "data"):
                subdir_path = os.path.join(ep_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                for fname in os.listdir(subdir_path):
                    fpath = os.path.join(subdir_path, fname)
                    try:
                        os.remove(fpath)
                        deleted += 1
                    except Exception as e:
                        sys.stderr.write(f"[Discard] Failed to delete {fpath}: {e}\n")

        self._stream_chunk_idx = 0
        # Do NOT advance episode_idx — next attempt reuses the same slot
        sys.stderr.write(
            f"[Discard] Episode {ep_idx:02d} discarded "
            f"({deleted} file(s) deleted). Ready to re-record.\n"
        )
        sys.stderr.flush()

    def wait_for_pending_saves(self, timeout: float = 60.0):
        """Block until all background save threads finish (or timeout expires)."""
        alive = [t for t in self._pending_saves if t.is_alive()]
        if alive:
            sys.stderr.write(f"[Save] Waiting for {len(alive)} background save thread(s)...\n")
            sys.stderr.flush()
        for t in alive:
            t.join(timeout=timeout)
            if t.is_alive():
                sys.stderr.write(f"[Save] WARNING: thread {t.name} did not finish within {timeout}s\n")
                sys.stderr.flush()
        self._pending_saves = [t for t in self._pending_saves if t.is_alive()]

