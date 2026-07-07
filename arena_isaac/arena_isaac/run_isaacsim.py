# fmt: off


# preload attrs
import argparse
import os
import arena_simulation_setup
import arena_simulation_setup.utils.cattrs

# Use the isaacsim to import SimulationApp
from isaacsim import SimulationApp

# Setting the config for simulation and make an simulation.
CONFIG = {
    #"renderer": "Wireframe",
    "renderer": "RayTracedLighting",
    "headless": False,
}
#import parent directory
import sys
from pathlib import Path

simulation_app = SimulationApp(CONFIG)
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(parent_dir))

# stdlib
import queue
import random
import traceback

# Import Isaac Sim dependencies

import carb
import omni.kit.commands as commands
import omni.timeline
import omni.usd
import yaml
from isaac_utils.utils.assets import get_assets_root_path_safe
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.urdf")
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.utils import extensions, prims, stage
from pxr import Sdf

try:
    from isaacsim.core.world import World
except ImportError:
    from omni.isaac.core import World

try:
    from isaacsim.core.api.simulation_context import SimulationContext
except ImportError:
    from omni.isaac.core import SimulationContext

EXTENSIONS_PEOPLE = [
    'omni.anim.people', 
    'omni.anim.navigation.bundle', 
    'omni.anim.timeline',
    'omni.anim.graph.bundle', 
    'omni.anim.graph.core', 
    'omni.anim.graph.ui',
    'omni.anim.retarget.bundle', 
    'omni.anim.retarget.core',
    'omni.anim.retarget.ui', 
    'omni.kit.scripting',
    'omni.graph.nodes',
    'omni.anim.curve.core',
    'omni.anim.navigation.core'
]

EXTENSIONS_MATERIAL = [
    'omni.kit.material.library',
    'omni.kit.browser.material',
    'omni.kit.browser.asset',
    'omni.kit.window.material'
]
for ext_people in EXTENSIONS_PEOPLE:
    extensions.enable_extension(ext_people)

for ext_material in EXTENSIONS_MATERIAL:
    extensions.enable_extension(ext_material)

import tomllib

def enable_extensions_from_kit(kit_path):
    with open(kit_path, "rb") as f:
        data = tomllib.load(f)
        dependencies = data.get("dependencies", {})

        for ext_name in dependencies.keys():
            # Skip filter selectors in .kit files (e.g. "filter:platform"),
            # they are not real extension names and cannot be enabled directly.
            if isinstance(ext_name, str) and ext_name.startswith("filter:"):
                continue
            print(f"Enabling: {ext_name}")
            enable_extension(ext_name)

# --- In your main code ---
KIT_FILE_PATH = "/isaac-sim/apps/isaacsim.exp.full.kit"
enable_extensions_from_kit(KIT_FILE_PATH)

# Update the simulation app with the new extensions
for _ in range(100):
    simulation_app.update()

# -------------------------------------------------------------------------------------------------
# These lines are needed to restart the USD stage and make sure that the people extension is loaded
# -------------------------------------------------------------------------------------------------
omni.usd.get_context().new_stage()

try:
    from isaacsim.core.utils.prims import define_prim
except ImportError:
    from omni.isaac.core.utils.prims import define_prim

# 显式定义根节点和分类容器，防止服务调用时这些路径不存在
define_prim("/World", "Xform")
define_prim("/World/Walls", "Xform")
define_prim("/World/Doors", "Xform")
define_prim("/World/Floors", "Xform")
define_prim("/World/Obstacles", "Xform")
define_prim("/World/Pedestrians", "Xform")

extensions.enable_extension("isaacsim.ros2.bridge")
extensions.enable_extension("isaacsim.sensors.physics")
extensions.enable_extension("isaacsim.sensors.camera")
extensions.enable_extension("isaacsim.sensors.rtx")

import numpy as np

#Import world generation dependencies
import omni.anim.graph.core as ag

#imprt navmesh gen
import omni.anim.navigation.core as nav
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd

# rclpy
import rclpy
import rclpy.node
import std_srvs.srv
import std_msgs.msg
import action_msgs.msg

# graphs
from isaac_utils.graphs.time import PublishTime
from isaac_utils.managers.door_manager import DoorManager
from isaac_utils.managers.elevator_manager import elevator_manager

#Import services
from arena_isaac.services import services
from pedestrian.simulator.logic.people_manager import PeopleManager
from rclpy.qos import QoSProfile
from arena_isaac import run_after_tick_queue
import traceback
try:
    from isaacsim.core.utils.prims import is_prim_path_valid
except ImportError:
    from omni.isaac.core.utils.prims import is_prim_path_valid

# fmt: on
# ======================================Base======================================
# Setting up world and enable ros2_bridge extentions.
# BACKGROUND_STAGE_PATH = "/background"
# BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
plane_material_paths = [
    'https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/Base/Wood/Walnut_Planks.mdl',
    # 'https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl',
    # 'https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl'
]
world = World()
world.scene.add_ground_plane(size=100, z_position=0.0)
_stage = omni.usd.get_context().get_stage()
plane_mdl_path = random.choice(plane_material_paths)
plane_mtl_name = plane_mdl_path.split('/')[-1][:-4]
plane_mtl_path = "/World/Looks/PlaneMaterial"
plane_mtl = _stage.GetPrimAtPath(plane_mtl_path)
# if not (plane_mtl and plane_mtl.IsValid()):
#     create_res = omni.kit.commands.execute('CreateMdlMaterialPrimCommand',
#                                                 mtl_url=plane_mdl_path,
#                                                 mtl_name=plane_mtl_name,
#                                                 mtl_path=plane_mtl_path)

#     bind_res = omni.kit.commands.execute('BindMaterialCommand',
#                                             prim_path="/World/groundPlane",
#                                             material_path=plane_mtl_path)
simulation_app.update()  # update the simulation once for update ros2_bridge.
simulation_context = SimulationContext(stage_units_in_meters=1.0)  # currently we use 1m for simulation.
light_1 = prims.create_prim(
    "/World/Light_1",
    "DomeLight",
    position=np.array([1.0, 1.0, 1.0]),
    attributes={
        "inputs:texture:format": "latlong",
        "inputs:intensity": 1000.0,
        "inputs:color": (1.0, 1.0, 1.0)
    }
)
assets_root_path = get_assets_root_path_safe()

# navmesh_enabled stays TRUE (default) so omni.anim.people properly registers characters
# and initializes animation graphs (ag.get_character() works). 
# dynamic_avoidance_enabled=False: disables agent-agent collision avoidance (handled by hunav SFM instead)
omni.kit.commands.execute(
    'ChangeSetting',
    path='/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled',
    value=False)
simulation_app.update()

# =================================================================================

# ===========================raycast obstacle publisher============================
# Publishes per-agent obstacle data from PhysX raycasts to hunav.py

import math as _math
import omni.physx
from arena_people_msgs.msg import Pedestrians
from hunav_msgs.msg import Agent, Agents
from geometry_msgs.msg import Point as GeoPoint


class RaycastObstaclePublisher(rclpy.node.Node):
    """Runs in Isaac Sim process: casts PhysX rays around each pedestrian
    and publishes obstacle hits to hunav.py's _obstacle_subscriber.

    Mirrors hunav_isaac_wrapper's get_closest_obstacles() approach:
      - 36 rays covering 360° (wrapper uses 90; 36 balances accuracy/perf)
      - 4 sensor heights to catch low/mid/high wall geometry
      - uses hit['position'] directly (exact world-space hit point)
    """

    NUM_RAYS = 36
    RAY_DISTANCE = 4.0
    SENSOR_HEIGHTS = [0.1, 0.25, 0.5, 1.0]

    def __init__(self, peds_topic: str, obstacles_topic: str):
        super().__init__(node_name='raycast_obstacle_publisher')
        self._publisher = self.create_publisher(Agents, obstacles_topic, 10)
        self._subscriber = self.create_subscription(
            Pedestrians, peds_topic, self._peds_callback, 10
        )
        self._physx_query = omni.physx.get_physx_scene_query_interface()
        self.get_logger().info(
            f'RaycastObstaclePublisher: peds={peds_topic}, obs={obstacles_topic}'
        )

    
    def _peds_callback(self, msg: Pedestrians):
        result = Agents()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = 'map'

        angle_step =2.0 * _math.pi / self.NUM_RAYS

        for ped in msg.pedestrians:
            agent = Agent()
            agent.name = ped.name

            ox = ped.pose.position.x
            oy = ped.pose.position.y
            oz = ped.pose.position.z  

            for i in range(self.NUM_RAYS):
                angle = angle_step * i
                dx = _math.cos(angle)
                dy = _math.sin(angle)

                best_dist = self.RAY_DISTANCE
                best_pos = None

                # Cast at each sensor height; keep the closest hit
                for h in self.SENSOR_HEIGHTS:
                    hit = self._physx_query.raycast_closest(
                        carb.Float3(ox, oy, oz + h),
                        carb.Float3(dx, dy, 0.0),
                        self.RAY_DISTANCE
                    )
                    if hit and hit.get('hit', False):
                        d = hit.get('distance', self.RAY_DISTANCE)
                        if d < best_dist:
                            best_dist = d
                            best_pos = hit.get('position') # exact world-space hit
                
                if best_pos is not None:
                    pt = GeoPoint()
                    pt.x = float(best_pos[0])
                    pt.y = float(best_pos[1])
                    pt.z = float(best_pos[2])
                    agent.closest_obs.append(pt)
                
            result.agents.append(agent)
        
        self._publisher.publish(result)



# =================================================================================

# ===================================controller====================================
# create controller node for isaacsim.


class IsaacController(rclpy.node.Node):
    def __init__(self, *args, **kwargs):
        super().__init__(node_name="isaac", *args, **kwargs)
        self._running = False
        self._should_step_once = False
        self._capture_requested = False

        self.__pause_srv = self.create_service(
            std_srvs.srv.Trigger,
            os.path.join('isaac/PauseSimulation'),
            self._cb_pause,
        )
        self.__unpause_srv = self.create_service(
            std_srvs.srv.Trigger,
            os.path.join('isaac/UnpauseSimulation'),
            self._cb_unpause,
        )
        self.__step_srv = self.create_service(
            std_srvs.srv.Trigger,
            os.path.join('isaac/StepSimulation'),
            self._cb_step,
        )
        self.__capture_srv = self.create_service(
            std_srvs.srv.Trigger,
            os.path.join('isaac/CaptureSnapshot'),
            self._cb_capture,
        )

        # Publish sim running state (latched) so external nodes — e.g. the
        # proactive-yielding trigger — can react to pause/unpause transitions.
        from rclpy.qos import QoSProfile, DurabilityPolicy
        _state_qos = QoSProfile(depth=1)
        _state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.__running_pub = self.create_publisher(
            std_msgs.msg.Bool, 'isaac/sim_running', _state_qos)
        self._publish_running_state()

        # Publish "snapshot finished, dir=<path>" so the social-replan
        # orchestrator can pick it up (select goal -> reproject -> unpause).
        self.__snapshot_ready_pub = self.create_publisher(
            std_msgs.msg.String, 'isaac/snapshot_ready', 10)

    def _publish_running_state(self):
        try:
            self.__running_pub.publish(std_msgs.msg.Bool(data=bool(self._running)))
        except Exception:  # noqa: BLE001
            pass

    def publish_snapshot_ready(self, out_dir: str):
        try:
            self.__snapshot_ready_pub.publish(std_msgs.msg.String(data=str(out_dir)))
            self.get_logger().info(f"Published snapshot_ready: {out_dir}")
        except Exception:  # noqa: BLE001
            pass

    def _cb_pause(self, request: std_srvs.srv.Trigger.Request, response: std_srvs.srv.Trigger.Response):
        self._running = False
        self._publish_running_state()
        self.get_logger().info("Simulation PAUSED (external request)")
        response.success = True
        return response

    def _cb_unpause(self, request: std_srvs.srv.Trigger.Request, response: std_srvs.srv.Trigger.Response):
        self._running = True
        self._publish_running_state()
        self.get_logger().info("Simulation UNPAUSED (external request)")
        response.success = True
        return response

    def _cb_step(self, request: std_srvs.srv.Trigger.Request, response: std_srvs.srv.Trigger.Response):
        self._should_step_once = True
        response.success = True
        return response

    def _cb_capture(self, request: std_srvs.srv.Trigger.Request, response: std_srvs.srv.Trigger.Response):
        # Method A: only raise the flag. The main loop performs the actual
        # capture while the sim is paused (render context lives there).
        self._capture_requested = True
        self.get_logger().info("Snapshot requested (queued for paused main loop)")
        response.success = True
        response.message = "queued"
        return response

    @property
    def capture_requested(self) -> bool:
        return self._capture_requested

    def clear_capture_request(self):
        self._capture_requested = False

    @property
    def _step_once(self) -> bool:
        v = self._should_step_once
        self._should_step_once = False
        return v

    @property
    def running(self):
        return self._step_once or self._running

    @classmethod
    def wait_for_bridge(cls):
        extensions.enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

        from isaacsim.ros2.bridge._ros2_bridge import acquire_ros2_bridge_interface
        ros2_bridge = acquire_ros2_bridge_interface()
        while not ros2_bridge.get_startup_status():
            simulation_app.update()

        carb.log_info("ROS 2 bridge started successfully!")


# ======================================main=======================================
from arena_isaac.data_logging.data_logger_replicator import DataLoggerReplicator
import signal


def _resolve_data_logger_prim_paths(robot_model: str) -> tuple[str, str, str] | None:
    """Read data logger USD prim paths from arena_robots/robots/<model>/model_params.yaml.

    Expected YAML block (optional): ``data_logger`` with keys:
      - robot_stage_name: last segment under /World/Robots/ (e.g. robot0 or Ai2_Bot2)
      - camera_prim_suffix: path under that robot prim to the Camera prim
      - lidar_prim_suffix: path under that robot prim to the lidar link prim
      - pedestrian_root_path: optional, default /World/Pedestrians

    Returns (lidar_path, camera_path, pedestrian_root) or None if unset/incomplete.
    """
    try:
        from ament_index_python.packages import get_package_share_path
    except Exception:
        return None

    yaml_path = get_package_share_path('arena_robots') / 'robots' / robot_model / 'model_params.yaml'
    if not yaml_path.is_file():
        return None

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    logger_cfg = data.get('data_logger')
    if not isinstance(logger_cfg, dict):
        return None

    stage = logger_cfg.get('robot_stage_name')
    cam_suffix = logger_cfg.get('camera_prim_suffix')
    lidar_suffix = logger_cfg.get('lidar_prim_suffix')
    if not stage or not cam_suffix or not lidar_suffix:
        return None

    root = os.path.join('/World', 'Robots', str(stage))
    lidar_path = os.path.join(root, str(lidar_suffix))
    camera_path = os.path.join(root, str(cam_suffix))
    ped_root = str(logger_cfg.get('pedestrian_root_path') or '/World/Pedestrians')
    return lidar_path, camera_path, ped_root


def _resolve_snapshot_config(robot_model: str) -> dict | None:
    """Read the ``snapshot`` block from arena_robots/robots/<model>/model_params.yaml.

    Expected YAML block (optional): ``snapshot`` with keys:
      - head_camera_suffix / back_camera_suffix: path under /World/Robots/<stage>
        to each Camera prim (either may be omitted → that view is skipped)
      - topdown_half_extent: half side length (m) the top-down view covers
      - topdown_height: camera height (m) above the robot

    ``robot_stage_name`` is reused from the ``data_logger`` block so the
    robot root path stays consistent. Returns a dict with resolved absolute
    prim paths and top-down params, or None if unset/incomplete.
    """
    try:
        from ament_index_python.packages import get_package_share_path
    except Exception:
        return None

    yaml_path = get_package_share_path('arena_robots') / 'robots' / robot_model / 'model_params.yaml'
    if not yaml_path.is_file():
        return None

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    snap = data.get('snapshot')
    if not isinstance(snap, dict):
        return None

    logger_cfg = data.get('data_logger') or {}
    stage = snap.get('robot_stage_name') or logger_cfg.get('robot_stage_name')
    if not stage:
        return None

    root = os.path.join('/World', 'Robots', str(stage))
    head_suffix = snap.get('head_camera_suffix')
    back_suffix = snap.get('back_camera_suffix')
    # base_link tracks the robot as it moves (the root Xform stays at spawn).
    # Fall back to deriving it from the lidar suffix (drop the trailing link).
    base_suffix = snap.get('base_link_suffix')
    if not base_suffix:
        lidar_suffix = logger_cfg.get('lidar_prim_suffix')
        if lidar_suffix:
            base_suffix = str(lidar_suffix).rsplit('/', 1)[0]  # drop /lidar_link
    return {
        'robot_root_path': root,
        'base_link_path': os.path.join(root, str(base_suffix)) if base_suffix else root,
        'head_camera_path': os.path.join(root, str(head_suffix)) if head_suffix else '',
        'back_camera_path': os.path.join(root, str(back_suffix)) if back_suffix else '',
        'topdown_half_extent': float(snap.get('topdown_half_extent', 5.0)),
        'topdown_height': float(snap.get('topdown_height', 8.0)),
    }


def _get_prim_world_position(prim_path: str) -> tuple[float, float, float]:
    """Return the world-space (x, y, z) translation of a prim.

    Uses the batched XformPrim API (same one the data logger uses for
    pedestrians). Falls back to (0, 0, 0) on failure.
    """
    try:
        try:
            from isaacsim.core.experimental.prims import XformPrim as XFormPrim
        except ImportError:
            from omni.isaac.core.prims import XFormPrim
        positions, _ = XFormPrim(prim_path).get_world_poses()
        p = positions.numpy()[0]
        return float(p[0]), float(p[1]), float(p[2])
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[Snapshot] world position lookup failed for {prim_path}: {e}\n")
        return 0.0, 0.0, 0.0


def main(args=None):
    """
    Main function to initialize the simulation, create the ROS 2 node,
    and run the simulation loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-data', type=str, default='false',
                       help='Enable dataset logging')
    parser.add_argument('--robot', type=str, default='jackal',
                       help='Robot name used in task_generator topic namespace')
    parser.add_argument('--log-level', type=str, default='info',
                       help='log level for IsaacSim (debug/info/warn/error)')
    parser.add_argument('--session-tag', type=str, default='',
                       help='Label prepended to the session directory, e.g. grscenes_1__default')
    parsed_args = parser.parse_args(args)

    enable_logging = parsed_args.save_data.lower() == 'true'
    robot_name = parsed_args.robot
    session_tag = parsed_args.session_tag
    data_logger_replicator_cls = None
    # data_logger_rosbag_cls = None

    data_logger_paths = _resolve_data_logger_prim_paths(robot_name)
    if data_logger_paths is None:
        target_lidar_path = ""
        target_camera_path = ""
        target_pedestrian_root_path = "/World/Pedestrians"
    else:
        target_lidar_path, target_camera_path, target_pedestrian_root_path = data_logger_paths

    if enable_logging and data_logger_paths is None:
        enable_logging = False
        sys.stderr.write(
            "[WARN] Data logging disabled: missing or incomplete "
            f"'data_logger' in arena_robots/robots/{robot_name}/model_params.yaml "
            "(need robot_stage_name, camera_prim_suffix, lidar_prim_suffix).\n"
        )
        sys.stderr.flush()
    
    # data logger state
    logger = None
    frame_step_counter = 0
    has_saved = False
    collecting = False   # True only while robot is actively executing a task

    # Snapshot capturer state (proactive-yielding trigger → paused → capture).
    # Lazily created on first capture request; needs the robot to be spawned.
    snapshot_capturer = None
    snapshot_config = _resolve_snapshot_config(robot_name)

    def _snapshot_output_root():
        """Base dir for snapshots/, mirroring the data logger output layout."""
        _p = Path(__file__).resolve()
        while _p.name != 'Arena' and _p.parent != _p:
            _p = _p.parent
        base = _p / 'collected_data'
        return str(base / session_tag) if session_tag else str(base)


    if enable_logging:
        try:
            from arena_isaac.data_logging.data_logger_replicator import DataLoggerReplicator
            # from arena_isaac.data_logging.data_logger_rosbag import DataLoggerRosbag

            data_logger_replicator_cls = DataLoggerReplicator
            # data_logger_rosbag_cls = DataLoggerRosbag
            sys.stderr.write(
                "[INFO] Data logging modules loaded successfully.\n"
            )
            sys.stderr.flush()

        except Exception as e:
            enable_logging = False
            sys.stderr.write(
                f"[WARN] Data logging disabled because optional dependencies are unavailable: {e}\n"
            )
            sys.stderr.flush()

    # apply log level if requested
    try:
        ll = parsed_args.log_level.lower()
        # carb logging thresholds are uppercase
        log.set_level_threshold(getattr(carb.logging, f"LEVEL_{ll.upper()}"))
    except Exception:
        pass

    sim = SimulationContext()

    IsaacController.wait_for_bridge()
    rclpy.init()
    controller = IsaacController()

    # Subscribe to task lifecycle events to control when data collection happens
    def _on_task_reset(msg: std_msgs.msg.Int16):
        """task_generator fires this after each reset — treat as episode START.
        If there is leftover data from an interrupted episode, save it first."""
        nonlocal collecting, frame_step_counter
        if not enable_logging or logger is None:
            return

        episode_num = msg.data

        def _start_episode():
            nonlocal collecting, frame_step_counter
            # Discard leftover buffer if robot was interrupted mid-task (no save signal received)
            if collecting or len(logger.param_buffer) > 0:
                sys.stderr.write(f"[TaskReset #{episode_num}] Discarding incomplete episode ({len(logger.param_buffer)} frames)...\n")
                try:
                    logger.discard_episode()
                except Exception as e:
                    sys.stderr.write(f"[TaskReset #{episode_num}] Discard failed: {e}\n")
            # Clear stale pedestrian prims — they are deleted/respawned on reset
            logger.pedestrian_prims = []
            frame_step_counter = 0
            collecting = True
            sys.stderr.write(f"[TaskReset #{episode_num}] Episode started, collecting data.\n")
            sys.stderr.flush()

        run_after_tick_queue.put_nowait(_start_episode)

    def _on_nav_status(msg: action_msgs.msg.GoalStatusArray):
        """Nav2 action status — save on success, discard on abort/cancel."""
        nonlocal collecting
        if not enable_logging or logger is None or not collecting:
            return
        last = next(reversed(list(msg.status_list)), None)
        if last is None:
            return
        status = last.status
        terminal = frozenset({
            action_msgs.msg.GoalStatus.STATUS_SUCCEEDED,
            action_msgs.msg.GoalStatus.STATUS_ABORTED,
            action_msgs.msg.GoalStatus.STATUS_CANCELED,
        })
        if status not in terminal:
            return
        succeeded = status == action_msgs.msg.GoalStatus.STATUS_SUCCEEDED
        label = "SUCCEEDED" if succeeded else "ABORTED/CANCELED"

        def _end_episode():
            nonlocal collecting
            collecting = False
            if not succeeded:
                sys.stderr.write(f"[NavStatus {label}] Discarding episode...\n")
                try:
                    logger.discard_episode()
                    sys.stderr.write(f"[NavStatus {label}] Discard complete.\n")
                except Exception as e:
                    sys.stderr.write(f"[NavStatus {label}] Discard failed: {e}\n")
                sys.stderr.flush()
                return
            if len(logger.param_buffer) == 0:
                return
            sys.stderr.write(f"[NavStatus {label}] Saving episode ({len(logger.param_buffer)} frames)...\n")
            try:
                logger.save_episode()
                logger.wait_for_pending_saves(timeout=120.0)
                sys.stderr.write(f"[NavStatus {label}] Save complete.\n")
                _pub_episode_saved.publish(std_msgs.msg.Empty())
            except Exception as e:
                sys.stderr.write(f"[NavStatus {label}] Save failed: {e}\n")
            sys.stderr.flush()

        run_after_tick_queue.put_nowait(_end_episode)

    def _on_finished(_msg: std_msgs.msg.Empty):
        """Fired when all desired episodes are done — save whatever remains."""
        nonlocal collecting
        if not enable_logging or logger is None:
            return

        def _save():
            nonlocal collecting
            collecting = False
            if len(logger.param_buffer) == 0:
                return
            sys.stderr.write(f"[Finished] Saving final episode ({len(logger.param_buffer)} frames)...\n")
            try:
                logger.save_episode()
                sys.stderr.write("[Finished] Final save complete.\n")
            except Exception as e:
                sys.stderr.write(f"[Finished] Final save failed: {e}\n")
            sys.stderr.flush()

        run_after_tick_queue.put_nowait(_save)

    controller.create_subscription(
        std_msgs.msg.Int16, '/task_generator_node/task_reset', _on_task_reset, 10)
    controller.create_subscription(
        action_msgs.msg.GoalStatusArray,
        f'/task_generator_node/{robot_name}/navigate_to_pose/_action/status',
        _on_nav_status, 1)
    controller.create_subscription(
        std_msgs.msg.Empty, '/task_generator_node/finished', _on_finished, 10)

    _pub_episode_saved = controller.create_publisher(
        std_msgs.msg.Empty, '/data_logger/episode_saved', 10)

    door_manager = DoorManager.instance(controller)
    elevator_manager.register_node(controller)  # Register controller for odom subscriptions
    for service in services:
        service.create(controller, qos_profile=QoSProfile(depth=2000))

    # RaycastObstaclePublisher: publishes PhysX raycast hits to hunav's obstacle subscriber
    # Topic names must match: peds published by hunav.py, obstacles consumed by hunav.py
    raycast_pub = RaycastObstaclePublisher(
        peds_topic='/task_generator_node/arena_peds',
        obstacles_topic='/task_generator_node/hunav_closest_obstacles',
    )

    PublishTime('/World/publish_time')
    world.reset()

    # Replicator
    #处理Ctrl+C 退出
    # def emergency_save_handler(signum, frame):        
    #     nonlocal has_saved
    #     sys.stderr.write(f"\n[URGENT] 收到终止信号 ({signum})! 正在保存数据...\n")
    #     sys.stderr.flush()
        
    #     if not has_saved and logger and len(logger.param_buffer) > 0:
    #         try:
    #             # 强制保存
    #             logger.save_episode()
    #             has_saved = True  # 标记已保存
    #             sys.stderr.write("[URGENT] ✅ 数据保存成功！\n")
    #         except Exception as e:
    #             sys.stderr.write(f"[URGENT] ❌ 保存失败: {e}\n")
    #     else:
    #         sys.stderr.write("[URGENT] Buffer 为空，无需保存。\n")
        
    #     sys.stderr.flush()
    #     # 保存完后，手动退出程序
    #     sys.exit(0)
        
    # signal.signal(signal.SIGINT, emergency_save_handler)
    # signal.signal(signal.SIGTERM, emergency_save_handler)
    # signal.signal(signal.SIGHUP, emergency_save_handler)   # docker exec 断开时触发
    # set photoreal settings
    import isaac_utils.config.photoreal as photoreal
    if os.environ.get('RENDER_PRESET', 'photoreal') != 'boring':
        photoreal.PRESET_PHOTOREAL.apply()
    else:
        photoreal.PRESET_DEFAULT.apply()

    # hard reset once
    omni.timeline.get_timeline_interface().stop()

    # mainloop
    was_playing: bool = False
    try:
        while simulation_app.is_running():
            stepped_this_iteration: bool = False
            rclpy.spin_once(controller, timeout_sec=0)
            rclpy.spin_once(raycast_pub, timeout_sec=0)
            if controller.running:
                if not was_playing:
                    world.play()
                    was_playing = True
                door_manager.update()
                elevator_manager.update()
                world.step(render=True)
                stepped_this_iteration = True
                # Start data collection and saving
                if enable_logging:
                    # Check every 50 frames whether the robot appears
                    if logger is None:
                        if frame_step_counter % 50 == 0:
                            if is_prim_path_valid(target_camera_path):
                                try:
                                    _p = Path(__file__).resolve()
                                    while _p.name != 'Arena' and _p.parent != _p:
                                        _p = _p.parent
                                    if session_tag:
                                        _output_dir = str(_p / 'collected_data' / session_tag)
                                    else:
                                        _output_dir = str(_p / 'collected_data')
                                    logger = data_logger_replicator_cls(
                                        camera_prim_path=target_camera_path,
                                        pedestrian_root_path=target_pedestrian_root_path,
                                        lidar_prim_path=target_lidar_path,
                                        output_dir=_output_dir,
                                    )
                                    sys.stderr.write(f"\n [Logger] Output dir: {_output_dir}\n")
                                    '''
                                    logger = data_logger_rosbag_cls(
                                        topics=[
                                            "/task_generator_node/jackal/odom",
                                            "/task_generator_node/jackal/front_camera/camera_info",
                                            "/task_generator_node/jackal/front_camera/image",
                                            "/task_generator_node/jackal/front_camera/depth",
                                            "/task_generator_node/jackal/lidar/points",
                                            "/task_generator_node/human_states",
                                            "/tf",
                                            "/tf_static",
                                        ],
                                        output_dir="collected_data"
                                        )
                                    logger.start_recording()  # start rosbag recording
                                    rosbag_process = logger.process  # rosbag logic
                                    controller.get_logger().info('Rosbag Logger initialized successfully')
                                    '''
                                except Exception as e:
                                    sys.stderr.write(f"\n DataLogger initialization failed: {e}\n")
                            elif frame_step_counter % 300 == 0:
                                sys.stderr.write(f" Waiting for robot spawn... searching path: {target_camera_path}\n")
                    else:
                        if collecting:
                            try:
                                logger.step(step_idx=frame_step_counter) # replicator logic
                                frame_step_counter += 1  # increment after each capture

                            except Exception as e:
                                # Force stack trace to stderr
                                sys.stderr.write(f"\nLogger step crashed: {e}\n")
                                traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
                            sys.stderr.flush()
            else:
                if was_playing:
                    world.pause()
                    was_playing = False
                simulation_app.update()

                # Snapshot capture: only while genuinely paused (this branch),
                # so images are of a frozen scene, never a moving one. The ROS
                # service just raised the flag; the real work happens here where
                # the render context lives.
                if controller.capture_requested:
                    try:
                        sys.stderr.write("[Snapshot] === capture start (paused main loop) ===\n")
                        if snapshot_config is None:
                            sys.stderr.write(
                                "[Snapshot] ABORT: no 'snapshot' block in model_params.yaml "
                                f"for robot '{robot_name}'.\n")
                        else:
                            robot_root = snapshot_config['robot_root_path']
                            sys.stderr.write(
                                f"[Snapshot] step A: checking robot prim {robot_root}\n")
                            if not is_prim_path_valid(robot_root):
                                sys.stderr.write(
                                    f"[Snapshot] ABORT: robot not spawned yet ({robot_root}).\n")
                            else:
                                if snapshot_capturer is None:
                                    from arena_isaac.social_yielding.snapshot_capturer import SnapshotCapturer
                                    snapshot_capturer = SnapshotCapturer(
                                        output_root=_snapshot_output_root(),
                                        simulation_app=simulation_app,
                                    )
                                    sys.stderr.write(
                                        f"[Snapshot] step B: capturer created, "
                                        f"output_root={_snapshot_output_root()}\n")
                                # Center the top-down on the robot's CURRENT pose.
                                # The root Xform stays frozen at spawn, so use
                                # base_link which moves with the articulation.
                                base_link_path = snapshot_config.get('base_link_path') or robot_root
                                robot_pos = _get_prim_world_position(base_link_path)
                                sys.stderr.write(
                                    f"[Snapshot] step C: robot base_link world pos="
                                    f"({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}) "
                                    f"from {base_link_path}\n")
                                sys.stderr.write(
                                    "[Snapshot] step D: calling capture() "
                                    f"(head={snapshot_config['head_camera_path']}, "
                                    f"back={snapshot_config['back_camera_path']})\n")
                                sys.stderr.flush()
                                ok, out_dir = snapshot_capturer.capture(
                                    robot_position=robot_pos,
                                    head_cam_path=snapshot_config['head_camera_path'],
                                    back_cam_path=snapshot_config['back_camera_path'],
                                    topdown_half_extent=snapshot_config['topdown_half_extent'],
                                    topdown_height=snapshot_config['topdown_height'],
                                )
                                if ok:
                                    sys.stderr.write(f"[Snapshot] 截图已保存 -> {out_dir}\n")
                                    # Notify the social-replan orchestrator.
                                    controller.publish_snapshot_ready(out_dir)
                                else:
                                    sys.stderr.write("[Snapshot] 截图失败（无有效视图）\n")
                    except Exception as e:
                        sys.stderr.write(f"[Snapshot] capture crashed: {e}\n")
                        traceback.print_exc(file=sys.stderr)
                    finally:
                        controller.clear_capture_request()
                        sys.stderr.write("[Snapshot] === capture end ===\n")
                        sys.stderr.flush()

            if stepped_this_iteration:
                pending_actions = run_after_tick_queue.qsize()
                for _ in range(pending_actions):
                    try:
                        deferred_action = run_after_tick_queue.get_nowait()
                    except queue.Empty:
                        break

                    try:
                        deferred_action()
                    except Exception as e:
                        carb.log_error(f"Deferred action failed: {e}\n{traceback.format_exc()}")

    except KeyboardInterrupt:
        controller.get_logger().info('Received KeyboardInterrupt, shutting down.')
    except Exception as e:
        controller.get_logger().error(f'Exception in main loop: {e}')
        controller.get_logger().error(traceback.format_exc())
        traceback.print_exc(file=sys.stdout)
    finally:
        if enable_logging:
            if logger and not has_saved:
                sys.stderr.write(f"[SAVE] Writing {len(logger.param_buffer)} frames to disk...\n")
                logger.save_episode() # for replicator
                # logger.stop_recording()  # for rosbag
                has_saved = True
                sys.stderr.write("[SAVE] Save complete!\n")
            else:
                sys.stderr.write("[INFO] Logger not initialized or already saved.\n")
            if logger:
                logger.wait_for_pending_saves(timeout=60.0)
        
        sys.stderr.write("[Finally] Shutting down ROS 2 node and simulation....\n")
        if controller is not None:
            controller.destroy_node()
        if raycast_pub is not None:
            raycast_pub.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

        simulation_app.close()


# =================================================================================
if __name__ == "__main__":
    main()
