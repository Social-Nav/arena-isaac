# fmt: off


# preload attrs
import argparse
import os
import arena_simulation_setup
import arena_simulation_setup.utils.cattrs

# Use the isaacsim to import SimulationApp
from isaacsim import SimulationApp


def _env_flag(name: str, default: bool = False) -> bool:
    return str(os.environ.get(name, '1' if default else '0')).strip().lower() in {'1', 'true', 'yes', 'on'}


# Setting the config for simulation and make an simulation.
_headless = _env_flag('ARENA_ISAAC_HEADLESS', False)
CONFIG = {
    #"renderer": "Wireframe",
    "renderer": os.environ.get('ARENA_ISAAC_RENDERER', 'RayTracedLighting'),
    "headless": _headless,
    # Headless eval still needs real render-product updates for VLN RGB/depth
    # observations.  Disabling viewport updates leaves Replicator annotators
    # registered but starved of image data.
    "disable_viewport_updates": _env_flag('ARENA_ISAAC_DISABLE_VIEWPORT_UPDATES', False),
    "sync_loads": _env_flag('ARENA_ISAAC_SYNC_LOADS', True),
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
import time
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
from pxr import Gf, Sdf, UsdGeom

try:
    from isaacsim.core.world import World
except ImportError:
    from omni.isaac.core import World

try:
    from isaacsim.core.api.simulation_context import SimulationContext
except ImportError:
    from omni.isaac.core import SimulationContext

from isaac_utils.utils import geom as isaac_geom

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
import builtin_interfaces.msg
import geometry_msgs.msg
import nav_msgs.msg
import rosgraph_msgs.msg
import sensor_msgs.msg
import tf2_msgs.msg

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


def _camera_follow_target_from_name(camera_prim) -> str | None:
    attr = camera_prim.GetAttribute('arena:followPrimPath')
    if attr and attr.HasValue():
        value = str(attr.Get() or '').strip()
        if value:
            return value
    prefix = 'vln_top_down_camera_'
    name = camera_prim.GetName()
    if not name.startswith(prefix):
        return None
    # Backward-compatible fallback for cameras created before the follow target
    # metadata existed.  Current robot path is /World/Robots/<robot_name>.
    safe = name[len(prefix):]
    if safe.startswith('World_Robots_'):
        return '/World/Robots/' + safe[len('World_Robots_'):]
    return None


def _follow_vln_top_down_cameras() -> None:
    if not _env_flag('ARENA_ISAAC_TOP_DOWN_CAMERA_FOLLOW_ROBOT', True):
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    absolute_height = os.environ.get('ARENA_SPAWN_USD_ROBOT_TOP_DOWN_CAMERA_HEIGHT')
    default_relative_height = float(os.environ.get('ARENA_SPAWN_USD_ROBOT_TOP_DOWN_CAMERA_RELATIVE_HEIGHT', '8.0'))
    world_prim = stage.GetPrimAtPath('/World')
    camera_prims = world_prim.GetChildren() if world_prim and world_prim.IsValid() else []
    for camera_prim in camera_prims:
        if camera_prim.GetTypeName() != 'Camera' or not camera_prim.GetName().startswith('vln_top_down_camera_'):
            continue
        target_path = _camera_follow_target_from_name(camera_prim)
        if not target_path:
            continue
        target_translation = isaac_geom.get_world_translation(target_path)
        if target_translation is None:
            continue
        relative_height = default_relative_height
        rel_attr = camera_prim.GetAttribute('arena:followRelativeHeight')
        if rel_attr and rel_attr.HasValue():
            try:
                relative_height = float(rel_attr.Get())
            except Exception:
                relative_height = default_relative_height
        base_z = None
        base_z_attr = camera_prim.GetAttribute('arena:followBaseZ')
        if base_z_attr and base_z_attr.HasValue():
            try:
                base_z = float(base_z_attr.Get())
            except Exception:
                base_z = None
        z_reference = base_z if base_z is not None else float(target_translation.z)
        z = float(absolute_height) if absolute_height is not None else z_reference + relative_height
        xform = UsdGeom.Xformable(camera_prim)
        translate_attr = camera_prim.GetAttribute('xformOp:translate')
        if not translate_attr or not translate_attr.IsValid():
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(float(target_translation.x), float(target_translation.y), z))
            xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            continue
        translate_attr.Set(Gf.Vec3d(float(target_translation.x), float(target_translation.y), z))
        orient_attr = camera_prim.GetAttribute('xformOp:orient')
        if orient_attr and orient_attr.IsValid():
            orient_attr.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

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
from isaacsim_msgs.msg import PedestrianGoal as IsaacPedestrianGoal

from arena_isaac.services.NavigatePedestrians import navigate_pedestrian as _drive_visible_pedestrian


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
        self._latest_visible_pedestrian_goals: list[IsaacPedestrianGoal] = []
        self._subscriber = self.create_subscription(
            Pedestrians, peds_topic, self._peds_callback, 10
        )
        self._physx_query = omni.physx.get_physx_scene_query_interface()
        self.get_logger().info(
            f'RaycastObstaclePublisher: peds={peds_topic}, obs={obstacles_topic}'
        )

    
    def _peds_callback(self, msg: Pedestrians):
        drive_enabled_value = os.environ.get(
            'ARENA_ISAAC_DRIVE_PEDESTRIANS_FROM_ARENA_PEDS',
            os.environ.get('ARENA_ISAAC_SYNC_PEDESTRIANS_FROM_ARENA_PEDS', '1'),
        )
        drive_enabled = str(drive_enabled_value).strip().lower() not in {
            '0',
            'false',
            'no',
            'off',
        }
        if drive_enabled:
            latest_goals = []
            for ped in msg.pedestrians:
                try:
                    goal = IsaacPedestrianGoal()
                    goal.name = f'Pedestrians/{ped.name}'
                    goal.position = ped.pose.position
                    goal.velocity = float(_math.hypot(ped.twist.linear.x, ped.twist.linear.y))
                    latest_goals.append(goal)
                except Exception as exc:
                    carb.log_warn(f'[run_isaacsim] failed to cache visible pedestrian {ped.name}: {exc}')
            self._latest_visible_pedestrian_goals = latest_goals

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

    def drive_visible_pedestrians(self) -> None:
        for goal in list(self._latest_visible_pedestrian_goals):
            try:
                _drive_visible_pedestrian(goal)
            except Exception as exc:
                carb.log_warn(f'[run_isaacsim] failed to drive visible pedestrian {goal.name}: {exc}')


class ManualStepStatePublisher:
    """Publishes core ROS state from the Python step loop.

    Headless eval can advance Isaac through ``world.step()`` without firing
    OnPlaybackTick-driven OmniGraph ROS publishers.  This publisher keeps the
    task-generator readiness path tied to the actual Isaac step loop.
    """

    def __init__(self, node: rclpy.node.Node):
        self._node = node
        self._clock_pub = node.create_publisher(rosgraph_msgs.msg.Clock, '/clock', 10)
        self._tf_pub = node.create_publisher(tf2_msgs.msg.TFMessage, '/tf', 10)
        self._odom_pubs: dict[str, object] = {}
        self._last_pose: dict[str, tuple[float, isaac_geom.Translation, isaac_geom.Rotation]] = {}
        self._last_robot_paths: set[str] = set()

    @staticmethod
    def _stamp(sim_time: float):
        stamp = builtin_interfaces.msg.Time()
        stamp.sec = int(sim_time)
        stamp.nanosec = int((sim_time - stamp.sec) * 1_000_000_000)
        return stamp

    @staticmethod
    def _namespace_from_robot_path(robot_path: str) -> str:
        name = robot_path.rstrip('/').split('/')[-1]
        return f'task_generator_node/{name}' if name else ''

    @staticmethod
    def _odom_topic(namespace: str) -> str:
        return f'/{namespace}/odom' if namespace else '/odom'

    @staticmethod
    def _frames(namespace: str) -> tuple[str, str, str]:
        robot_name = namespace.rstrip('/').split('/')[-1] if namespace else ''
        if robot_name:
            return f'{robot_name}/world', f'{robot_name}/odom', f'{robot_name}/base_link'
        return 'world', 'odom', 'base_link'

    def _publisher_for(self, topic: str):
        pub = self._odom_pubs.get(topic)
        if pub is None:
            pub = self._node.create_publisher(nav_msgs.msg.Odometry, topic, 10)
            self._odom_pubs[topic] = pub
            carb.log_warn(f'[run_isaacsim] Manual odom publisher active on {topic}')
        return pub

    def publish(self, sim_time: float) -> None:
        stamp = self._stamp(max(sim_time, 0.0))

        clock_msg = rosgraph_msgs.msg.Clock()
        clock_msg.clock = stamp
        self._clock_pub.publish(clock_msg)

        current_robot_paths = set(isaac_geom.registered_robots().keys())
        new_robot_paths = current_robot_paths - self._last_robot_paths
        for robot_path in sorted(new_robot_paths):
            carb.log_warn(
                f'[run_isaacsim] Manual step state publisher tracking {robot_path} '
                f'-> {isaac_geom.registered_robots().get(robot_path)}'
            )
        self._last_robot_paths = current_robot_paths

        for robot_path in sorted(current_robot_paths):
            namespace = self._namespace_from_robot_path(robot_path)
            world_frame, odom_frame, base_frame = self._frames(namespace)
            odom_topic = self._odom_topic(namespace)
            pose = isaac_geom.get_world_pose(robot_path)
            if pose is None:
                continue
            translation, rotation = pose
            last = self._last_pose.get(robot_path)

            vx = vy = vz = wx = wy = wz = 0.0
            if last is not None:
                last_time, last_translation, last_rotation = last
                dt = sim_time - last_time
                if dt > 1e-9:
                    vx = (translation.x - last_translation.x) / dt
                    vy = (translation.y - last_translation.y) / dt
                    vz = (translation.z - last_translation.z) / dt
                    prev_yaw = last_rotation.euler('z')[0]
                    yaw = rotation.euler('z')[0]
                    dyaw = (yaw - prev_yaw + _math.pi) % (2.0 * _math.pi) - _math.pi
                    wz = dyaw / dt

            self._last_pose[robot_path] = (sim_time, translation, rotation)

            odom_msg = nav_msgs.msg.Odometry()
            odom_msg.header.stamp = stamp
            odom_msg.header.frame_id = odom_frame
            odom_msg.child_frame_id = base_frame
            odom_msg.pose.pose.position.x = translation.x
            odom_msg.pose.pose.position.y = translation.y
            odom_msg.pose.pose.position.z = translation.z
            odom_msg.pose.pose.orientation.w = rotation.w
            odom_msg.pose.pose.orientation.x = rotation.x
            odom_msg.pose.pose.orientation.y = rotation.y
            odom_msg.pose.pose.orientation.z = rotation.z
            odom_msg.twist.twist.linear.x = vx
            odom_msg.twist.twist.linear.y = vy
            odom_msg.twist.twist.linear.z = vz
            odom_msg.twist.twist.angular.x = wx
            odom_msg.twist.twist.angular.y = wy
            odom_msg.twist.twist.angular.z = wz
            self._publisher_for(odom_topic).publish(odom_msg)

            map_tf = geometry_msgs.msg.TransformStamped()
            map_tf.header.stamp = stamp
            map_tf.header.frame_id = world_frame
            map_tf.child_frame_id = odom_frame
            map_tf.transform.rotation.w = 1.0

            odom_tf = geometry_msgs.msg.TransformStamped()
            odom_tf.header.stamp = stamp
            odom_tf.header.frame_id = odom_frame
            odom_tf.child_frame_id = base_frame
            odom_tf.transform.translation.x = translation.x
            odom_tf.transform.translation.y = translation.y
            odom_tf.transform.translation.z = translation.z
            odom_tf.transform.rotation.w = rotation.w
            odom_tf.transform.rotation.x = rotation.x
            odom_tf.transform.rotation.y = rotation.y
            odom_tf.transform.rotation.z = rotation.z
            self._tf_pub.publish(tf2_msgs.msg.TFMessage(transforms=[map_tf, odom_tf]))


class ManualReplicatorCameraPublisher:
    """Publishes real Isaac camera annotator frames through rclpy."""

    def __init__(self, node: rclpy.node.Node):
        self._node = node
        self._cameras: dict[str, dict] = {}

    @staticmethod
    def _stamp(sim_time: float):
        stamp = builtin_interfaces.msg.Time()
        stamp.sec = int(sim_time)
        stamp.nanosec = int((sim_time - stamp.sec) * 1_000_000_000)
        return stamp

    @staticmethod
    def _image_msg(array: np.ndarray, *, stamp, frame_id: str, encoding: str) -> sensor_msgs.msg.Image:
        contiguous = np.ascontiguousarray(array)
        msg = sensor_msgs.msg.Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(contiguous.shape[0])
        msg.width = int(contiguous.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = int(msg.width * (3 if encoding == 'rgb8' else 4))
        msg.data = contiguous.tobytes()
        return msg

    @staticmethod
    def _camera_info_msg(params, *, stamp, frame_id: str, width: int, height: int) -> sensor_msgs.msg.CameraInfo:
        msg = sensor_msgs.msg.CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.width = int(width)
        msg.height = int(height)
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0] * 5
        fx = fy = 500.0
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        try:
            if isinstance(params, dict):
                resolution = params.get('renderProductResolution')
                if resolution is not None and len(resolution) >= 2:
                    msg.width = int(resolution[0])
                    msg.height = int(resolution[1])
                focal = params.get('cameraFocalLength')
                aperture = params.get('cameraAperture')
                if focal is not None and aperture is not None:
                    focal_x = float(focal[0] if hasattr(focal, '__len__') else focal)
                    focal_y = float(focal[1] if hasattr(focal, '__len__') and len(focal) > 1 else focal_x)
                    aperture_x = float(aperture[0] if hasattr(aperture, '__len__') else aperture)
                    aperture_y = float(aperture[1] if hasattr(aperture, '__len__') and len(aperture) > 1 else aperture_x)
                    if aperture_x:
                        fx = msg.width * focal_x / aperture_x
                    if aperture_y:
                        fy = msg.height * focal_y / aperture_y
                cx = float(msg.width) / 2.0
                cy = float(msg.height) / 2.0
        except Exception:
            pass
        msg.k = [float(fx), 0.0, float(cx), 0.0, float(fy), float(cy), 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [float(fx), 0.0, float(cx), 0.0, 0.0, float(fy), float(cy), 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    @staticmethod
    def _extract_array(data):
        if data is None:
            return None
        if isinstance(data, dict):
            for key in ('data', 'rgb', 'image', 'depth'):
                if key in data:
                    arr = ManualReplicatorCameraPublisher._extract_array(data[key])
                    if arr is not None:
                        return arr
            for value in data.values():
                arr = ManualReplicatorCameraPublisher._extract_array(value)
                if arr is not None:
                    return arr
            return None
        if isinstance(data, (tuple, list)):
            for value in data:
                arr = ManualReplicatorCameraPublisher._extract_array(value)
                if arr is not None:
                    return arr
            return None
        arr = np.asarray(data)
        if arr.ndim >= 2 and arr.size > 0:
            return arr
        return None

    @staticmethod
    def _extract_rgb(data):
        arr = ManualReplicatorCameraPublisher._extract_array(data)
        if arr is None:
            return None
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _extract_depth(data):
        arr = ManualReplicatorCameraPublisher._extract_array(data)
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        arr[~np.isfinite(arr)] = 0.0
        return arr

    def _ensure_camera(self, spec: dict) -> dict | None:
        key = str(spec.get('key') or spec.get('camera_prim_path') or '')
        if not key:
            return None
        state = self._cameras.get(key)
        if state is not None:
            return state
        try:
            resolution = tuple(int(v) for v in spec.get('resolution') or [640, 480])
            render_product = rep.create.render_product(str(spec['camera_prim_path']), resolution)
            rgb_annot = rep.AnnotatorRegistry.get_annotator('rgb')
            params_annot = rep.AnnotatorRegistry.get_annotator('camera_params')
            rgb_annot.attach(render_product)
            params_annot.attach(render_product)
            depth_annot = None
            depth_pub = None
            topic_base = str(spec['topic_base'])
            if bool(spec.get('publish_depth')):
                depth_annot = rep.AnnotatorRegistry.get_annotator('distance_to_camera')
                depth_annot.attach(render_product)
                depth_pub = self._node.create_publisher(sensor_msgs.msg.Image, f'{topic_base}/depth', 10)
            rgb_topic_name = str(spec.get('rgb_topic_name') or 'image').strip('/') or 'image'
            state = {
                'render_product': render_product,
                'rgb_annot': rgb_annot,
                'depth_annot': depth_annot,
                'params_annot': params_annot,
                'rgb_pub': self._node.create_publisher(sensor_msgs.msg.Image, f'{topic_base}/{rgb_topic_name}', 10),
                'depth_pub': depth_pub,
                'info_pub': self._node.create_publisher(sensor_msgs.msg.CameraInfo, f'{topic_base}/camera_info', 10),
                'frame': str(spec['frame']),
                'warned_empty': False,
            }
            self._cameras[key] = state
            carb.log_warn(
                f"[run_isaacsim] Manual Replicator camera publisher active for {spec['camera_prim_path']} "
                f"on {topic_base}/*"
            )
            return state
        except Exception as exc:
            carb.log_warn(f'[run_isaacsim] Failed to initialize manual camera publisher for {spec}: {exc}')
            return None

    def publish(self, sim_time: float) -> None:
        try:
            from arena_isaac.services.SpawnUsdRobot import get_vln_camera_publish_specs
        except Exception:
            return
        stamp = self._stamp(max(sim_time, 0.0))
        for spec in get_vln_camera_publish_specs():
            state = self._ensure_camera(spec)
            if state is None:
                continue
            try:
                rgb = self._extract_rgb(state['rgb_annot'].get_data())
                params = state['params_annot'].get_data()
                if rgb is None:
                    if not state['warned_empty']:
                        carb.log_warn(f"[run_isaacsim] Manual camera publisher waiting for RGB data from {spec['camera_prim_path']}")
                        state['warned_empty'] = True
                    continue
                state['rgb_pub'].publish(self._image_msg(rgb, stamp=stamp, frame_id=state['frame'], encoding='rgb8'))
                height, width = int(rgb.shape[0]), int(rgb.shape[1])
                state['info_pub'].publish(
                    self._camera_info_msg(params, stamp=stamp, frame_id=state['frame'], width=width, height=height)
                )
                if state['depth_annot'] is not None and state['depth_pub'] is not None:
                    depth = self._extract_depth(state['depth_annot'].get_data())
                    if depth is not None:
                        state['depth_pub'].publish(
                            self._image_msg(depth, stamp=stamp, frame_id=state['frame'], encoding='32FC1')
                        )
            except Exception as exc:
                carb.log_warn(f"[run_isaacsim] Manual camera publish failed for {spec.get('key')}: {exc}")


# =================================================================================

# ===================================controller====================================
# create controller node for isaacsim.


class IsaacController(rclpy.node.Node):
    def __init__(self, *args, **kwargs):
        super().__init__(node_name="isaac", *args, **kwargs)
        self._running = False
        self._should_step_once = False

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

    def _cb_pause(self, request: std_srvs.srv.Trigger.Request, response: std_srvs.srv.Trigger.Response):
        self._running = False
        response.success = True
        return response

    def _cb_unpause(self, request: std_srvs.srv.Trigger.Request, response: std_srvs.srv.Trigger.Response):
        self._running = True
        response.success = True
        return response

    def _cb_step(self, request: std_srvs.srv.Trigger.Request, response: std_srvs.srv.Trigger.Response):
        self._should_step_once = True
        response.success = True
        return response

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
from vln_dataset_logger_replicator import VLNDataLoggerReplicator
import signal


def _resolve_vln_dataset_logger_prim_paths(robot_model: str) -> tuple[str, str, str] | None:
    """Read VLN logger USD prim paths from arena_robots/robots/<model>/model_params.yaml.

    Expected YAML block (optional): ``vln_dataset_logger`` with keys:
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

    vln = data.get('vln_dataset_logger')
    if not isinstance(vln, dict):
        return None

    stage = vln.get('robot_stage_name')
    cam_suffix = vln.get('camera_prim_suffix')
    lidar_suffix = vln.get('lidar_prim_suffix')
    if not stage or not cam_suffix or not lidar_suffix:
        return None

    root = os.path.join('/World', 'Robots', str(stage))
    lidar_path = os.path.join(root, str(lidar_suffix))
    camera_path = os.path.join(root, str(cam_suffix))
    ped_root = str(vln.get('pedestrian_root_path') or '/World/Pedestrians')
    return lidar_path, camera_path, ped_root


def main(args=None):
    """
    Main function to initialize the simulation, create the ROS 2 node,
    and run the simulation loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-data', type=str, default='false',
                       help='Enable VLN dataset logging')
    parser.add_argument('--robot', type=str, default='jackal',
                       help='Robot name used in task_generator topic namespace')
    parser.add_argument('--log-level', type=str, default='info',
                       help='log level for IsaacSim (debug/info/warn/error)')
    parsed_args = parser.parse_args(args)

    enable_logging = parsed_args.save_data.lower() == 'true'
    robot_name = parsed_args.robot
    vln_logger_replicator_cls = None
    # vln_logger_rosbag_cls = None

    vln_paths = _resolve_vln_dataset_logger_prim_paths(robot_name)
    if vln_paths is None:
        target_lidar_path = ""
        target_camera_path = ""
        target_pedestrian_root_path = "/World/Pedestrians"
    else:
        target_lidar_path, target_camera_path, target_pedestrian_root_path = vln_paths

    if enable_logging and vln_paths is None:
        enable_logging = False
        sys.stderr.write(
            "[WARN] VLN logging disabled: missing or incomplete "
            f"'vln_dataset_logger' in arena_robots/robots/{robot_name}/model_params.yaml "
            "(need robot_stage_name, camera_prim_suffix, lidar_prim_suffix).\n"
        )
        sys.stderr.flush()
    
    # VLN logger state
    logger = None
    frame_step_counter = 0
    has_saved = False
    collecting = False   # True only while robot is actively executing a task
    
    if enable_logging:
        try:
            from vln_dataset_logger_replicator import VLNDataLoggerReplicator
            # from vln_dataset_logger_rosbag import VLNDataLoggerRosbag

            vln_logger_replicator_cls = VLNDataLoggerReplicator
            # vln_logger_rosbag_cls = VLNDataLoggerRosbag
            sys.stderr.write(
                "[INFO] VLN logging modules loaded successfully.\n"
            )
            sys.stderr.flush()

        except Exception as e:
            enable_logging = False
            sys.stderr.write(
                f"[WARN] VLN logging disabled because optional dependencies are unavailable: {e}\n"
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
    manual_step_state_pub = ManualStepStatePublisher(controller)
    manual_camera_pub = ManualReplicatorCameraPublisher(controller)

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
            # Save leftover buffer if robot was interrupted mid-task
            if len(logger.param_buffer) > 0:
                sys.stderr.write(f"[TaskReset #{episode_num}] Saving interrupted episode ({len(logger.param_buffer)} frames)...\n")
                try:
                    logger.save_episode()
                except Exception as e:
                    sys.stderr.write(f"[TaskReset #{episode_num}] Save failed: {e}\n")
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
                sys.stderr.write(f"[NavStatus {label}] Save complete.\n")
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
    initial_world_reset = str(
        os.environ.get('ARENA_ISAAC_INITIAL_WORLD_RESET', '0')
    ).strip().lower() not in {'0', 'false', 'no', 'off'}
    if initial_world_reset:
        carb.log_warn('[run_isaacsim] Running synchronous world.reset() before ROS service loop')
        world.reset()
    else:
        carb.log_warn(
            '[run_isaacsim] Skipping synchronous world.reset() before ROS service loop; '
            'the main loop will advance Isaac with simulation_app.update()/world.step()'
        )

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
    render_preset = os.environ.get('RENDER_PRESET')
    if render_preset is None and str(os.environ.get('ARENA_ISAAC_HEADLESS', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}:
        render_preset = 'boring'
    if (render_preset or 'photoreal') != 'boring':
        photoreal.PRESET_PHOTOREAL.apply()
    else:
        carb.log_warn('[run_isaacsim] Skipping viewport render preset actions for boring/headless mode')

    # Avoid synchronous timeline operations before the ROS service loop in
    # headless eval; Isaac/Kit can block here and prevent service callbacks from
    # being processed.  The main loop controls play/pause as episodes run.
    initial_timeline_stop = str(
        os.environ.get('ARENA_ISAAC_INITIAL_TIMELINE_STOP', '0')
    ).strip().lower() not in {'0', 'false', 'no', 'off'}
    if initial_timeline_stop:
        carb.log_warn('[run_isaacsim] Stopping timeline before main loop')
        omni.timeline.get_timeline_interface().stop()
    else:
        carb.log_warn('[run_isaacsim] Skipping synchronous timeline.stop() before ROS service loop')

    # mainloop
    was_playing: bool = False
    rclpy_spin_timeout_sec = max(
        float(os.environ.get('ARENA_ISAAC_RCLPY_SPIN_TIMEOUT_SEC', '0.001')),
        0.0,
    )
    render_every_n_steps = max(
        int(os.environ.get('ARENA_ISAAC_RENDER_EVERY_N_STEPS', '6' if str(os.environ.get('ARENA_ISAAC_HEADLESS', '0')).strip().lower() in {'1', 'true', 'yes', 'on'} else '1')),
        1,
    )
    # Do not force the very first post-unpause step to render in headless eval.
    # GRScenes + multiple Replicator render products can spend several minutes
    # compiling RTX/material state on that first rendered step.  If it happens
    # before the first physics-only tick, /clock never advances and the task
    # generator remains at eval_ready=reset_started even though Isaac is busy,
    # which looks like a deadlock.  Let the first cheap physics tick publish
    # /clock, then render on the normal cadence below.
    initial_force_render_steps = max(
        int(os.environ.get('ARENA_ISAAC_INITIAL_FORCE_RENDER_STEPS', '0')),
        0,
    )
    default_initial_no_render_steps = '300' if str(os.environ.get('ARENA_ISAAC_HEADLESS', '0')).strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    } else '0'
    initial_no_render_steps = max(
        int(os.environ.get('ARENA_ISAAC_INITIAL_NO_RENDER_STEPS', default_initial_no_render_steps)),
        0,
    )
    pre_play_render_warmup_frames = max(
        int(os.environ.get('ARENA_ISAAC_PRE_PLAY_RENDER_WARMUP_FRAMES', '0')),
        0,
    )
    pre_play_render_warmup_method = str(
        os.environ.get('ARENA_ISAAC_PRE_PLAY_RENDER_WARMUP_METHOD', 'simulation_app_update')
    ).strip().lower()
    initial_no_render_method = str(
        os.environ.get('ARENA_ISAAC_INITIAL_NO_RENDER_METHOD', 'world_step')
    ).strip().lower()
    default_play_method = 'timeline' if str(os.environ.get('ARENA_ISAAC_HEADLESS', '0')).strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    } else 'world'
    play_method = str(os.environ.get('ARENA_ISAAC_PLAY_METHOD', default_play_method)).strip().lower()
    sim_step_counter = 0
    pre_play_warmup_pending = False
    last_idle_heartbeat = time.monotonic()
    try:
        while simulation_app.is_running():
            stepped_this_iteration: bool = False
            rclpy.spin_once(controller, timeout_sec=rclpy_spin_timeout_sec)
            rclpy.spin_once(raycast_pub, timeout_sec=0)
            if controller.running:
                if not was_playing:
                    if not pre_play_warmup_pending:
                        pre_play_warmup_pending = True
                        carb.log_warn(
                            '[run_isaacsim] Simulation unpaused; deferring play/step by one main-loop turn '
                            'so the ROS service response can flush before any expensive render work'
                        )
                        rclpy.spin_once(controller, timeout_sec=0)
                        continue

                    if pre_play_render_warmup_frames > 0:
                        warmup_started = time.monotonic()
                        carb.log_warn(
                            f'[run_isaacsim] Running {pre_play_render_warmup_frames} paused render warmup frame(s) '
                            'before world.play(); this initializes heavy GRScenes/HuNav/RTX assets without advancing sim time'
                        )
                        for warmup_idx in range(pre_play_render_warmup_frames):
                            frame_started = time.monotonic()
                            try:
                                if pre_play_render_warmup_method in {'rep', 'rep_orchestrator', 'orchestrator'}:
                                    rep.orchestrator.step(delta_time=0.0, rt_subframes=1)
                                else:
                                    # Keep the timeline/world stopped while giving Kit/RTX/asset loading a render/update
                                    # opportunity.  In heavy headless GRScenes, rep.orchestrator.step(delta_time=0.0)
                                    # can block long enough that the eval-side sim-tick gate times out before world.play().
                                    simulation_app.update()
                            except Exception as exc:
                                carb.log_warn(
                                    f'[run_isaacsim] rep.orchestrator paused warmup failed on frame {warmup_idx + 1}: {exc}; '
                                    'falling back to simulation_app.update()'
                                )
                                simulation_app.update()
                            frame_elapsed = time.monotonic() - frame_started
                            if frame_elapsed >= 5.0:
                                carb.log_warn(
                                    f'[run_isaacsim] paused render warmup frame {warmup_idx + 1}/'
                                    f'{pre_play_render_warmup_frames} took {frame_elapsed:.3f}s'
                                )
                        carb.log_warn(
                            f'[run_isaacsim] Paused render warmup completed in {time.monotonic() - warmup_started:.3f}s'
                        )

                    play_started = time.monotonic()
                    if play_method in {'timeline', 'timeline_play'}:
                        carb.log_warn('[run_isaacsim] Calling timeline.play()')
                        omni.timeline.get_timeline_interface().play()
                        carb.log_warn(
                            f'[run_isaacsim] timeline.play() returned in {time.monotonic() - play_started:.3f}s'
                        )
                    else:
                        carb.log_warn('[run_isaacsim] Calling world.play()')
                        world.play()
                        carb.log_warn(
                            f'[run_isaacsim] world.play() returned in {time.monotonic() - play_started:.3f}s'
                        )
                    was_playing = True
                    pre_play_warmup_pending = False
                    sim_step_counter = 0
                    continue
                door_manager.update()
                elevator_manager.update()
                sim_step_counter += 1
                should_render = (
                    sim_step_counter <= initial_force_render_steps
                    or (
                        sim_step_counter > initial_no_render_steps
                        and (sim_step_counter % render_every_n_steps) == 0
                    )
                )
                step_started = time.monotonic()
                if sim_step_counter <= 5 or (sim_step_counter % 100) == 0:
                    carb.log_warn(
                        f'[run_isaacsim] stepping sim_step_counter={sim_step_counter}, '
                        f'render={should_render}, initial_no_render_method={initial_no_render_method}'
                    )
                if (
                    not should_render
                    and sim_step_counter <= initial_no_render_steps
                    and initial_no_render_method in {'simulation_app_update', 'app_update', 'kit_update'}
                ):
                    carb.log_warn(
                        '[run_isaacsim] ARENA_ISAAC_INITIAL_NO_RENDER_METHOD=simulation_app_update '
                        'does not advance Isaac physics/ROS clock; prefer world_step for eval runs'
                    )
                    simulation_app.update()
                else:
                    if should_render:
                        raycast_pub.drive_visible_pedestrians()
                    world.step(render=should_render)
                raycast_pub.drive_visible_pedestrians()
                _follow_vln_top_down_cameras()
                try:
                    sim_time = float(SimulationContext.instance().current_time)
                except Exception:
                    sim_time = sim_step_counter * float(
                        os.environ.get('ARENA_ISAAC_MANUAL_CLOCK_DT_SEC', '0.0166666667')
                    )
                manual_step_state_pub.publish(sim_time)
                if should_render:
                    manual_camera_pub.publish(sim_time)
                step_elapsed = time.monotonic() - step_started
                if step_elapsed >= 5.0:
                    carb.log_warn(
                        f'[run_isaacsim] simulation step render={should_render} took {step_elapsed:.3f}s '
                        f'(render_every_n_steps={render_every_n_steps}, '
                        f'initial_no_render_steps={initial_no_render_steps}, '
                        f'initial_force_render_steps={initial_force_render_steps}, '
                        f'initial_no_render_method={initial_no_render_method}, '
                        f'sim_step_counter={sim_step_counter})'
                    )
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
                                    _output_dir = str(_p / 'collected_data')
                                    logger = vln_logger_replicator_cls(
                                        camera_prim_path=target_camera_path,
                                        pedestrian_root_path=target_pedestrian_root_path,
                                        lidar_prim_path=target_lidar_path,
                                        output_dir=_output_dir,
                                    )
                                    sys.stderr.write(f"\n [Logger] Output dir: {_output_dir}\n")
                                    '''
                                    logger = vln_logger_rosbag_cls(
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
                                    sys.stderr.write(f"\n VLNDataLogger initialization failed: {e}\n")
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
                    if play_method in {'timeline', 'timeline_play'}:
                        omni.timeline.get_timeline_interface().pause()
                    else:
                        world.pause()
                    was_playing = False
                pre_play_warmup_pending = False
                idle_update = str(
                    os.environ.get('ARENA_ISAAC_IDLE_SIMULATION_APP_UPDATE', '0')
                ).strip().lower() not in {'0', 'false', 'no', 'off'}
                if idle_update:
                    simulation_app.update()
                else:
                    time.sleep(0.001)
                now = time.monotonic()
                if now - last_idle_heartbeat >= 5.0:
                    carb.log_warn('[run_isaacsim] ROS service loop heartbeat while simulation is paused')
                    last_idle_heartbeat = now

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
