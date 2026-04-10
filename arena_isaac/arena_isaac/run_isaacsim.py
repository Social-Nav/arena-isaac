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


def main(args=None):
    """
    Main function to initialize the simulation, create the ROS 2 node,
    and run the simulation loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-data', type=str, default='false',
                       help='Enable VLN dataset logging')
    parser.add_argument('--log-level', type=str, default='info',
                       help='log level for IsaacSim (debug/info/warn/error)')
    parsed_args = parser.parse_args(args)

    enable_logging = parsed_args.save_data.lower() == 'true'
    vln_logger_replicator_cls = None
    vln_logger_rosbag_cls = None

    if enable_logging:
        try:
            from vln_dataset_logger_replicator import VLNDataLoggerReplicator
            from vln_dataset_logger_rosbag import VLNDataLoggerRosbag

            vln_logger_replicator_cls = VLNDataLoggerReplicator
            vln_logger_rosbag_cls = VLNDataLoggerRosbag
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
                    if logger is None and frame_step_counter % 50 == 0:

                        if is_prim_path_valid(target_camera_path):
                            try:
                                logger = vln_logger_replicator_cls(camera_prim_path=target_camera_path, pedestrian_root_path = target_pedestrian_root_path, lidar_prim_path=target_lidar_path) # replicator logic
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
                                controller.get_logger().info('✅ VLNDataLoggerRosbag initialized successfully')
                                '''
                            except Exception as e:
                                sys.stderr.write(f"\n VLNDataLogger initialization failed: {e}\n")
                        else:
                            if frame_step_counter % 300 == 0:
                                sys.stderr.write(f" Waiting for robot spawn... searching path: {target_camera_path}")
                    if logger:
                        try:
                            logger.step(step_idx=frame_step_counter) # replicator logic
                            frame_step_counter += 1  # increment after each capture

                        except Exception as e:
                            # Force stack trace to stderr
                            sys.stderr.write(f"\n🔥 Logger step crashed: {e}\n")
                            traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
                            sys.stderr.flush()
            else:
                if was_playing:
                    world.pause()
                    was_playing = False
                simulation_app.update()

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
                pass
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
