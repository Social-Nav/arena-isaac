import xml.etree.ElementTree as ET
import tempfile
import os
import sys
import traceback
from pathlib import Path

import carb
import isaac_utils.graphs.joint_states as joint_states
import isaac_utils.graphs.odom as odom
import isaac_utils.graphs.sensors.sensors as sensors
import isaac_utils.graphs.tf as tf
import omni.kit.commands as commands
import omni.usd
from isaac_utils.graphs import control
from isaac_utils.managers.door_manager import DoorManager
from isaac_utils.managers.elevator_manager import ElevatorManager
from isaac_utils.utils import geom
from isaac_utils.utils.path import world_path
from isaac_utils.utils.prim import ensure_path
from pxr import Usd, UsdPhysics

from isaacsim_msgs.srv import SpawnUrdf

from .utils import Service, on_exception
from typing import Dict

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))


def _find_articulation_root(prim_path: str) -> str | None:
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return None

    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())

    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())

    return None


def _resolve_base_prim_path(prim_path: str, base_frame: str) -> str:
    stage = omni.usd.get_context().get_stage()
    expected_base_path = os.path.join(prim_path, base_frame)
    expected_base_prim = stage.GetPrimAtPath(expected_base_path)
    if expected_base_prim and expected_base_prim.IsValid():
        return expected_base_path

    articulation_prim_path = _find_articulation_root(prim_path)
    if articulation_prim_path:
        carb.log_warn(
            f"[SpawnUrdf] Base prim '{expected_base_path}' not found; "
            f"using articulation root '{articulation_prim_path}' for Isaac graphs"
        )
        return articulation_prim_path

    carb.log_warn(
        f"[SpawnUrdf] Base prim '{expected_base_path}' and articulation root not found; "
        f"falling back to robot prim '{prim_path}' for Isaac graphs"
    )
    return prim_path


def _resolve_mesh_filename(filename: str) -> str:
    placeholder = '/tmp/arena_isaac_placeholder_mesh.dae'
    if not os.path.exists(placeholder):
        with open(placeholder, 'w', encoding='utf-8') as f:
            f.write('''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries>
    <geometry id="tiny_box" name="tiny_box">
      <mesh>
        <source id="tiny_box_positions"><float_array id="tiny_box_positions_array" count="24">-0.01 -0.01 -0.01 0.01 -0.01 -0.01 0.01 0.01 -0.01 -0.01 0.01 -0.01 -0.01 -0.01 0.01 0.01 -0.01 0.01 0.01 0.01 0.01 -0.01 0.01 0.01</float_array><technique_common><accessor source="#tiny_box_positions_array" count="8" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
        <vertices id="tiny_box_vertices"><input semantic="POSITION" source="#tiny_box_positions"/></vertices>
        <triangles count="12"><input semantic="VERTEX" source="#tiny_box_vertices" offset="0"/><p>0 1 2 0 2 3 4 6 5 4 7 6 0 4 5 0 5 1 1 5 6 1 6 2 2 6 7 2 7 3 3 7 4 3 4 0 0 3 7 0 7 4 1 5 6 1 6 2</p></triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes><visual_scene id="Scene"><node id="tiny_box"><instance_geometry url="#tiny_box"/></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
''')
    return placeholder

    if not filename:
        return filename
    if os.path.exists(filename):
        return filename

    package_name = None
    package_rel = None
    marker = 'package:/'
    if marker in filename:
        package_uri = filename[filename.index(marker):]
        package_rel = package_uri[len(marker):].lstrip('/')
        parts = package_rel.split('/', 1)
        if len(parts) == 2:
            package_name, package_rel = parts

    basename = os.path.basename(filename)
    candidates = []
    if package_name and package_rel:
        candidates.extend([
            os.path.join('/opt/ros/jazzy/share', package_name, package_rel),
            os.path.join('/home/ubuntu/arena_jazzy_ws/install/arena_robots/share/arena_robots/robots/turtlebot/urdf', package_rel),
            os.path.join('/home/ubuntu/arena_jazzy_ws/src/Arena/arena_robots/arena_robots/robots/turtlebot/urdf', package_rel),
            os.path.join('/home/ubuntu/arena_jazzy_ws/src/Arena/arena_robots/deps/turtlebot4', package_name, package_rel),
        ])
    candidates.extend([
        os.path.join('/home/ubuntu/arena_jazzy_ws/install/arena_robots/share/arena_robots/robots/turtlebot/urdf/meshes', basename),
        os.path.join('/home/ubuntu/arena_jazzy_ws/src/Arena/arena_robots/arena_robots/robots/turtlebot/urdf/meshes', basename),
        os.path.join('/home/ubuntu/arena_jazzy_ws/src/Arena/arena_robots/deps/turtlebot4/turtlebot4_description/meshes', basename),
    ])

    for candidate in candidates:
        if os.path.exists(candidate):
            carb.log_warn(f"[SpawnUrdf] Resolved mesh '{filename}' -> '{candidate}'")
            return candidate

    fallback = '/home/ubuntu/arena_jazzy_ws/install/arena_robots/share/arena_robots/robots/turtlebot/urdf/meshes/shell.dae'
    if os.path.exists(fallback):
        carb.log_warn(f"[SpawnUrdf] Mesh '{filename}' not found; using fallback '{fallback}'")
        return fallback

    carb.log_warn(f"[SpawnUrdf] Mesh '{filename}' not found and no fallback mesh exists")
    return filename


def sanitize_urdf_for_isaac(urdf_path: str) -> str:
    # usd hates dashes in names, so i hate usd
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_name_map: Dict[str, str] = {}
    joint_name_map: Dict[str, str] = {}

    for tag in root.iter():
        if tag.tag == 'link':
            name = tag.attrib.get('name')
            if name and '-' in name:
                link_name_map[name] = name.replace('-', '_')
        elif tag.tag == 'joint':
            name = tag.attrib.get('name')
            if name and '-' in name:
                joint_name_map[name] = name.replace('-', '_')

    tmp_mesh_dir_path = tempfile.mkdtemp(prefix="isaac_urdf_")

    for tag in root.iter():
        if tag.tag in ['robot', 'link', 'joint']:
            name = tag.attrib.get('name')
            if name and '-' in name:
                tag.attrib['name'] = name.replace('-', '_')
        elif tag.tag in ['parent', 'child']:
            link = tag.attrib.get('link')
            if link in link_name_map:
                tag.attrib['link'] = link_name_map[link]
        elif tag.tag in ['mimic', 'actuator']:
            joint = tag.attrib.get('joint')
            if joint in joint_name_map:
                tag.attrib['joint'] = joint_name_map[joint]
        elif tag.tag == 'gazebo':
            reference = tag.attrib.get('reference')
            if reference in link_name_map:
                tag.attrib['reference'] = link_name_map[reference]

        elif tag.tag == 'mesh':
            original_abs_path = tag.attrib.get('filename')
            if not original_abs_path:
                continue

            original_abs_path = _resolve_mesh_filename(original_abs_path)
            tag.attrib['filename'] = original_abs_path

            filename = os.path.basename(original_abs_path)

            if '-' in filename:
                sanitized_filename = filename.replace('-', '_')
                symlink_path = os.path.join(tmp_mesh_dir_path, sanitized_filename)

                if not os.path.exists(symlink_path):
                    os.symlink(original_abs_path, symlink_path)

                tag.attrib['filename'] = symlink_path

    tmp_urdf = tempfile.NamedTemporaryFile(delete=False, suffix="_sanitized.urdf", mode='w')
    tree.write(tmp_urdf.name, encoding='unicode', xml_declaration=True)

    return tmp_urdf.name


@on_exception('')
def spawn_urdf(request: SpawnUrdf.Request) -> str:
    name = request.name
    urdf_path = request.urdf_path
    robot_model = request.robot_model

    prim_path = world_path(name)

    urdf_path = sanitize_urdf_for_isaac(urdf_path)

    status, import_config = commands.execute("URDFCreateImportConfig")
    import_config.set_merge_fixed_joints(False)
    import_config.set_convex_decomp(False)
    import_config.set_import_inertia_tensor(False)
    import_config.set_make_default_prim(False)
    import_config.set_distance_scale(1.0)
    import_config.set_fix_base(False)
    import_config.set_default_drive_type(2)
    import_config.set_self_collision(False)

    ensure_path(os.path.dirname(prim_path))
    status, usd_path = commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
    )

    if usd_path is None:
        raise ValueError(f"Failed to import URDF from '{urdf_path}'. Status {status}")

    commands.execute(
        "MovePrim",
        path_from=usd_path,
        path_to=prim_path,
        keep_world_transform=True
    )

    try:
        import omni.kit.app
        omni.kit.app.get_app().update()
    except Exception as e:
        carb.log_warn(f"[SpawnUrdf] app.update() after URDF import failed: {e}")

    base_prim_path = _resolve_base_prim_path(prim_path, request.base_frame)
    carb.log_warn(
        f"[SpawnUrdf] Using base/articulation prim '{base_prim_path}' "
        f"for robot '{name}' with TF {request.tf_prefix}/{request.odom_frame} -> "
        f"{request.tf_prefix}/{request.base_frame}"
    )

    # print(usd_path)
    if request.localization:
        try:
            if not odom.odom(
                os.path.join(prim_path, 'odom_publisher'),
                prim_path=base_prim_path,
                base_frame_id=os.path.join(request.tf_prefix, request.base_frame),
                odom_frame_id=os.path.join(request.tf_prefix, request.odom_frame),
                odom_topic=request.odom_topic,
            ):
                carb.log_error("Failed to create odom graph")
        except Exception as e:
            carb.log_error(f"Failed to create odom graph: {e}\n{traceback.format_exc()}")

    if not tf.tf(
        os.path.join(prim_path, 'tf_publisher'),
        prim_path=base_prim_path,
        tf_prefix=request.tf_prefix,
    ):
        carb.log_error("Failed to create tf graph")

    if request.joint_states_topic:
        if not joint_states.joint_states(
            os.path.join(prim_path, 'joint_states_publisher'),
            prim_path=base_prim_path,
            joint_states_topic=request.joint_states_topic,
        ):
            carb.log_error("Failed to create joint_states graph")

    if request.cmd_vel_topic:
        if not control.Control(
            prim_path=prim_path,
            target_prim_path=base_prim_path,
            cmd_vel_topic=request.cmd_vel_topic,
        ).parse(
            robot_model=robot_model,
        ):
            carb.log_error("Failed to create control graph")

    with open(request.urdf_path, 'r') as f:
        sensors.Sensors(
            prim_path=prim_path,
            base_frame=request.tf_prefix,
            base_topic=os.path.dirname(request.cmd_vel_topic),
        ).parse_gazebo(f.read())

    geom.register_robot(
        robot_prim_path=prim_path,
        articulation_prim_path=base_prim_path,
    )

    # Spawn robot at (1, 1, 0)
    geom.move(
        prim_path=prim_path,
        translation=geom.Translation(2.0, 2.0, 0.0),
        rotation=geom.Rotation.parse(request.pose.orientation),
    )

    DoorManager.instance().add_robot(prim_path, request.odom_topic)
    carb.log_error(f"Prim path of robot: {str(prim_path)}")
    carb.log_info(f"Added robot: {prim_path}")
    ElevatorManager.instance().add_robot(prim_path)
    carb.log_error(f"Check robot in ElevatorManager: {str(ElevatorManager.instance().get_robots())}")
    return prim_path


def spawn_urdf_callback(request, response):
    response.path = spawn_urdf(request)
    return response

# Urdf importer service callback.


spawn_urdf_service = Service(
    srv_type=SpawnUrdf,
    srv_name='isaac/SpawnUrdf',
    callback=spawn_urdf_callback
)

__all__ = ['spawn_urdf_service']
