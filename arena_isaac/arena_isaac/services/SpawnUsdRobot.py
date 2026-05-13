import os
import traceback

import carb
import numpy as np
from isaac_utils.managers.door_manager import DoorManager

def _safe_import(module_name, obj_name=None):
    try:
        from importlib import import_module
        mod = import_module(module_name)
        if obj_name:
            return getattr(mod, obj_name)
        return mod
    except Exception as e:
        carb.log_error(f"Failed to import {obj_name or module_name} from {module_name}: {e}")
        return None

odom = _safe_import("isaac_utils.graphs.odom")
tf = _safe_import("isaac_utils.graphs.tf")
ElevatorManager = _safe_import("isaac_utils.managers.elevator_manager", "ElevatorManager")
geom = _safe_import("isaac_utils.utils", "geom")
world_path = _safe_import("isaac_utils.utils.path", "world_path")
create_prim_safe = _safe_import("isaac_utils.utils.prim", "create_prim_safe")
ensure_path = _safe_import("isaac_utils.utils.prim", "ensure_path")

from isaacsim_msgs.srv import SpawnUsdRobot as SpawnUsdRobot_srv
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import omni.usd
import omni.graph.core as og

try:
    import omni.replicator.core as rep
    import omni.syntheticdata._syntheticdata as sd
    from isaac_utils.graphs.sensors.camera import SensorCamera, SensorCameraRGBD
except Exception as e:
    rep = None
    sd = None
    SensorCamera = None
    SensorCameraRGBD = None
    carb.log_error(f"[SpawnUsdRobot] Failed to import camera publishing helpers: {e}")

from .utils import Service, on_exception

FRAME_KEYWORDS = {
    'frameid', 'parentframeid', 'childframeid',
    'odomframeid', 'chassisframeid',
}
BASE_FRAME_DEFAULT = 'base_link'
BASE_FRAME_ATTRS = {
    'childframeid',
    'chassisframeid',
    'baseframeid',
    'frameid',
}
TF_DETECT_KEYWORDS = {'parentframeid', 'childframeid'}
NAMESPACE_KEYWORDS = {'nodenamespace'}


def _find_articulation_root(prim_path: str) -> str | None:
    """Return the best articulation target under ``prim_path``.

    Some USD robots, including Ai2_Bot2, author ArticulationRootAPI on the
    top-level robot prim and RigidBodyAPI on child links.  Binding control to an
    arbitrary child rigid body in that case can target an arm/gripper link
    instead of the articulation.  Prefer the top-level articulation root when it
    exists, then any articulation root that is also a rigid body, then any
    authored articulation root.

    Returns None if no ArticulationRootAPI exists under prim_path.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return None

    if root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return str(root_prim.GetPath())

    first_articulation_root = None
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
        if first_articulation_root is None and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            first_articulation_root = str(prim.GetPath())

    return first_articulation_root


def _find_base_frame_prim(prim_path: str, base_frame: str) -> str | None:
    """Find the real chassis/base prim under a referenced USD robot.

    Humanoid USD robots often place ``base_footprint`` below a component prim
    (for example ``Ai2_Bot2_Chassis/base_footprint``), not directly below the
    top-level robot prim.  The odometry graph must bind to that moving rigid
    body, while articulation control can still target the top-level
    ArticulationRootAPI prim.
    """
    if not base_frame:
        return None

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return None

    direct_path = os.path.join(prim_path, base_frame)
    direct_prim = stage.GetPrimAtPath(direct_path)
    if direct_prim and direct_prim.IsValid():
        carb.log_warn(f"[SpawnUsdRobot] Selected direct base frame prim for odom: {direct_path}")
        return direct_path

    first_match = None
    first_rigid_match = None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() != base_frame:
            continue
        path = str(prim.GetPath())
        if first_match is None:
            first_match = path
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            first_rigid_match = path
            break

    selected = first_rigid_match or first_match
    if selected:
        rigid_note = 'rigid body' if selected == first_rigid_match else 'non-rigid name match'
        carb.log_warn(f"[SpawnUsdRobot] Selected nested {rigid_note} base frame prim for odom: {selected}")
    else:
        carb.log_warn(
            f"[SpawnUsdRobot] Could not find base frame prim named '{base_frame}' under {prim_path}; "
            "odom graph will fall back to articulation root"
        )
    return selected


def _find_spawn_body_fallback(prim_path: str, base_frame: str = '') -> str:
    """Return a usable body/prim path when a USD lacks the expected PhysX APIs.

    Some robot USDs in the eval setup are useful for camera/render publishing but
    do not expose a prim that has both ArticulationRootAPI and RigidBodyAPI after
    being referenced under /World/Robots.  In that case we should not abort the
    whole spawn service: keep the prim in the stage, publish cameras, and let the
    task_generator fallback odom/scan state keep Nav2 alive.
    """
    stage = omni.usd.get_context().get_stage()
    if base_frame:
        candidate = os.path.join(prim_path, base_frame)
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            return candidate
    root_prim = stage.GetPrimAtPath(prim_path)
    if root_prim and root_prim.IsValid():
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
    return prim_path




def _has_odom_publisher(prim_path: str, odom_topic: str) -> bool:
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return False

    for prim in Usd.PrimRange(root_prim):
        prim_type = prim.GetTypeName()
        if 'OmniGraph' not in prim_type:
            continue

        prim_path_str = str(prim.GetPath())
        for attr in prim.GetAttributes():
            name_lower = attr.GetName().split(':')[-1].lower()
            if name_lower not in {'topicname', 'topic'}:
                continue

            val = attr.Get()
            if val is None:
                val = _get_og_attr_value(prim_path_str, attr.GetName())
            if val is None:
                continue

            val_str = str(val).strip()
            if odom_topic and val_str == odom_topic:
                return True
            if val_str.endswith('/odom') or val_str == 'odom' or val_str.endswith('odom'):
                return True

    return False


def _disable_odom_publishers(prim_path: str, odom_topic: str) -> int:
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return 0

    disabled = 0
    for prim in Usd.PrimRange(root_prim):
        prim_type = prim.GetTypeName()
        if 'OmniGraph' not in prim_type:
            continue

        prim_path_str = str(prim.GetPath())
        for attr in prim.GetAttributes():
            name_lower = attr.GetName().split(':')[-1].lower()
            if name_lower not in {'topicname', 'topic'}:
                continue

            val = attr.Get()
            if val is None:
                val = _get_og_attr_value(prim_path_str, attr.GetName())
            if val is None:
                continue

            val_str = str(val).strip()
            if not val_str:
                continue

            is_odom = False
            if odom_topic and val_str == odom_topic:
                is_odom = True
            elif val_str.endswith('/odom') or val_str == 'odom' or val_str.endswith('odom'):
                is_odom = True

            if is_odom:
                try:
                    attr.Set(f"{val_str}__disabled")
                    carb.log_warn(
                        f"[SpawnUsdRobot] Disabled odom publisher topic {val_str} on {prim_path_str}"
                    )
                    disabled += 1
                except Exception as e:
                    carb.log_warn(
                        f"[SpawnUsdRobot] Failed to disable odom topic on {prim_path_str}: {e}"
                    )

    return disabled


TF_PUBLISHER_TYPES = {
    'isaacsim.ros2.bridge.ROS2PublishRawTransformTree',
    'isaacsim.ros2.bridge.ROS2PublishTransformTree',
}

ODOM_PUBLISHER_TYPES = {
    'isaacsim.ros2.bridge.ROS2PublishOdometry',
}


def _disable_odom_graph_tf(prim_path: str) -> int:
    """Disable TF and odom publishers from USD-embedded OmniGraph nodes.

    The arena framework creates its own odom OmniGraph via odom.py which
    publishes the same TF frames (world→odom, odom→base_footprint).
    If the USD also contains an ROS_Odometry graph with TF publisher nodes,
    both will publish conflicting transforms for the same frames, causing
    TF flickering and laser scan distortion.

    This function disables all ROS2PublishRawTransformTree and
    ROS2PublishOdometry nodes found under prim_path by setting their
    topicName to a dead topic.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return 0

    disabled = 0
    target_types = TF_PUBLISHER_TYPES | ODOM_PUBLISHER_TYPES
    for prim in Usd.PrimRange(root_prim):
        if 'OmniGraph' not in prim.GetTypeName():
            continue

        node_type_attr = prim.GetAttribute('node:type')
        if not node_type_attr or not node_type_attr.IsValid():
            continue
        node_type = str(node_type_attr.Get() or '')
        if node_type not in target_types:
            continue

        # Disable by renaming topicName to a dead topic
        topic_attr = prim.GetAttribute('inputs:topicName')
        if topic_attr and topic_attr.IsValid():
            old_val = topic_attr.Get()
            if old_val and not str(old_val).endswith('__disabled'):
                topic_attr.Set(f"{old_val}__disabled")
                carb.log_warn(
                    f"[SpawnUsdRobot] Disabled {node_type.split('.')[-1]} "
                    f"topic '{old_val}' on {prim.GetPath()}"
                )
                disabled += 1
                continue

        # For TF publishers that publish to /tf by default (no topicName attr),
        # create and set a dead topicName
        try:
            if not topic_attr or not topic_attr.IsValid():
                prim.CreateAttribute('inputs:topicName', Sdf.ValueTypeNames.String).Set('/tf__disabled')
            else:
                topic_attr.Set('/tf__disabled')
            carb.log_warn(
                f"[SpawnUsdRobot] Disabled {node_type.split('.')[-1]} "
                f"(no topicName) on {prim.GetPath()}"
            )
            disabled += 1
        except Exception as e:
            carb.log_warn(
                f"[SpawnUsdRobot] Failed to disable {node_type.split('.')[-1]} "
                f"on {prim.GetPath()}: {e}"
            )

    if disabled:
        carb.log_warn(
            f"[SpawnUsdRobot] Disabled {disabled} USD-embedded TF/odom "
            f"publisher(s) under {prim_path}"
        )
    return disabled


def _get_og_attr_value(prim_path_str: str, attr_name: str):
    """Read attribute value via OmniGraph API (for unauthored defaults)."""
    try:
        node = og.get_node_by_path(prim_path_str)
        if node and node.is_valid():
            og_attr = node.get_attribute(attr_name)
            if og_attr and og_attr.is_valid():
                val = og_attr.get()
                carb.log_warn(
                    f"[SpawnUsdRobot] OG fallback {attr_name}: "
                    f"{repr(val)} on {prim_path_str}"
                )
                return val
    except Exception as e:
        carb.log_warn(
            f"[SpawnUsdRobot] OG fallback failed for {attr_name}: {e}"
        )
    return None


def _replace_base_frame(value: str, base_frame: str) -> str:
    if not base_frame or base_frame == BASE_FRAME_DEFAULT:
        return value
    parts = value.split('/')
    if parts and parts[-1] == BASE_FRAME_DEFAULT:
        parts[-1] = base_frame
        return '/'.join(parts)
    return value


ARTICULATION_CONTROLLER_TYPES = {
    'isaacsim.core.nodes.IsaacArticulationController',
    'isaacsim.ros2.bridge.ROS2PublishJointState',
    'isaacsim.ros2.bridge.ROS2SubscribeJointState',
}

TWIST_SUBSCRIBER_TYPES = {
    'isaacsim.ros2.bridge.ROS2SubscribeTwist',
}


def _set_or_create_input_attr(prim, attr_name: str, value, value_type=Sdf.ValueTypeNames.String) -> bool:
    attr = prim.GetAttribute(attr_name)
    try:
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(attr_name, value_type)
        attr.Set(value)
        return True
    except Exception as e:
        carb.log_warn(
            f"[SpawnUsdRobot] Failed to set {attr_name}='{value}' "
            f"on {prim.GetPath()}: {e}"
        )
        return False


def _remap_cmd_vel_bridge(prim_path: str, namespace: str) -> tuple[int, list[str]]:
    """Point embedded ROS2SubscribeTwist nodes at the robot cmd_vel topic.

    USD-only robots often ship with a ready-made control OmniGraph, but it may
    subscribe to a root-level or author-time topic.  Nav2 publishes final
    velocity commands in the task-generator robot namespace, so make every
    embedded Twist subscriber consume ``/<namespace>/cmd_vel``.  When the node
    exposes ``nodeNamespace`` we keep ``topicName`` relative (``cmd_vel``);
    otherwise use the absolute topic as a safe fallback.
    """
    if not namespace:
        carb.log_warn('[SpawnUsdRobot] No namespace provided; skipping cmd_vel bridge remap')
        return 0, []

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        carb.log_warn(f'[SpawnUsdRobot] Cannot remap cmd_vel bridge: prim {prim_path} not found')
        return 0, []

    ns = namespace.strip('/')
    absolute_cmd_vel = f'/{ns}/cmd_vel'
    remapped = 0
    subscriptions: list[str] = []

    for prim in Usd.PrimRange(root_prim):
        if 'OmniGraph' not in prim.GetTypeName():
            continue
        node_type_attr = prim.GetAttribute('node:type')
        node_type = str(node_type_attr.Get() or '') if node_type_attr and node_type_attr.IsValid() else ''
        if node_type not in TWIST_SUBSCRIBER_TYPES:
            continue

        has_namespace_attr = False
        node_namespace_attr = prim.GetAttribute('inputs:nodeNamespace')
        if node_namespace_attr and node_namespace_attr.IsValid():
            has_namespace_attr = _set_or_create_input_attr(prim, 'inputs:nodeNamespace', ns)

        topic_value = 'cmd_vel' if has_namespace_attr else absolute_cmd_vel
        topic_set = _set_or_create_input_attr(prim, 'inputs:topicName', topic_value)
        if topic_set:
            remapped += 1
            subscription = os.path.join('/', ns, topic_value) if topic_value == 'cmd_vel' else topic_value
            subscriptions.append(subscription)
            carb.log_warn(
                f"[SpawnUsdRobot] Remapped ROS2SubscribeTwist on {prim.GetPath()} "
                f"to namespace='{ns if has_namespace_attr else '<absolute>'}', topic='{topic_value}' "
                f"(effective {subscription})"
            )

    if not remapped:
        carb.log_warn(
            f"[SpawnUsdRobot] No embedded ROS2SubscribeTwist node found under {prim_path}; "
            f"USD must already provide another controller or robot will not consume {absolute_cmd_vel}"
        )
    return remapped, subscriptions


def _remap_articulation_target(prim_path: str, articulation_prim_path: str) -> int:
    """Rewrite IsaacArticulationController targetPrim relationships to point
    at the actual articulation root body instead of the top-level robot Xform.

    USD robots authored in Isaac Sim Composer often set
    ``IsaacArticulationController.inputs:targetPrim`` to the top-level Xform
    (e.g. ``/Ai2_Bot2``).  When the ArticulationRootAPI lives on a deeper body
    prim (e.g. ``base_footprint``), PhysX cannot match the top-level path and
    logs ``Pattern … did not match any articulations`` every frame.

    This function rewrites those relationships at spawn time so that
    controllers point at the prim that actually carries ArticulationRootAPI.
    """
    if not articulation_prim_path:
        return 0

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return 0

    remapped = 0
    for prim in Usd.PrimRange(root_prim):
        prim_type = prim.GetTypeName()
        if 'OmniGraph' not in prim_type:
            continue

        # Check if this is an ArticulationController node
        node_type_attr = prim.GetAttribute('node:type')
        if not node_type_attr or not node_type_attr.IsValid():
            continue
        node_type = node_type_attr.Get()
        if node_type not in ARTICULATION_CONTROLLER_TYPES:
            continue

        # Rewrite inputs:targetPrim relationship
        target_rel = prim.GetRelationship('inputs:targetPrim')
        if not target_rel or not target_rel.IsValid():
            try:
                target_rel = prim.CreateRelationship('inputs:targetPrim')
            except Exception as e:
                carb.log_warn(
                    f"[SpawnUsdRobot] Failed to create targetPrim relationship "
                    f"on {prim.GetPath()}: {e}"
                )
                continue

        targets = target_rel.GetForwardedTargets()
        old_target = str(targets[0]) if targets else '<unset>'
        if old_target == articulation_prim_path:
            continue  # already correct

        target_rel.ClearTargets(removeSpec=False)
        target_rel.AddTarget(articulation_prim_path)
        carb.log_warn(
            f"[SpawnUsdRobot] Remapped ArticulationController targetPrim: "
            f"'{old_target}' -> '{articulation_prim_path}' on {prim.GetPath()}"
        )
        remapped += 1

    if remapped:
        carb.log_warn(
            f"[SpawnUsdRobot] Remapped {remapped} articulation/joint target(s) "
            f"under {prim_path} to {articulation_prim_path}"
        )
    return remapped


def _log_motion_bridge_diagnostics(
    prim_path: str,
    namespace: str,
    articulation_prim_path: str,
    odom_prim_path: str,
    cmd_vel_subscriptions: list[str],
):
    cmd_vel_topic = f"/{namespace.strip('/')}/cmd_vel" if namespace else '/cmd_vel'
    odom_topic = f"/{namespace.strip('/')}/odom" if namespace else '/odom'
    subscriptions = cmd_vel_subscriptions or ['<no ROS2SubscribeTwist remapped>']
    carb.log_warn(
        f"[SpawnUsdRobot] Motion bridge diagnostics: robot_prim={prim_path}, "
        f"namespace=/{namespace.strip('/') if namespace else ''}, cmd_vel={cmd_vel_topic}, "
        f"embedded_twist_subscriptions={subscriptions}, "
        f"articulation_target={articulation_prim_path}, odom_target={odom_prim_path}, odom_topic={odom_topic}"
    )


def _find_preferred_camera_prim(prim_path: str) -> str | None:
    """Find a usable Camera prim in an imported robot USD.

    Prefer a head/front camera for VLN ego-video observations, but fall back
    to the first authored camera so USD robots without URDF gazebo plugins can
    still publish real ROS image topics.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return None

    cameras: list[str] = []
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() == 'Camera':
            cameras.append(str(prim.GetPath()))

    if not cameras:
        return None

    preferences = ('head_camera', 'front_camera', 'chassis_camera', 'camera')
    for key in preferences:
        for camera_path in cameras:
            if key in camera_path.lower():
                return camera_path
    return cameras[0]


def _ensure_fallback_camera_prim(prim_path: str, base_frame: str | None = None) -> str | None:
    stage = omni.usd.get_context().get_stage()
    base_frame_id = base_frame or BASE_FRAME_DEFAULT
    base_prim_path = os.path.join(prim_path, base_frame_id)
    base_prim = stage.GetPrimAtPath(base_prim_path)
    parent_path = base_prim_path if base_prim and base_prim.IsValid() else prim_path
    camera_path = os.path.join(parent_path, 'vln_head_camera')
    try:
        camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
        xform = UsdGeom.XformCommonAPI(camera)
        xform.SetTranslate(Gf.Vec3d(0.35, 0.0, 0.75))
        # USD cameras look down local -Z.  This orientation makes the optical
        # axis point forward along the robot base frame while keeping image up
        # close to +Z, so the ego video is a real robot head-camera render rather
        # than a ceiling/floor diagnostic view.
        xform.SetRotate(Gf.Vec3f(90.0, 0.0, -90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
        carb.log_warn(f'[SpawnUsdRobot] Created fallback VLN Camera prim at {camera_path}')
        return camera_path
    except Exception as e:
        carb.log_error(f'[SpawnUsdRobot] Failed to create fallback camera: {e}\n{traceback.format_exc()}')
        return None


def _publish_rgbd_camera(prim_path: str, namespace: str, base_frame: str | None, pose=None):
    """Attach ROS2 image/depth/camera-info writers to an existing USD Camera.

    Most USD robots carry camera prims but no ROS2 image OmniGraph.  The dual
    VLN evaluator needs real frames, so create a render product for the chosen
    camera and publish it under /<namespace>/head_camera/*.
    """
    if rep is None or SensorCamera is None or SensorCameraRGBD is None:
        carb.log_error('[SpawnUsdRobot] Camera publishing helpers unavailable')
        return

    camera_prim_path = _find_preferred_camera_prim(prim_path)
    if not camera_prim_path:
        camera_prim_path = _ensure_fallback_camera_prim(prim_path, base_frame)
    if not camera_prim_path:
        carb.log_warn(f'[SpawnUsdRobot] No usable Camera prim under {prim_path}; no RGBD topics published')
        return

    try:
        camera_topic = f"/{namespace.strip('/')}/head_camera" if namespace else '/head_camera'
        frame_namespace = namespace.split('/')[-1] if namespace else ''
        base_frame_id = base_frame or BASE_FRAME_DEFAULT
        frame = os.path.join(frame_namespace, base_frame_id, 'head_camera') if frame_namespace else os.path.join(base_frame_id, 'head_camera')
        step_size = 6  # 60Hz sim graph / 10Hz image output
        render_product = rep.create.render_product(camera_prim_path, (640, 480))
        render_product_path = getattr(render_product, 'path', None) or str(render_product)
        SensorCamera._publish_camera_info(render_product_path, frame, '', 1, camera_topic, step_size)
        SensorCamera._publish_rgb(render_product_path, frame, '', 1, camera_topic, step_size)
        SensorCameraRGBD._publish_depth(render_product_path, frame, '', 1, camera_topic, step_size)
        carb.log_warn(
            f"[SpawnUsdRobot] Publishing RGBD camera {camera_prim_path} on {camera_topic}/{{image,depth,camera_info}}"
        )

        safe_name = prim_path.strip('/').replace('/', '_')
        top_down_camera_path = f'/World/vln_top_down_camera_{safe_name}'
        stage = omni.usd.get_context().get_stage()
        try:
            top_down_camera = UsdGeom.Camera.Define(stage, Sdf.Path(top_down_camera_path))
            xform = UsdGeom.XformCommonAPI(top_down_camera)
            top_x = float(getattr(getattr(pose, 'position', None), 'x', 0.0)) if pose is not None else 0.0
            top_y = float(getattr(getattr(pose, 'position', None), 'y', 0.0)) if pose is not None else 0.0
            xform.SetTranslate(Gf.Vec3d(top_x, top_y, 8.0))
            # USD cameras look down local -Z.  Identity rotation gives a stable
            # floating top-down world view from above the spawned robot.
            xform.SetRotate(Gf.Vec3f(0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
            top_down_camera.CreateFocalLengthAttr(12.0)
            top_down_camera.CreateHorizontalApertureAttr(35.0)
            top_down_camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
            top_render_product = rep.create.render_product(top_down_camera_path, (640, 640))
            top_render_product_path = getattr(top_render_product, 'path', None) or str(top_render_product)
            top_topic = f"/{namespace.strip('/')}/top_down_camera" if namespace else '/top_down_camera'
            top_frame = os.path.join(frame_namespace, base_frame_id, 'top_down_camera') if frame_namespace else os.path.join(base_frame_id, 'top_down_camera')
            SensorCamera._publish_camera_info(top_render_product_path, top_frame, '', 1, top_topic, step_size)
            SensorCamera._publish_rgb(top_render_product_path, top_frame, '', 1, top_topic, step_size)
            carb.log_warn(
            f"[SpawnUsdRobot] Publishing top-down camera {top_down_camera_path} at ({top_x:.2f}, {top_y:.2f}, 8.00) on {top_topic}/{{image,camera_info}}"
            )
        except Exception as top_exc:
            carb.log_error(f"[SpawnUsdRobot] Failed to publish top-down camera: {top_exc}\n{traceback.format_exc()}")
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Failed to publish RGBD camera: {e}\n{traceback.format_exc()}")

def _remap_namespace(prim_path: str, namespace: str, base_frame: str | None = None):
    """Traverse USD stage prims under prim_path and modify OmniGraph node
    attributes directly via the USD API."""
    if not namespace:
        carb.log_error(f"[SpawnUsdRobot] No namespace provided, skipping remap")
        return

    frame_namespace = namespace.split('/')[-1]

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        carb.log_error(f"[SpawnUsdRobot] Cannot remap: prim {prim_path} not found")
        return

    remapped_count = 0

    for prim in Usd.PrimRange(root_prim):
        prim_type = prim.GetTypeName()
        if 'OmniGraph' not in prim_type:
            continue

        prim_path_str = str(prim.GetPath())
        attrs = prim.GetAttributes()

        attr_names_lower = {
            a.GetName().split(':')[-1].lower() for a in attrs
        }
        is_tf_node = TF_DETECT_KEYWORDS.issubset(attr_names_lower)

        for attr in attrs:
            attr_name = attr.GetName()
            name_lower = attr_name.split(':')[-1].lower()

            if name_lower in NAMESPACE_KEYWORDS:
                if is_tf_node:
                    carb.log_warn(
                        f"[SpawnUsdRobot] Skipping nodeNamespace on TF node "
                        f"(has parentFrameId+childFrameId): {prim_path_str}"
                    )
                    continue
                try:
                    attr.Set(namespace)
                    carb.log_warn(
                        f"[SpawnUsdRobot] Set {attr_name}='{namespace}' "
                        f"on {prim_path_str}"
                    )
                    remapped_count += 1
                except Exception as e:
                    carb.log_warn(
                        f"[SpawnUsdRobot] Failed to set {attr_name} "
                        f"on {prim_path_str}: {e}"
                    )
                continue

            if name_lower in FRAME_KEYWORDS:
                try:
                    old_val = attr.Get()
                    if old_val is None:
                        old_val = _get_og_attr_value(prim_path_str, attr_name)
                    old_str = str(old_val).strip() if old_val is not None else ''
                    if not old_str:
                        carb.log_warn(
                            f"[SpawnUsdRobot] Skipping {attr_name} on "
                            f"{prim_path_str}: no value from USD or OG API"
                        )
                        continue
                    if base_frame and name_lower in BASE_FRAME_ATTRS:
                        old_str = _replace_base_frame(old_str, base_frame)
                    new_val = f"{frame_namespace}/{old_str}"
                    attr.Set(new_val)
                    carb.log_warn(
                        f"[SpawnUsdRobot] Remapped {attr_name}: "
                        f"'{old_str}' -> '{new_val}' on {prim_path_str}"
                    )
                    remapped_count += 1
                except Exception as e:
                    carb.log_warn(
                        f"[SpawnUsdRobot] Failed to remap {attr_name} "
                        f"on {prim_path_str}: {e}"
                    )

    carb.log_warn(
        f"[SpawnUsdRobot] Namespace remap complete: "
        f"{remapped_count} attributes updated under {prim_path}"
    )


@on_exception('')
def spawn_usd_robot(request: SpawnUsdRobot_srv.Request) -> str:
    name = request.name
    usd_path = request.usd_path
    
    # We overloaded the 'model' field to pass namespace and base_frame since
    # SpawnUsd has different fields than the missing SpawnUsdRobot
    namespace = request.robot_namespace
    base_frame = request.base_frame

    carb.log_warn(f"[SpawnUsdRobot] === Starting spawn: name={name}, ns={namespace} ===")

    prim_path = world_path(name)
    ensure_path(os.path.dirname(prim_path))

    stage = omni.usd.get_context().get_stage()
    existing_prim = stage.GetPrimAtPath(prim_path) if stage else None
    if existing_prim and existing_prim.IsValid():
        carb.log_warn(
            f"[SpawnUsdRobot] Existing prim at {prim_path}; removing it before "
            "re-spawn so camera render products and ROS writers are rebuilt."
        )
        try:
            stage.RemovePrim(Sdf.Path(prim_path))
            try:
                from omni.kit.app import get_app
                get_app().update()
            except Exception:
                pass
        except Exception as e:
            carb.log_error(f"[SpawnUsdRobot] Failed to remove existing prim {prim_path}: {e}\n{traceback.format_exc()}")

    spawn_position = np.array(geom.Translation.parse(request.pose.position).tuple(), dtype=float)
    spawn_orientation = np.array(geom.Rotation.parse(request.pose.orientation).quat(), dtype=float)
    create_prim_safe(
        prim_path=prim_path,
        usd_path=usd_path,
        position=spawn_position,
        orientation=spawn_orientation,
    )
    carb.log_warn(
        f"[SpawnUsdRobot] USD loaded at {prim_path} pose="
        f"({spawn_position[0]:.3f}, {spawn_position[1]:.3f}, {spawn_position[2]:.3f})"
    )

    try:
        from omni.kit.app import get_app
        get_app().update()
        carb.log_warn("[SpawnUsdRobot] Triggered app.update() for OmniGraph init")
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] app.update() failed: {e}")

    articulation_prim_path = _find_articulation_root(prim_path)
    if articulation_prim_path is None:
        articulation_prim_path = _find_spawn_body_fallback(prim_path, base_frame)
        carb.log_warn(
            f"[SpawnUsdRobot] No prim with both ArticulationRootAPI and RigidBodyAPI "
            f"found under {prim_path}; continuing with fallback body {articulation_prim_path}. "
            f"Odom will be provided by task_generator fallback state if needed."
        )
    else:
        carb.log_warn(f"[SpawnUsdRobot] Articulation root at: {articulation_prim_path}")

    try:
        _remap_namespace(prim_path, namespace, base_frame=base_frame)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Remap failed: {e}\n{traceback.format_exc()}")

    cmd_vel_subscriptions: list[str] = []
    try:
        _, cmd_vel_subscriptions = _remap_cmd_vel_bridge(prim_path, namespace)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] cmd_vel bridge remap failed: {e}\n{traceback.format_exc()}")

    try:
        _remap_articulation_target(prim_path, articulation_prim_path)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Articulation target remap failed: {e}\n{traceback.format_exc()}")

    try:
        base_frame_id = base_frame or BASE_FRAME_DEFAULT
        frame_namespace = namespace.split('/')[-1] if namespace else ''
        fq_base_frame = os.path.join(frame_namespace, base_frame_id) if frame_namespace else base_frame_id
        fq_odom_frame = os.path.join(frame_namespace, 'odom') if frame_namespace else 'odom'
        fq_world_frame = os.path.join(frame_namespace, 'world') if frame_namespace else 'world'

        odom_prim_path = _find_base_frame_prim(prim_path, base_frame_id) or articulation_prim_path

        odom_topic = f"/{namespace.lstrip('/')}/odom" if namespace else "/odom"

        _disable_odom_graph_tf(prim_path)

        disable_isaac_odom_graph = os.environ.get('ARENA_DISABLE_ISAAC_ODOM_GRAPH', '1').strip().lower() in {
            '1', 'true', 'yes', 'on'
        }
        odom_created = False
        if disable_isaac_odom_graph:
            carb.log_warn(
                '[SpawnUsdRobot] Skipping Isaac odom/TF graph because '
                'ARENA_DISABLE_ISAAC_ODOM_GRAPH is enabled; task_generator publishes the single TF/odom source.'
            )
            odom_created = True
        elif odom is not None and odom_prim_path:
            odom_created = odom.odom(
            os.path.join(prim_path, 'odom_publisher'),
            prim_path=odom_prim_path,
            base_frame_id=fq_base_frame,
            odom_frame_id=fq_odom_frame,
            map_frame_id='map',
            odom_topic=odom_topic,
            )
        if not odom_created:
            carb.log_error('[SpawnUsdRobot] Failed to create odom graph')

        _log_motion_bridge_diagnostics(
            prim_path=prim_path,
            namespace=namespace,
            articulation_prim_path=articulation_prim_path,
            odom_prim_path=odom_prim_path,
            cmd_vel_subscriptions=cmd_vel_subscriptions,
        )

    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Graph setup failed: {e}\n{traceback.format_exc()}")

    try:
        _publish_rgbd_camera(prim_path, namespace, base_frame, pose=request.pose)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] RGBD camera setup failed: {e}\n{traceback.format_exc()}")

    try:
        geom.register_robot(
            robot_prim_path=prim_path,
            articulation_prim_path=articulation_prim_path,
        )
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Robot registration failed: {e}\n{traceback.format_exc()}")

    geom.move(
        prim_path=prim_path,
        translation=geom.Translation.parse(request.pose.position),
        rotation=geom.Rotation.parse(request.pose.orientation),
    )

    odom_topic = f"/{namespace.lstrip('/')}/odom" if namespace else "/odom"
    try:
        DoorManager.instance().add_robot(prim_path, odom_topic)
        ElevatorManager.instance().add_robot(prim_path)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Door/elevator registration failed: {e}\n{traceback.format_exc()}")

    carb.log_warn(f"[SpawnUsdRobot] === Spawn complete: {prim_path} ===")
    return prim_path


def spawn_usd_robot_callback(request, response):
    spawn_usd_robot(request)
    # Isaac Sim's embedded Python/rclpy stack currently aborts in the generated
    # SpawnUsdRobot response converter. The caller has an explicit timeout path;
    # suppress the response after setup so the simulator process survives.
    raise RuntimeError('SpawnUsdRobot response intentionally suppressed after spawn setup')


spawn_usd_robot_service = Service(
    srv_type=SpawnUsdRobot_srv,
    srv_name='isaac/SpawnUsdRobot_srv',
    callback=spawn_usd_robot_callback,
)

__all__ = ['spawn_usd_robot_service']
