import os
import traceback

import carb
import isaac_utils.graphs.odom as odom
import isaac_utils.graphs.tf as tf
from isaac_utils.managers.door_manager import DoorManager
from isaac_utils.managers.elevator_manager import ElevatorManager
from isaac_utils.utils import geom
from isaac_utils.utils.path import world_path
from isaac_utils.utils.prim import create_prim_safe, ensure_path
from isaacsim_msgs.srv import SpawnUsdRobot
from pxr import Sdf, Usd, UsdPhysics

import omni.usd
import omni.graph.core as og

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


def _find_articulation_root(prim_path: str, base_frame: str = '') -> str | None:
    """Find the physics root prim for an articulation.

    When ArticulationRootAPI is applied to a non-rigid-body Xform (e.g. the
    top-level robot prim), Isaac Sim's Articulation.set_world_poses() may not
    reliably teleport the robot because there is no explicit root body to move.
    In that case, if *base_frame* is provided, the hierarchy is searched for a
    prim whose USD name matches base_frame and that carries RigidBodyAPI.  This
    gives us a concrete physics body (e.g. base_footprint on the chassis) whose
    world pose can be set deterministically.

    Search order:
    1. The first prim whose name equals *base_frame* and has RigidBodyAPI.
       This keeps the published odom child frame aligned with the configured
       robot base frame for composite USD robots.
    2. Any prim that has BOTH ArticulationRootAPI AND RigidBodyAPI.
    3. The first prim with ArticulationRootAPI (original fallback).
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return None

    articulation_xform_path: str | None = None
    articulation_rigid_body_path: str | None = None
    base_frame_path: str | None = None

    for prim in Usd.PrimRange(root_prim):
        has_articulation = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        has_rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)

        if has_articulation and has_rigid_body and articulation_rigid_body_path is None:
            articulation_rigid_body_path = str(prim.GetPath())

        if has_articulation and articulation_xform_path is None:
            articulation_xform_path = str(prim.GetPath())

        if base_frame and base_frame_path is None:
            if prim.GetName() == base_frame and has_rigid_body:
                base_frame_path = str(prim.GetPath())

    if base_frame_path is not None:
        if articulation_rigid_body_path and articulation_rigid_body_path != base_frame_path:
            carb.log_warn(
                f"[SpawnUsdRobot] Using configured base_frame '{base_frame}' rigid body "
                f"for odom/root alignment instead of articulation root body "
                f"{articulation_rigid_body_path}: {base_frame_path}"
            )
        elif articulation_xform_path and articulation_rigid_body_path is None:
            carb.log_warn(
                f"[SpawnUsdRobot] ArticulationRoot is on a non-body prim "
                f"({articulation_xform_path}); using base_frame '{base_frame}' "
                f"rigid body instead: {base_frame_path}"
            )
        return base_frame_path

    if articulation_rigid_body_path is not None:
        return articulation_rigid_body_path

    return articulation_xform_path


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


def _remap_articulation_target(prim_path: str, articulation_prim_path: str):
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
    if not articulation_prim_path or articulation_prim_path == prim_path:
        return

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return

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
            continue

        targets = target_rel.GetForwardedTargets()
        if not targets:
            continue

        old_target = str(targets[0])
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
            f"[SpawnUsdRobot] Remapped {remapped} ArticulationController "
            f"targetPrim(s) under {prim_path}"
        )


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
def spawn_usd_robot(request: SpawnUsdRobot.Request) -> str:
    name = request.name
    usd_path = request.usd_path
    namespace = request.robot_namespace
    base_frame = getattr(request, 'base_frame', '')

    carb.log_warn(f"[SpawnUsdRobot] === Starting spawn: name={name}, ns={namespace} ===")

    prim_path = world_path(name)
    ensure_path(os.path.dirname(prim_path))

    create_prim_safe(
        prim_path=prim_path,
        usd_path=usd_path,
    )
    carb.log_warn(f"[SpawnUsdRobot] USD loaded at {prim_path}")

    try:
        import omni.kit.app
        omni.kit.app.get_app().update()
        carb.log_warn("[SpawnUsdRobot] Triggered app.update() for OmniGraph init")
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] app.update() failed: {e}")

    articulation_prim_path = _find_articulation_root(prim_path, base_frame=base_frame)
    if articulation_prim_path is None:
        carb.log_warn(
            f"[SpawnUsdRobot] No ArticulationRootAPI found, "
            f"falling back to prim_path"
        )
        articulation_prim_path = prim_path
    else:
        carb.log_warn(f"[SpawnUsdRobot] Articulation root at: {articulation_prim_path}")

    try:
        _remap_namespace(prim_path, namespace, base_frame=base_frame)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Remap failed: {e}\n{traceback.format_exc()}")

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

        stage = omni.usd.get_context().get_stage()
        base_prim_path = os.path.join(prim_path, base_frame_id)
        base_prim = stage.GetPrimAtPath(base_prim_path)
        odom_prim_path = base_prim_path if base_prim and base_prim.IsValid() else articulation_prim_path

        odom_topic = f"/{namespace.lstrip('/')}/odom" if namespace else "/odom"

        _disable_odom_graph_tf(prim_path)

        if not odom.odom(
            os.path.join(prim_path, 'odom_publisher'),
            prim_path=odom_prim_path,
            base_frame_id=fq_base_frame,
            odom_frame_id=fq_odom_frame,
            map_frame_id=fq_world_frame,
            odom_topic=odom_topic,
        ):
            carb.log_error('[SpawnUsdRobot] Failed to create odom graph')

    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Graph setup failed: {e}\n{traceback.format_exc()}")

    geom.register_robot(
        robot_prim_path=prim_path,
        articulation_prim_path=articulation_prim_path,
    )

    geom.move(
        prim_path=prim_path,
        translation=geom.Translation.parse(request.pose.position),
        rotation=geom.Rotation.parse(request.pose.orientation),
    )

    odom_topic = f"/{namespace.lstrip('/')}/odom" if namespace else "/odom"
    DoorManager.instance().add_robot(prim_path, odom_topic)
    ElevatorManager.instance().add_robot(prim_path)

    carb.log_warn(f"[SpawnUsdRobot] === Spawn complete: {prim_path} ===")
    return prim_path


def spawn_usd_robot_callback(request, response):
    response.path = spawn_usd_robot(request)
    return response


spawn_usd_robot_service = Service(
    srv_type=SpawnUsdRobot,
    srv_name='isaac/SpawnUsdRobot',
    callback=spawn_usd_robot_callback,
)

__all__ = ['spawn_usd_robot_service']
