import os
import traceback

import carb
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
diff_drive_graph = _safe_import("isaac_utils.graphs.control.differential", "differential")
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
    import omni.syntheticdata
    import omni.replicator.core as rep
    import omni.syntheticdata._syntheticdata as sd
    from isaacsim.ros2.bridge import read_camera_info
except Exception:
    rep = None
    sd = None
    read_camera_info = None

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
    """Return the PhysX articulation root body for a spawned robot.

    Some third-party USD robots are authored with more than one prim carrying
    ArticulationRootAPI.  Ai2_Bot2 is one such asset: the real full-body root is
    ``base_link`` while the nested chassis ``base_footprint`` also has
    ArticulationRootAPI.  Returning the first traversal hit can target only the
    nested chassis and make reset/controller commands fight PhysX's active
    articulation.  Prefer an explicit ``base_link`` root when present, then fall
    back to the first valid root for other robots.

    Returns None if no such prim exists under prim_path.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return None

    candidates: list[str] = []
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            candidates.append(str(prim.GetPath()))

    if not candidates:
        return None

    for candidate in candidates:
        if os.path.basename(candidate) == 'base_link':
            if len(candidates) > 1:
                carb.log_warn(
                    f"[SpawnUsdRobot] Multiple articulation roots under {prim_path}: {candidates}; "
                    f"preferring full-body root {candidate}"
                )
            return candidate

    if len(candidates) > 1:
        carb.log_warn(
            f"[SpawnUsdRobot] Multiple articulation roots under {prim_path}: {candidates}; "
            f"falling back to first traversal candidate {candidates[0]}"
        )
    return candidates[0]


def _remove_extra_articulation_roots(prim_path: str, preferred_root_path: str) -> int:
    """Remove duplicate ArticulationRootAPI schemas under a robot at runtime.

    Ai2_Bot2 currently ships with both ``base_link`` and the fixed child
    ``Ai2_Bot2_Chassis/base_footprint`` marked as articulation roots.  PhysX can
    then split the robot into separate articulations: the body/head can fall over
    while the newly-created diff-drive controller targets the wrong root.  We do
    not modify the source USD; this only cleans the composed stage instance.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid() or not preferred_root_path:
        return 0

    removed = 0
    for prim in Usd.PrimRange(root_prim):
        if str(prim.GetPath()) == preferred_root_path:
            continue
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        try:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            carb.log_warn(
                f"[SpawnUsdRobot] Removed duplicate ArticulationRootAPI from {prim.GetPath()} "
                f"while keeping {preferred_root_path}"
            )
            removed += 1
        except Exception as e:
            carb.log_warn(
                f"[SpawnUsdRobot] Failed to remove duplicate ArticulationRootAPI from {prim.GetPath()}: {e}"
            )
    return removed




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


def _disable_embedded_twist_subscribers(prim_path: str) -> int:
    """Disable USD-authored cmd_vel subscribers before adding Arena's graph.

    Ai2_Bot2 ships with a USD-embedded differential controller graph.  In Arena
    evals that graph has an unauthored/empty ``topicName`` on its
    ROS2SubscribeTwist node, so it does not reliably consume the namespaced
    ``/<namespace>/cmd_vel`` commands and can also race with an Arena-created
    controller.  Disable those embedded twist subscribers and install one
    explicit graph that subscribes to the fully-qualified command topic.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return 0

    disabled = 0
    twist_graph_paths: set[str] = set()
    for prim in Usd.PrimRange(root_prim):
        if 'OmniGraph' not in prim.GetTypeName():
            continue

        node_type_attr = prim.GetAttribute('node:type')
        if not node_type_attr or not node_type_attr.IsValid():
            continue
        node_type = str(node_type_attr.Get() or '')
        if node_type not in TWIST_SUBSCRIBER_TYPES:
            continue

        twist_graph_paths.add(str(prim.GetParent().GetPath()))

        topic_attr = prim.GetAttribute('inputs:topicName')
        try:
            if not topic_attr or not topic_attr.IsValid():
                topic_attr = prim.CreateAttribute('inputs:topicName', Sdf.ValueTypeNames.String)
            old_val = topic_attr.Get()
            topic_attr.Set('/__arena_disabled_embedded_cmd_vel')
            carb.log_warn(
                f"[SpawnUsdRobot] Disabled embedded ROS2SubscribeTwist "
                f"topic '{old_val}' on {prim.GetPath()}"
            )
            disabled += 1
        except Exception as e:
            carb.log_warn(
                f"[SpawnUsdRobot] Failed to disable embedded ROS2SubscribeTwist "
                f"on {prim.GetPath()}: {e}"
            )

    # USD-authored IsaacArticulationController nodes in the same graph as the
    # embedded ROS2SubscribeTwist can keep executing with stale/zero commands and
    # overwrite Arena's explicit controller every tick.  Do not blanket-disable
    # unrelated controllers: Ai2_Bot2 also has a ROS_JointStates graph whose
    # controller targets base_link and is not part of cmd_vel handling.
    for prim in Usd.PrimRange(root_prim):
        if 'OmniGraph' not in prim.GetTypeName():
            continue
        parent_graph_path = str(prim.GetParent().GetPath())
        if parent_graph_path not in twist_graph_paths:
            continue
        node_type_attr = prim.GetAttribute('node:type')
        if not node_type_attr or not node_type_attr.IsValid():
            continue
        node_type = str(node_type_attr.Get() or '')
        if node_type != 'isaacsim.core.nodes.IsaacArticulationController':
            continue
        target_rel = prim.GetRelationship('inputs:targetPrim')
        if not target_rel or not target_rel.IsValid():
            continue
        try:
            old_targets = [str(t) for t in target_rel.GetForwardedTargets()]
            target_rel.ClearTargets(removeSpec=False)
            carb.log_warn(
                f"[SpawnUsdRobot] Disabled embedded ArticulationController "
                f"targets {old_targets} on {prim.GetPath()}"
            )
            disabled += 1
        except Exception as e:
            carb.log_warn(
                f"[SpawnUsdRobot] Failed to disable embedded ArticulationController "
                f"on {prim.GetPath()}: {e}"
            )

    return disabled


def _setup_ai2_bot2_control_graph(prim_path: str, articulation_prim_path: str, namespace: str) -> bool:
    """Create the explicit Ai2_Bot2 diff-drive controller used by Isaac evals."""
    if diff_drive_graph is None:
        carb.log_error('[SpawnUsdRobot] Cannot create Ai2_Bot2 control graph: differential graph helper unavailable')
        return False
    if not namespace:
        carb.log_error('[SpawnUsdRobot] Cannot create Ai2_Bot2 control graph: namespace is empty')
        return False

    cmd_vel_topic = f"/{namespace.lstrip('/')}/cmd_vel"
    graph_path = os.path.join(prim_path, 'arena_ai2_bot2_diff_drive_controller')
    _disable_embedded_twist_subscribers(prim_path)

    ok = diff_drive_graph(
        graph_path=graph_path,
        prim_path=articulation_prim_path,
        cmd_vel_topic=cmd_vel_topic,
        joint_names=['driving_left_joint', 'driving_right_joint'],
        wheel_distance=0.416,
        wheel_radius=0.085,
        max_linear_speed=0.65,
        min_linear_speed=-0.65,
        max_angular_speed=1.5,
        min_angular_speed=-1.5,
    )
    if ok:
        carb.log_warn(
            f"[SpawnUsdRobot] Created Ai2_Bot2 Arena diff-drive graph at {graph_path} "
            f"subscribing to {cmd_vel_topic} and targeting {articulation_prim_path}"
        )
    else:
        carb.log_error(f"[SpawnUsdRobot] Failed to create Ai2_Bot2 Arena diff-drive graph at {graph_path}")
    return bool(ok)


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


def _reshape_param(value, columns: int):
    if hasattr(value, 'reshape'):
        return value.reshape([1, columns])
    if isinstance(value, (list, tuple)):
        return [list(value)]
    return value


def _first_available(source, *keys):
    for key in keys:
        if isinstance(source, dict) and key in source:
            return source[key]
        if hasattr(source, key):
            return getattr(source, key)
    return None


def _normalize_camera_info(camera_info):
    if isinstance(camera_info, dict):
        return {
            'width': _first_available(camera_info, 'width'),
            'height': _first_available(camera_info, 'height'),
            'projectionType': _first_available(camera_info, 'projectionType', 'projection_type') or 'pinhole',
            'k': _reshape_param(_first_available(camera_info, 'k'), 9),
            'r': _reshape_param(_first_available(camera_info, 'r'), 9),
            'p': _reshape_param(_first_available(camera_info, 'p'), 12),
            'physicalDistortionModel': _first_available(camera_info, 'physicalDistortionModel', 'physical_distortion_model', 'distortionModel', 'distortion_model') or 'plumb_bob',
            'physicalDistortionCoefficients': _first_available(camera_info, 'physicalDistortionCoefficients', 'physical_distortion_coefficients', 'distortionCoefficients', 'distortion_coefficients', 'd') or [0.0] * 5,
        }

    if isinstance(camera_info, (tuple, list)):
        if len(camera_info) >= 8:
            width, height, projection_type, k, r, p, distortion_model, distortion_coefficients = camera_info[:8]
            return {
                'width': width,
                'height': height,
                'projectionType': projection_type,
                'k': _reshape_param(k, 9),
                'r': _reshape_param(r, 9),
                'p': _reshape_param(p, 12),
                'physicalDistortionModel': distortion_model,
                'physicalDistortionCoefficients': distortion_coefficients,
            }
        for item in camera_info:
            try:
                return _normalize_camera_info(item)
            except Exception:
                continue

    if all(hasattr(camera_info, attr) for attr in ('width', 'height', 'k', 'r', 'p')):
        return {
            'width': getattr(camera_info, 'width'),
            'height': getattr(camera_info, 'height'),
            'projectionType': getattr(camera_info, 'projectionType', 'pinhole'),
            'k': _reshape_param(getattr(camera_info, 'k'), 9),
            'r': _reshape_param(getattr(camera_info, 'r'), 9),
            'p': _reshape_param(getattr(camera_info, 'p'), 12),
            'physicalDistortionModel': getattr(camera_info, 'physicalDistortionModel', getattr(camera_info, 'distortion_model', 'plumb_bob')),
            'physicalDistortionCoefficients': getattr(camera_info, 'physicalDistortionCoefficients', getattr(camera_info, 'd', [0.0] * 5)),
        }

    raise TypeError(f'Unsupported camera info format: {type(camera_info)!r}')


def _render_product_path(render_product) -> str:
    if isinstance(render_product, str):
        return render_product
    for attr_name in ('path', 'prim_path', 'render_product_path'):
        if not hasattr(render_product, attr_name):
            continue
        value = getattr(render_product, attr_name)
        if callable(value):
            value = value()
        if value:
            return str(value)
    for method_name in ('GetPath', 'get_path'):
        if not hasattr(render_product, method_name):
            continue
        try:
            value = getattr(render_product, method_name)()
        except Exception:
            continue
        if value:
            return str(value)
    return str(render_product)


def _publish_camera_info(render_product: str, frame: str, namespace: str, camera_topic: str, step_size: int):
    render_product_path = _render_product_path(render_product)
    camera_info = _normalize_camera_info(read_camera_info(render_product_path=render_product_path))
    writer = rep.writers.get('ROS2PublishCameraInfo')
    writer.initialize(
        frameId=frame,
        nodeNamespace=namespace,
        queueSize=1,
        topicName=os.path.join(camera_topic, 'camera_info'),
        width=camera_info['width'],
        height=camera_info['height'],
        projectionType=camera_info['projectionType'],
        k=camera_info['k'],
        r=camera_info['r'],
        p=camera_info['p'],
        physicalDistortionModel=camera_info['physicalDistortionModel'],
        physicalDistortionCoefficients=camera_info['physicalDistortionCoefficients'],
    )
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        'PostProcessDispatchIsaacSimulationGate', render_product_path
    )
    og.Controller.attribute(gate_path + '.inputs:step').set(step_size)


def _publish_rgb(render_product: str, frame: str, namespace: str, camera_topic: str, step_size: int):
    render_product_path = _render_product_path(render_product)
    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    writer = rep.writers.get(rv + 'ROS2PublishImage')
    writer.initialize(
        frameId=frame,
        nodeNamespace=namespace,
        queueSize=1,
        topicName=os.path.join(camera_topic, 'image'),
    )
    writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + 'IsaacSimulationGate', render_product_path)
    og.Controller.attribute(gate_path + '.inputs:step').set(step_size)


def _publish_depth(render_product: str, frame: str, namespace: str, camera_topic: str, step_size: int):
    render_product_path = _render_product_path(render_product)
    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.DistanceToImagePlane.name)
    for suffix in ('depth', 'depth_image'):
        writer = rep.writers.get(rv + 'ROS2PublishImage')
        writer.initialize(
            frameId=frame,
            nodeNamespace=namespace,
            queueSize=1,
            topicName=os.path.join(camera_topic, suffix),
        )
        writer.attach([render_product])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + 'IsaacSimulationGate', render_product_path)
    og.Controller.attribute(gate_path + '.inputs:step').set(step_size)


def _find_named_camera_prim(prim_path: str, preferred_name: str) -> str | None:
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return None

    first_camera = None
    preferred_lower = preferred_name.lower()
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() != 'Camera':
            continue
        path = str(prim.GetPath())
        if first_camera is None:
            first_camera = path
        if prim.GetName().lower() == preferred_lower or preferred_lower in path.lower():
            return path
    return first_camera


def _ensure_top_down_camera(prim_path: str, request_pose, namespace: str) -> str:
    safe_name = prim_path.strip('/').replace('/', '_')
    top_down_camera_path = f'/World/vln_top_down_camera_{safe_name}'
    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, top_down_camera_path)
    camera.CreateFocalLengthAttr().Set(12.0)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 200.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(request_pose.position.x), float(request_pose.position.y), 8.0))
    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    carb.log_warn(
        f"[SpawnUsdRobot] Created top-down camera {top_down_camera_path} "
        f"at ({float(request_pose.position.x):.3f}, {float(request_pose.position.y):.3f}, 8.000)"
    )
    return top_down_camera_path


def _setup_vln_camera_publishers(prim_path: str, namespace: str, base_frame: str | None, request_pose) -> None:
    if rep is None or sd is None or read_camera_info is None:
        carb.log_error('[SpawnUsdRobot] Cannot publish VLN cameras: Replicator/ROS2 bridge camera APIs unavailable')
        return

    frame_namespace = namespace.split('/')[-1] if namespace else ''
    base_frame_id = base_frame or BASE_FRAME_DEFAULT
    head_frame = os.path.join(frame_namespace, base_frame_id, 'head_camera') if frame_namespace else os.path.join(base_frame_id, 'head_camera')
    top_down_frame = os.path.join(frame_namespace, 'top_down_camera') if frame_namespace else 'top_down_camera'
    step_size = 6  # Isaac default 60 Hz -> 10 Hz image streams.

    head_camera_path = _find_named_camera_prim(prim_path, 'head_camera')
    if head_camera_path is None:
        carb.log_error(f'[SpawnUsdRobot] No Camera prim found under {prim_path}; head_camera ROS publishers not created')
    else:
        try:
            head_rp = rep.create.render_product(head_camera_path, (640, 480))
            _publish_camera_info(head_rp, head_frame, namespace, 'head_camera', step_size)
            _publish_rgb(head_rp, head_frame, namespace, 'head_camera', step_size)
            _publish_depth(head_rp, head_frame, namespace, 'head_camera', step_size)
            carb.log_warn(f'[SpawnUsdRobot] Publishing head camera {head_camera_path} on /{namespace}/head_camera/*')
        except Exception as e:
            carb.log_error(f'[SpawnUsdRobot] Failed to create head_camera ROS publishers: {e}\n{traceback.format_exc()}')

    try:
        top_down_path = _ensure_top_down_camera(prim_path, request_pose, namespace)
        top_rp = rep.create.render_product(top_down_path, (640, 640))
        _publish_camera_info(top_rp, top_down_frame, namespace, 'top_down_camera', step_size)
        _publish_rgb(top_rp, top_down_frame, namespace, 'top_down_camera', step_size)
        carb.log_warn(f'[SpawnUsdRobot] Publishing top-down camera {top_down_path} on /{namespace}/top_down_camera/*')
    except Exception as e:
        carb.log_error(f'[SpawnUsdRobot] Failed to create top_down_camera ROS publishers: {e}\n{traceback.format_exc()}')


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

    create_prim_safe(
        prim_path=prim_path,
        usd_path=usd_path,
    )
    carb.log_warn(f"[SpawnUsdRobot] USD loaded at {prim_path}")

    usd_name = os.path.splitext(os.path.basename(usd_path))[0]
    is_ai2_bot2 = name == 'Ai2_Bot2' or usd_name == 'Ai2_Bot2' or 'Ai2_Bot2' in usd_path
    if is_ai2_bot2:
        # Clean duplicate roots before the first Kit update gives PhysX a
        # chance to instantiate a split articulation.
        pre_update_root = _find_articulation_root(prim_path)
        if pre_update_root:
            _remove_extra_articulation_roots(prim_path, pre_update_root)

    try:
        import omni.kit.app
        omni.kit.app.get_app().update()
        carb.log_warn("[SpawnUsdRobot] Triggered app.update() for OmniGraph init")
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] app.update() failed: {e}")

    articulation_prim_path = _find_articulation_root(prim_path)
    if articulation_prim_path is None:
        carb.log_error(
            f"[SpawnUsdRobot] No prim with both ArticulationRootAPI and RigidBodyAPI "
            f"found under {prim_path}. Spawn aborted."
        )
        return ''
    carb.log_warn(f"[SpawnUsdRobot] Articulation root at: {articulation_prim_path}")

    try:
        if is_ai2_bot2:
            removed_roots = _remove_extra_articulation_roots(prim_path, articulation_prim_path)
            if removed_roots:
                try:
                    import omni.kit.app
                    omni.kit.app.get_app().update()
                except Exception:
                    pass
    except Exception as e:
        carb.log_warn(f"[SpawnUsdRobot] Duplicate articulation-root cleanup failed: {e}")

    try:
        _remap_namespace(prim_path, namespace, base_frame=base_frame)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Remap failed: {e}\n{traceback.format_exc()}")

    try:
        _remap_articulation_target(prim_path, articulation_prim_path)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Articulation target remap failed: {e}\n{traceback.format_exc()}")

    try:
        if is_ai2_bot2:
            _setup_ai2_bot2_control_graph(prim_path, articulation_prim_path, namespace)
            try:
                import omni.kit.app
                omni.kit.app.get_app().update()
            except Exception:
                pass
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Ai2_Bot2 control graph setup failed: {e}\n{traceback.format_exc()}")

    try:
        _setup_vln_camera_publishers(prim_path, namespace, base_frame, request.pose)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] VLN camera publisher setup failed: {e}\n{traceback.format_exc()}")

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

        # Arena's task_generator installs a synthetic fallback odom/TF publisher
        # for USD robots before this service is called.  Creating a second Isaac
        # odom graph on the same namespaced /odom topic is harmful when the USD
        # articulation/controller graph is present but not actually responding
        # to cmd_vel: Nav2 and the data recorder then receive alternating
        # stationary Isaac odom and moving fallback odom samples.  Keep the
        # explicit fallback as the single odom source unless a developer opts
        # into the Isaac graph for controller debugging.
        enable_isaac_odom_graph = str(
            os.environ.get('ARENA_SPAWN_USD_ROBOT_ENABLE_ISAAC_ODOM_GRAPH', '')
        ).strip().lower() in {'1', 'true', 'yes', 'on'}
        if enable_isaac_odom_graph:
            if not odom.odom(
                os.path.join(prim_path, 'odom_publisher'),
                prim_path=odom_prim_path,
                base_frame_id=fq_base_frame,
                odom_frame_id=fq_odom_frame,
                map_frame_id=fq_world_frame,
                odom_topic=odom_topic,
            ):
                carb.log_error('[SpawnUsdRobot] Failed to create odom graph')
        else:
            carb.log_warn(
                '[SpawnUsdRobot] Skipping Isaac USD odom graph; using Arena '
                f'task-generator fallback odom/TF on {odom_topic}'
            )

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
        # During spawn PhysX may not have finalized the articulation view yet.
        # Place the top-level Xform only; task resets use EditPrims with a
        # physics teleport after initialization has settled.
        physics_teleport=False,
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
    srv_type=SpawnUsdRobot_srv,
    srv_name='isaac/SpawnUsdRobot_srv',
    callback=spawn_usd_robot_callback,
)

__all__ = ['spawn_usd_robot_service']
