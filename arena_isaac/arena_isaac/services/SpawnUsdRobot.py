import os
import traceback

import carb
from isaac_utils.managers.door_manager import DoorManager
from isaac_utils.managers.elevator_manager import ElevatorManager
from isaac_utils.utils import geom
from isaac_utils.utils.path import world_path
from isaac_utils.utils.prim import create_prim_safe, ensure_path
from isaacsim_msgs.srv import SpawnUsdRobot
from pxr import Usd, UsdPhysics

import omni.usd
import omni.graph.core as og

from .utils import Service, on_exception

FRAME_KEYWORDS = {
    'frameid', 'parentframeid', 'childframeid',
    'odomframeid', 'chassisframeid',
}
TF_DETECT_KEYWORDS = {'parentframeid', 'childframeid'}
NAMESPACE_KEYWORDS = {'nodenamespace'}


def _find_articulation_root(prim_path: str) -> str | None:
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    return None


def _get_og_attr_value(prim_path_str: str, attr_name: str):
    """Read attribute value via OmniGraph API (for unauthored defaults)."""
    try:
        node = og.get_node_by_path(prim_path_str)
        if node and node.is_valid():
            og_attr = node.get_attribute(attr_name)
            if og_attr and og_attr.is_valid():
                val = og_attr.get()
                carb.log_error(
                    f"[SpawnUsdRobot] OG fallback {attr_name}: "
                    f"{repr(val)} on {prim_path_str}"
                )
                return val
    except Exception as e:
        carb.log_error(
            f"[SpawnUsdRobot] OG fallback failed for {attr_name}: {e}"
        )
    return None


def _remap_namespace(prim_path: str, namespace: str):
    """Traverse USD stage prims under prim_path and modify OmniGraph node
    attributes directly via the USD API."""
    if not namespace:
        carb.log_error(f"[SpawnUsdRobot] No namespace provided, skipping remap")
        return

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
                    carb.log_error(
                        f"[SpawnUsdRobot] Skipping nodeNamespace on TF node "
                        f"(has parentFrameId+childFrameId): {prim_path_str}"
                    )
                    continue
                try:
                    attr.Set(namespace)
                    carb.log_error(
                        f"[SpawnUsdRobot] Set {attr_name}='{namespace}' "
                        f"on {prim_path_str}"
                    )
                    remapped_count += 1
                except Exception as e:
                    carb.log_error(
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
                        carb.log_error(
                            f"[SpawnUsdRobot] Skipping {attr_name} on "
                            f"{prim_path_str}: no value from USD or OG API"
                        )
                        continue
                    new_val = f"{namespace}/{old_str}"
                    attr.Set(new_val)
                    carb.log_error(
                        f"[SpawnUsdRobot] Remapped {attr_name}: "
                        f"'{old_str}' -> '{new_val}' on {prim_path_str}"
                    )
                    remapped_count += 1
                except Exception as e:
                    carb.log_error(
                        f"[SpawnUsdRobot] Failed to remap {attr_name} "
                        f"on {prim_path_str}: {e}"
                    )

    carb.log_error(
        f"[SpawnUsdRobot] Namespace remap complete: "
        f"{remapped_count} attributes updated under {prim_path}"
    )


@on_exception('')
def spawn_usd_robot(request: SpawnUsdRobot.Request) -> str:
    name = request.name
    usd_path = request.usd_path
    namespace = request.robot_namespace

    carb.log_error(f"[SpawnUsdRobot] === Starting spawn: name={name}, ns={namespace} ===")

    prim_path = world_path(name)
    ensure_path(os.path.dirname(prim_path))

    create_prim_safe(
        prim_path=prim_path,
        usd_path=usd_path,
    )
    carb.log_error(f"[SpawnUsdRobot] USD loaded at {prim_path}")

    try:
        import omni.kit.app
        omni.kit.app.get_app().update()
        carb.log_error("[SpawnUsdRobot] Triggered app.update() for OmniGraph init")
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] app.update() failed: {e}")

    articulation_prim_path = _find_articulation_root(prim_path)
    if articulation_prim_path is None:
        carb.log_error(
            f"[SpawnUsdRobot] No ArticulationRootAPI found, "
            f"falling back to prim_path"
        )
        articulation_prim_path = prim_path
    else:
        carb.log_error(f"[SpawnUsdRobot] Articulation root at: {articulation_prim_path}")

    try:
        _remap_namespace(prim_path, namespace)
    except Exception as e:
        carb.log_error(f"[SpawnUsdRobot] Remap failed: {e}\n{traceback.format_exc()}")

    geom.register_robot(
        robot_prim_path=prim_path,
        articulation_prim_path=articulation_prim_path,
    )

    geom.move(
        prim_path=prim_path,
        translation=geom.Translation.parse(request.pose.position),
        rotation=geom.Rotation.parse(request.pose.orientation),
    )

    DoorManager.instance().add_robot(prim_path, None)
    ElevatorManager.instance().add_robot(prim_path)

    carb.log_error(f"[SpawnUsdRobot] === Spawn complete: {prim_path} ===")
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
