from geometry_msgs.msg import Pose
import math

import carb
import omni.usd
from isaac_utils.utils import geom
from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import Scale
from isaacsim_msgs.srv import EditPrims

from .utils import Service, on_exception


def _resolve_existing_prim_path(name: str) -> str:
    """Resolve legacy bare robot names to the canonical /World/Robots path.

    Generic Isaac services still accept arbitrary prim names through
    `world_path()`.  For robot reset/move compatibility, if the direct path does
    not exist and a `/World/Robots/<name>` prim does, use the robot path instead.
    """
    prim_path = world_path(name)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return prim_path
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        return prim_path

    raw = str(name or '').strip().strip('/')
    if not raw or raw.startswith('World/') or raw.startswith('Robots/'):
        return prim_path

    robot_prim_path = world_path('Robots', raw)
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if robot_prim and robot_prim.IsValid():
        carb.log_warn(f"[EditPrims] Resolved legacy prim name '{name}' to {robot_prim_path}")
        return robot_prim_path
    return prim_path


@on_exception(False)
def move_prim(name: str, pose: Pose) -> bool:
    geom.move(
        prim_path=_resolve_existing_prim_path(name),
        translation=geom.Translation.parse(pose.position),
        rotation=geom.Rotation.parse(pose.orientation),
    )

    return True


@on_exception(False)
def move_attached_top_down_camera(name: str, pose: Pose) -> bool:
    robot_prim_path = _resolve_existing_prim_path(name)
    safe_name = robot_prim_path.strip('/').replace('/', '_')
    top_down_camera_path = f'/World/vln_top_down_camera_{safe_name}'

    stage = omni.usd.get_context().get_stage()
    top_down_camera_prim = stage.GetPrimAtPath(top_down_camera_path) if stage else None
    if top_down_camera_prim is None or not top_down_camera_prim.IsValid():
        carb.log_warn(
            f"[EditPrims] No attached top-down camera found at {top_down_camera_path}; "
            f"moved {robot_prim_path} only"
        )
        return True

    z = 8.0
    translation = geom.get_world_translation(top_down_camera_path)
    if translation is not None and math.isfinite(float(translation.z)):
        z = float(translation.z)

    geom.move(
        prim_path=top_down_camera_path,
        translation=geom.Translation(float(pose.position.x), float(pose.position.y), z),
        # Keep the standalone USD camera in a deterministic nadir view.  USD
        # cameras look along local -Z, so identity at z=8 points straight down.
        # It is not parented under the robot, so every reset must explicitly
        # re-lock both its position and orientation to the reset pose.
        rotation=geom.Rotation(1.0, 0.0, 0.0, 0.0),
    )
    carb.log_warn(
        f"[EditPrims] Moved attached top-down camera {top_down_camera_path} "
        f"to ({float(pose.position.x):.3f}, {float(pose.position.y):.3f}, {z:.3f})"
    )
    return True


def move_prim_with_attached_views(name: str, pose: Pose) -> bool:
    moved = move_prim(name, pose)
    if not moved:
        return False
    return move_attached_top_down_camera(name, pose)


@on_exception(False)
def scale_prim(name: str, scale: Scale) -> bool:
    prim_path = _resolve_existing_prim_path(name)

    geom.rescale(
        prim_path=prim_path,
        scale=geom.Scale.parse(scale),
    )

    return True


def edit_prims_callback(request: EditPrims.Request, response: EditPrims.Response):
    results = (True for _ in request.prims)

    if request.pose:
        results = (
            a and b
            for a, b in zip(
                results,
                map(move_prim_with_attached_views, (p.name for p in request.prims), (p.pose for p in request.prims))
            )
        )

    if request.scale:
        results = (a and b for a, b in zip(results, map(scale_prim, (p.name for p in request.prims), (p.scale for p in request.prims))))

    response.ret = list(results)
    return response


edit_prims_service = Service(
    srv_type=EditPrims,
    srv_name='isaac/EditPrims',
    callback=edit_prims_callback
)

__all__ = ['edit_prims_service']
