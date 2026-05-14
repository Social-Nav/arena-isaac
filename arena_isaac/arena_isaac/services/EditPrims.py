from geometry_msgs.msg import Pose
import math

import carb
from isaac_utils.utils import geom
from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import Scale
from isaacsim_msgs.srv import EditPrims

from .utils import Service, on_exception


@on_exception(False)
def move_prim(name: str, pose: Pose) -> bool:
    # Task resets must move the live PhysX articulation root, not just the USD
    # container Xform; otherwise the robot can remain simulated at its previous
    # pose and immediately fall/teleport once physics catches up. geom.move()
    # now queues articulation teleports onto the post-tick path in
    # isaac_utils/utils/geom.py, so reset-time physics teleports no longer touch
    # Articulation() inline during the fragile spawn/reset window.
    geom.move(
        prim_path=world_path(name),
        translation=geom.Translation.parse(pose.position),
        rotation=geom.Rotation.parse(pose.orientation),
        physics_teleport=True,
    )

    return True


@on_exception(False)
def move_attached_top_down_camera(name: str, pose: Pose) -> bool:
    robot_prim_path = world_path(name)
    safe_name = robot_prim_path.strip('/').replace('/', '_')
    top_down_camera_path = f'/World/vln_top_down_camera_{safe_name}'

    z = 8.0
    translation = geom.get_world_translation(top_down_camera_path)
    if translation is not None and math.isfinite(float(translation.z)):
        z = float(translation.z)

    geom.move(
        prim_path=top_down_camera_path,
        translation=geom.Translation(float(pose.position.x), float(pose.position.y), z),
        # Keep the standalone top-down camera in a deterministic nadir view.
        # It is not parented under the robot, so every reset must explicitly
        # re-lock both its position and orientation to the reset pose.
        rotation=geom.Rotation(w=1.0, x=0.0, y=0.0, z=0.0),
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
    prim_path = world_path(name)

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
    carb.log_warn(
        f"[EditPrims] Edited {sum(1 for ok in response.ret if ok)}/{len(response.ret)} prim(s); "
        "suppressing ROS response to avoid Isaac embedded rclpy response conversion abort"
    )
    raise RuntimeError('EditPrims response intentionally suppressed after edit')


edit_prims_service = Service(
    srv_type=EditPrims,
    srv_name='isaac/EditPrims',
    callback=edit_prims_callback
)

__all__ = ['edit_prims_service']
