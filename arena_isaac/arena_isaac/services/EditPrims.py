import carb
from geometry_msgs.msg import Pose
from pxr import Usd

import omni.graph.core as og
import omni.usd

from isaac_utils.utils import geom
from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import Scale
from isaacsim_msgs.srv import EditPrims

from .utils import Service, on_exception

DIFFERENTIAL_CONTROLLER_TYPE = 'isaacsim.robot.wheeled_robots.DifferentialController'


def _rescale_differential_controllers(prim_path: str, scale: float) -> int:
    """Scale a DifferentialController's wheel constants so they still describe the ACTUAL
    wheels after the prim was rescaled.

        w_l = (v - wz*wheelDistance/2) / wheelRadius      # what the controller emits
        w_r = (v + wz*wheelDistance/2) / wheelRadius

    Joint velocity is rad/s and does not scale with the prim, but ground speed is
    `w * real_radius` and the real radius just became `wheelRadius * scale`. Left
    unscaled, the robot travels at `scale` times the commanded vx (0.8 cmd -> 0.64
    measured at scale 0.8) while every nav2 stage faithfully reports the full value.

    Scale BOTH constants: `v` enters only via the wheel-speed sum and `wz*wheelDistance`
    only via the difference, so fixing the radius alone would leave wz overshooting by
    1/scale. Assumes joint anchors scale with the prim (their localPos0 is parent-local).
    A pure-rotation command (v=0, wz=0.5) checks that: measured/commanded wz ~1.0 if they
    do, ~scale if they do not -- in which case the radius alone was the right fix.
    """
    if scale == 1.0:
        return 0

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return 0

    updated = 0
    for prim in Usd.PrimRange(root_prim):
        if 'OmniGraph' not in prim.GetTypeName():
            continue

        node_type_attr = prim.GetAttribute('node:type')
        if not node_type_attr or not node_type_attr.IsValid():
            continue
        if str(node_type_attr.Get() or '') != DIFFERENTIAL_CONTROLLER_TYPE:
            continue

        prim_path_str = str(prim.GetPath())
        og_node = og.get_node_by_path(prim_path_str)

        for attr_name in ('inputs:wheelRadius', 'inputs:wheelDistance'):
            attr = prim.GetAttribute(attr_name)
            old = attr.Get() if attr and attr.IsValid() else None

            # The graph is already running and serves its own cached values, so a USD-only
            # Set() never reaches the controller (same reason _get_og_attr_value exists for
            # reads). Write both: OmniGraph for effect, USD to keep the authored value.
            og_attr = og_node.get_attribute(attr_name) if og_node and og_node.is_valid() else None
            if og_attr is not None and og_attr.is_valid() and old is None:
                old = og_attr.get()
            if old is None:
                continue
            new = float(old) * scale

            try:
                if attr and attr.IsValid():
                    attr.Set(new)
                if og_attr is not None and og_attr.is_valid():
                    og_attr.set(new)
                else:
                    carb.log_warn(
                        f"[EditPrims] {attr_name}: no live OmniGraph attribute on "
                        f"{prim_path_str}; USD-only write may not reach the controller"
                    )
                carb.log_warn(
                    f"[EditPrims] Rescaled {attr_name} {old:.6f} -> {new:.6f} "
                    f"(scale {scale}) on {prim_path_str}"
                )
                updated += 1
            except Exception as e:
                carb.log_warn(
                    f"[EditPrims] Failed to rescale {attr_name} on {prim_path_str}: {e}"
                )

    if not updated:
        carb.log_warn(
            f"[EditPrims] No DifferentialController found under {prim_path}; if this robot "
            "is wheel-driven its commanded and actual velocity will differ by the scale factor"
        )
    return updated


@on_exception(False)
def move_prim(name: str, pose: Pose) -> bool:
    geom.move(
        prim_path=world_path(name),
        translation=geom.Translation.parse(pose.position),
        rotation=geom.Rotation.parse(pose.orientation),
    )

    return True


@on_exception(False)
def scale_prim(name: str, scale: Scale) -> bool:
    prim_path = world_path(name)

    geom.rescale(
        prim_path=prim_path,
        scale=geom.Scale.parse(scale),
    )

    # Keep any differential-drive controller's wheel constants consistent with
    # the geometry we just rescaled. Uniform scale only -- a non-uniform scale
    # has no single wheel radius, so leave the constants alone and say so.
    sx, sy, sz = float(scale.x), float(scale.y), float(scale.z)
    if sx == sy == sz:
        _rescale_differential_controllers(prim_path, sx)
    else:
        carb.log_warn(
            f"[EditPrims] Non-uniform scale ({sx}, {sy}, {sz}) on {prim_path}; "
            "differential-drive wheel constants left unchanged -- commanded vs "
            "actual velocity will disagree if this robot is wheel-driven"
        )

    return True


def edit_prims_callback(request: EditPrims.Request, response: EditPrims.Response):
    results = (True for _ in request.prims)

    if request.pose:
        results = (a and b for a, b in zip(results, map(move_prim, (p.name for p in request.prims), (p.pose for p in request.prims))))

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
