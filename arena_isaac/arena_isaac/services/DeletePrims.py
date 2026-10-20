import omni.kit.commands as commands
from isaac_utils.utils import geom
from isaac_utils.utils.path import world_path
from isaacsim.core.experimental.prims import Prim

from isaacsim_msgs.srv import DeletePrims

from .SpawnUsdRobot import unregister_vln_camera_robot
from .utils import Service, on_exception


def _coerce_resolved_path(value) -> str | None:
    """Normalize Isaac path resolution results to a single prim path string."""
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if value is None:
        return None
    return str(value)


@on_exception(False)
def delete_prim(name: str) -> bool:
    target = world_path(name)
    # Cleanup is keyed by the exact requested robot prim path and runs even if
    # USD path resolution no longer finds the prim. A live handle cleanup error
    # fails closed before the prim can be destroyed underneath it.
    unregister_vln_camera_robot(target, fail_on_error=True)
    target = _coerce_resolved_path(Prim.resolve_paths([target]))
    if not target:
        return True
    geom.unregister_robot(target)
    commands.execute(
        "IsaacSimDestroyPrim",
        prim_path=target,
    )
    return True


def delete_prims_callback(request: DeletePrims.Request, response: DeletePrims.Response):
    response.ret = list(map(delete_prim, request.names))
    return response


delete_prims_service = Service(
    srv_type=DeletePrims,
    srv_name='isaac/DeletePrims',
    callback=delete_prims_callback
)

__all__ = ['delete_prims_service']
