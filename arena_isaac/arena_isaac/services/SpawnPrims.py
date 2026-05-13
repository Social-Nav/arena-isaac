import numpy as np
import carb

from isaac_utils.utils import geom, prim
from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import Prim
from isaacsim_msgs.srv import SpawnPrims

from .utils import Service, on_exception


@on_exception(False)
def prim_importer(prim_msg: Prim) -> bool:
    name = prim_msg.name
    usd_path = prim_msg.usd_path
    prim.create_prim_safe(
        prim_path=world_path(name),
        position=np.array(geom.Translation.parse(prim_msg.pose.position).tuple()),
        orientation=np.array(geom.Rotation.parse(prim_msg.pose.orientation).quat()),
        usd_path=usd_path,
        scale=np.array([prim_msg.scale.x, prim_msg.scale.y, prim_msg.scale.z]),
    )

    return True


def spawn_prims_callback(request: SpawnPrims.Request, response: SpawnPrims.Response):
    response.ret = list(map(prim_importer, request.prims))
    carb.log_warn(
        f"[SpawnPrims] Spawned {sum(1 for ok in response.ret if ok)}/{len(response.ret)} prim(s); "
        "suppressing ROS response to avoid Isaac embedded rclpy response conversion abort"
    )
    raise RuntimeError('SpawnPrims response intentionally suppressed after spawn setup')


spawn_prims_service = Service(
    srv_type=SpawnPrims,
    srv_name='isaac/SpawnPrims',
    callback=spawn_prims_callback
)


__all__ = ['spawn_prims_service']
