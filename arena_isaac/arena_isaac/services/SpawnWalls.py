import math

import carb
import numpy as np
import omni
import omni.usd
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import UsdPhysics


from isaac_utils.utils.geom import Rotation, Scale, Translation
from isaac_utils.utils.material import Material
from isaac_utils.utils.mesh import create_cube
from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import Wall
from isaacsim_msgs.srv import SpawnWalls

from .utils import Service, on_exception


@on_exception(False)
def wall_spawner(wall: Wall) -> bool:

    prim_path = world_path(wall.name)
    thickness = wall.thickness

    start = Translation.parse(wall.start).Vec3d()
    end = Translation.parse(wall.end).Vec3d()
    vector_ab = end - start

    center = ((start + end) / 2)

    length = float(np.linalg.norm(vector_ab[:2]))
    angle = math.atan2(vector_ab[1], vector_ab[0])
    # print("wall angle", angle)

    # create wall
    create_cube(
        prim_path=prim_path,
        position=Translation(*center),
        scale=Scale(length, thickness, end[2] - start[2]),
        rotation=Rotation.parse([0, 0, angle]),
    )

    # Add PhysX CollisionAPI so raycast_closest() can detect this wall.
    stage = omni.usd.get_context().get_stage()
    wall_prim = stage.GetPrimAtPath(prim_path)
    if wall_prim.IsValid() and not wall_prim.HasAPI(UsdPhysics.CollisionAPI):
        collision_api = UsdPhysics.CollisionAPI.Apply(wall_prim)
        collision_api.CreateCollisionEnabledAttr(True)

    if (material := Material.from_msg(wall.material)):
        material.bind_to(prim_path)

    return True


def spawn_walls_callback(request: SpawnWalls.Request, response: SpawnWalls.Response):
    response.ret = list(map(wall_spawner, request.walls))
    carb.log_warn(
        f"[SpawnWalls] Spawned {sum(1 for ok in response.ret if ok)}/{len(response.ret)} wall segment(s); "
        "suppressing ROS response to avoid Isaac embedded rclpy response conversion abort"
    )
    raise RuntimeError('SpawnWalls response intentionally suppressed after spawn setup')


spawn_walls_service = Service(
    srv_type=SpawnWalls,
    srv_name='isaac/SpawnWalls',
    callback=spawn_walls_callback


)

__all__ = ['spawn_walls_service']
