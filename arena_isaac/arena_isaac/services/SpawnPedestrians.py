import math
import os
import time

from omni.isaac.core import World
from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people.rendered_state_backend import make_backend_if_enabled
from pedestrian.simulator.logic.people_manager import PeopleManager

from isaac_utils.utils.path import world_path
from isaac_utils.utils.prim import ensure_path
from isaacsim_msgs.msg import Pedestrian
from isaacsim_msgs.srv import SpawnPedestrians

from .utils import Service, on_exception

# simple logger helpers
try:
    from rclpy.logging import get_logger
    _LOGGER = get_logger('isaac_spawn_ped')
except Exception:
    _LOGGER = None


@on_exception(False)
def spawn_pedestrian(pedestrian: Pedestrian) -> bool:
    started = time.monotonic()
    world = World.instance()

    position = [pedestrian.pose.position.x, pedestrian.pose.position.y, pedestrian.pose.position.z]
    orientation = 2.0 * math.atan2(pedestrian.pose.orientation.z, pedestrian.pose.orientation.w)

    usd_path = world_path(pedestrian.name)
    ensure_path(os.path.dirname(usd_path))
    if _LOGGER is not None:
        _LOGGER.warn(
            f"[SpawnPedestrians] spawning {pedestrian.name} as {pedestrian.character_name} at "
            f"({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
        )

    # Destroying the old Person first cleanly removes its callbacks and USD prim.
    PeopleManager.get_people_manager().remove_person(usd_path)  # no-op on first spawn

    # The backend makes the RENDERED pose observable.  Person.update_state() already reads it out of
    # the animation graph every physics step; without a backend it was thrown away, which is why a
    # divergence between the rendered pedestrian and the logical track the evaluation grades against
    # left no artifact and raised no error.
    backend = make_backend_if_enabled()

    if not pedestrian.controller_stats:
        Person(world, usd_path, pedestrian.character_name, position, orientation, backend=backend)
    else:
        Person(world, usd_path, pedestrian.character_name, position, orientation,
               pedestrian.controller_name, backend=backend)

    if _LOGGER is not None:
        _LOGGER.warn(f"[SpawnPedestrians] spawned {pedestrian.name} in {time.monotonic() - started:.3f}s")
    return True


def spawn_pedestrians_callback(request: SpawnPedestrians.Request, response: SpawnPedestrians.Response):
    started = time.monotonic()
    if _LOGGER is not None:
        _LOGGER.warn(f"[SpawnPedestrians] request with {len(request.pedestrians)} pedestrian(s)")
    response.ret = list(map(spawn_pedestrian, request.pedestrians))
    if _LOGGER is not None:
        _LOGGER.warn(f"[SpawnPedestrians] request completed in {time.monotonic() - started:.3f}s ret={list(response.ret)}")
    return response


spawn_pedestrians_service = Service(
    srv_type=SpawnPedestrians,
    srv_name='isaac/SpawnPedestrians',
    callback=spawn_pedestrians_callback
)

__all__ = ['spawn_pedestrians_service']
