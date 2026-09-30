import math
import os

from omni.isaac.core import World
from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people.character_selector import (
    CharacterPoolSelector,
)
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


_CHARACTER_SELECTOR = CharacterPoolSelector()
if _LOGGER is not None:
    _LOGGER.info(
        f"Pedestrian model selector seed: {_CHARACTER_SELECTOR.seed} "
        f"(override with ARENA_PEDESTRIAN_MODEL_SEED)"
    )


@on_exception(False)
def spawn_pedestrian(pedestrian: Pedestrian) -> bool:
    world = World.instance()

    position = [pedestrian.pose.position.x, pedestrian.pose.position.y, pedestrian.pose.position.z]
    orientation = 2.0 * math.atan2(pedestrian.pose.orientation.z, pedestrian.pose.orientation.w)

    usd_path = world_path(pedestrian.name)
    ensure_path(os.path.dirname(usd_path))

    # Destroying the old Person first cleanly removes its callbacks and USD prim.
    PeopleManager.get_people_manager().remove_person(usd_path)  # no-op on first spawn

    requested_model = pedestrian.character_name
    resolved_model = _CHARACTER_SELECTOR.resolve(
        pedestrian.name,
        requested_model,
    )
    if _LOGGER is not None:
        _LOGGER.info(
            f"Pedestrian model selection: entity={pedestrian.name} "
            f"requested={requested_model} resolved={resolved_model}"
        )

    if not pedestrian.controller_stats:
        Person(world, usd_path, resolved_model, position, orientation)
    else:
        Person(
            world,
            usd_path,
            resolved_model,
            position,
            orientation,
            pedestrian.controller_name,
        )

    return True


def spawn_pedestrians_callback(request: SpawnPedestrians.Request, response: SpawnPedestrians.Response):
    response.ret = list(map(spawn_pedestrian, request.pedestrians))
    return response


spawn_pedestrians_service = Service(
    srv_type=SpawnPedestrians,
    srv_name='isaac/SpawnPedestrians',
    callback=spawn_pedestrians_callback
)

__all__ = ['spawn_pedestrians_service']
