import numpy as np

from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people_manager import PeopleManager

from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import PedestrianGoal
from isaacsim_msgs.srv import NavigatePedestrians

from .utils import Service, on_exception

_DEFAULT_LOOK_AHEAD_M = 2.0
_MIN_LOOK_AHEAD_M = 0.35
_MIN_WALK_SPEED = 0.05   # below this hunav speed the agent counts as standing still

@on_exception(False)
def navigate_pedestrian(goal: PedestrianGoal) -> bool:
    usd_path = world_path(goal.name)
    person = PeopleManager.get_people_manager().get_person(usd_path)
    if not isinstance(person, Person):
        return False

    current_pos = person.state.position  # updated by Person.update_state() every physics step

    # PLANAR only: hunav is 2D and sends z=0, while person.state.position carries the
    # character's own z. A 3D norm mixed that offset into the distance and tilted the
    # look-ahead target off the ground plane.
    direction = np.array([goal.position.x, goal.position.y, 0.0]) - np.array(
        [current_pos[0], current_pos[1], 0.0]
    )
    dist = float(np.linalg.norm(direction))

    # HuNav reports zero speed after a non-cyclic agent consumes its final goal. Stop the
    # rendered character immediately as well: a distance-based catch-up here would keep the
    # character walking after the logical agent had already reached the end of its route.
    speed = float(getattr(goal, 'velocity', 0.8))
    if speed < _MIN_WALK_SPEED:
        person._target_positions.clear()
        person._target_speed = 0.0
        return True

    if dist < 1e-6:
        return True   # no direction this cycle; leave the queue alone

    # Project ahead so the waypoint stays beyond Person.update()'s 0.3 m
    # pop-threshold. direction has z=0, so the target keeps the character's own height.
    configured_look_ahead = float(getattr(goal, 'look_ahead', _DEFAULT_LOOK_AHEAD_M))
    look_ahead = max(
        _MIN_LOOK_AHEAD_M,
        configured_look_ahead if configured_look_ahead > 0.0 else _DEFAULT_LOOK_AHEAD_M,
    )
    unit = direction / dist
    far_target = current_pos + unit * look_ahead

    person._target_positions.clear()
    person.update_target_positions(
        [[float(far_target[0]), float(far_target[1]), float(far_target[2])]],
        speed,
    )
    return True


def navigate_pedestrians_callback(request: NavigatePedestrians.Request, response: NavigatePedestrians.Response):
    response.ret = list(map(navigate_pedestrian, request.goals))
    return response


navigate_pedestrians_service = Service(
    srv_type=NavigatePedestrians,
    srv_name="isaac/NavigatePedestrians",
    callback=navigate_pedestrians_callback
)

__all__ = ['navigate_pedestrians_service']
