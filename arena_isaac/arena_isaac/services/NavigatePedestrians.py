import numpy as np

from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people_manager import PeopleManager

from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import PedestrianGoal
from isaacsim_msgs.srv import NavigatePedestrians

from .utils import Service, on_exception

_LOOK_AHEAD_M = 2.0
_MIN_WALK_SPEED = 0.05   # below this hunav speed the agent counts as standing still
# Planar gap past which the character walks even when hunav reports no motion, to correct
# drift. Must exceed Person.update()'s 0.3 m pop-threshold.
_CATCHUP_M = 0.4
_CATCHUP_SPEED = 0.5

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

    # Idle on what hunav REPORTS, not on proximity. The old `dist < 0.01` was finer than
    # the 1 cm rounding hunav applies to the pose it sends, so quantisation noise read as
    # "arrived" and froze the character mid-route while /people kept moving in RViz.
    speed = float(getattr(goal, 'velocity', 0.8))
    if speed < _MIN_WALK_SPEED and dist < _CATCHUP_M:
        person._target_positions.clear()
        return True

    if dist < 1e-6:
        return True   # no direction this cycle; leave the queue alone

    if speed < _MIN_WALK_SPEED:
        speed = _CATCHUP_SPEED   # catching up: Walk=0 would animate in place

    # Project _LOOK_AHEAD_M ahead so the waypoint stays beyond Person.update()'s 0.3 m
    # pop-threshold. direction has z=0, so the target keeps the character's own height.
    unit = direction / dist
    far_target = current_pos + unit * _LOOK_AHEAD_M

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
