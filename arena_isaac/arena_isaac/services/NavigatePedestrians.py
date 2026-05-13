import numpy as np
import carb

from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people_manager import PeopleManager

from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import PedestrianGoal
from isaacsim_msgs.srv import NavigatePedestrians

from .utils import Service, on_exception

_LOOK_AHEAD_M = 2.0

@on_exception(False)
def navigate_pedestrian(goal: PedestrianGoal) -> bool:
    usd_path = world_path(goal.name)
    person = PeopleManager.get_people_manager().get_person(usd_path)
    if not isinstance(person, Person):
        return False

    hunav_pos = np.array([goal.position.x, goal.position.y, goal.position.z])
    current_pos = person.state.position  # updated by Person.update_state() every physics step

    direction = hunav_pos - current_pos
    dist = np.linalg.norm(direction)

    if dist < 0.01:
        # Essentially stationary: clear queue so character idles
        person._target_positions.clear()
        return True

    # Project _LOOK_AHEAD_M ahead in the hunav direction so the waypoint is
    # always beyond Person.update()'s 0.3 m pop-threshold.
    unit = direction / dist
    far_target = current_pos + unit * _LOOK_AHEAD_M

    person._target_positions.clear()
    person.update_target_positions(
        [[float(far_target[0]), float(far_target[1]), float(far_target[2])]],
        getattr(goal, 'velocity', 0.8),
    )
    return True


def navigate_pedestrians_callback(request: NavigatePedestrians.Request, response: NavigatePedestrians.Response):
    response.ret = list(map(navigate_pedestrian, request.goals))
    carb.log_warn(
        f"[NavigatePedestrians] Updated {sum(1 for ok in response.ret if ok)}/{len(response.ret)} pedestrian target(s); "
        "suppressing ROS response to avoid Isaac embedded rclpy response conversion abort"
    )
    raise RuntimeError('NavigatePedestrians response intentionally suppressed after navigation update')


navigate_pedestrians_service = Service(
    srv_type=NavigatePedestrians,
    srv_name="isaac/NavigatePedestrians",
    callback=navigate_pedestrians_callback
)

__all__ = ['navigate_pedestrians_service']
