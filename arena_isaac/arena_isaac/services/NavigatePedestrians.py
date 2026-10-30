import numpy as np
import carb

from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people_manager import PeopleManager

try:
    from isaacsim.core.utils.prims import is_prim_path_valid
except ImportError:
    from omni.isaac.core.utils.prims import is_prim_path_valid

from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import PedestrianGoal
from isaacsim_msgs.srv import NavigatePedestrians

from .utils import Service, on_exception

_DEFAULT_LOOK_AHEAD_M = 2.0
_MIN_LOOK_AHEAD_M = 0.35
_STOP_SPEED_EPS = 0.05
_STALL_MIN_LOGICAL_GAP_M = 0.15
_STALL_POSITION_EPS_M = 1.0e-4
_STALL_CHECK_TICKS = 10


def _character_is_stalled(person: Person, current_pos: np.ndarray, logical_gap: float) -> bool:
    """Detect an accepted walking command whose character transform never advances."""
    current_xy = np.asarray(current_pos[:2], dtype=float)
    previous_xy = getattr(person, "_arena_last_nav_xy", None)
    stalled_ticks = int(getattr(person, "_arena_nav_stalled_ticks", 0))

    if (
        logical_gap >= _STALL_MIN_LOGICAL_GAP_M
        and previous_xy is not None
        and float(np.linalg.norm(current_xy - previous_xy)) <= _STALL_POSITION_EPS_M
    ):
        stalled_ticks += 1
    else:
        stalled_ticks = 0

    person._arena_last_nav_xy = current_xy.copy()
    person._arena_nav_stalled_ticks = stalled_ticks
    return stalled_ticks >= _STALL_CHECK_TICKS

@on_exception(False)
def navigate_pedestrian(goal: PedestrianGoal) -> bool:
    usd_path = world_path(goal.name)
    person = PeopleManager.get_people_manager().get_person(usd_path)
    if (
        not isinstance(person, Person)
        or not is_prim_path_valid(usd_path)
        or not person.animation_ready
    ):
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
    # Treat very slow motion as idle to avoid driving the animation with small corrections.
    if speed <= _STOP_SPEED_EPS:
        person._target_positions.clear()
        person._target_speed = 0.0
        person._arena_nav_stalled_ticks = 0
        person._arena_last_nav_xy = np.asarray(current_pos[:2], dtype=float).copy()
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
    if _character_is_stalled(person, current_pos, dist):
        carb.log_error(
            f"Pedestrian {goal.name} accepted walking targets but its Isaac transform "
            f"did not move for {_STALL_CHECK_TICKS} updates; logical gap={dist:.3f} m, "
            f"speed={speed:.3f} m/s"
        )
        return False
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
