import time

import carb
import numpy as np

from pedestrian.simulator.logic.people.person import Person
from pedestrian.simulator.logic.people_manager import PeopleManager

from isaac_utils.utils.path import world_path
from isaacsim_msgs.msg import PedestrianGoal
from isaacsim_msgs.srv import NavigatePedestrians

from .utils import Service, on_exception

_LOOK_AHEAD_M = 2.0
_DEFAULT_WALK_SPEED_MPS = 0.8
_MIN_WALK_SPEED_MPS = 0.05
_LOG_PERIOD_SEC = 5.0
_LAST_LOG_TIME_BY_KEY: dict[str, float] = {}


def _throttled_warn(key: str, message: str) -> None:
    now = time.monotonic()
    last = _LAST_LOG_TIME_BY_KEY.get(key, 0.0)
    if now - last >= _LOG_PERIOD_SEC:
        _LAST_LOG_TIME_BY_KEY[key] = now
        carb.log_warn(message)


def _normalize_numeric_suffix(name: str) -> str:
    prefix, sep, suffix = name.rpartition("_")
    if sep and suffix.isdigit():
        return f"{prefix}_{int(suffix)}"
    return name


def _candidate_people_paths(goal_name: str) -> list[str]:
    base_path = world_path(goal_name)
    candidates = [base_path]

    parts = base_path.rstrip("/").split("/")
    leaf = parts[-1] if parts else base_path
    normalized_leaf = _normalize_numeric_suffix(leaf)
    if normalized_leaf != leaf:
        candidates.append("/".join(parts[:-1] + [normalized_leaf]))

    prefix, sep, suffix = leaf.rpartition("_")
    if sep and suffix.isdigit() and len(suffix) == 1:
        candidates.append("/".join(parts[:-1] + [f"{prefix}_{int(suffix):02d}"]))

    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(candidates))


def _resolve_person(goal_name: str) -> tuple[str, Person | None]:
    manager = PeopleManager.get_people_manager()
    for candidate_path in _candidate_people_paths(goal_name):
        person = manager.get_person(candidate_path)
        if isinstance(person, Person):
            return candidate_path, person

    available = sorted(getattr(manager, "people", {}).keys())
    _throttled_warn(
        f"missing:{goal_name}",
        f"[NavigatePedestrians] no visible person for goal.name={goal_name!r}; "
        f"tried={_candidate_people_paths(goal_name)} available={available}",
    )
    return world_path(goal_name), None


@on_exception(False)
def navigate_pedestrian(goal: PedestrianGoal) -> bool:
    usd_path, person = _resolve_person(goal.name)
    if person is None:
        return False

    hunav_pos = np.array([goal.position.x, goal.position.y, goal.position.z])
    current_pos = np.array(person.state.position, dtype=float)  # updated by Person.update_state() every physics step

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

    requested_speed = float(getattr(goal, 'velocity', 0.0) or 0.0)
    walk_speed = requested_speed if requested_speed >= _MIN_WALK_SPEED_MPS else _DEFAULT_WALK_SPEED_MPS
    person._target_positions.clear()
    person.update_target_positions(
        [[float(far_target[0]), float(far_target[1]), float(far_target[2])]],
        walk_speed,
    )
    _throttled_warn(
        f"target:{usd_path}",
        f"[NavigatePedestrians] animation target {goal.name!r} -> {usd_path} "
        f"current=({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}) "
        f"hunav=({hunav_pos[0]:.3f}, {hunav_pos[1]:.3f}, {hunav_pos[2]:.3f}) "
        f"far=({far_target[0]:.3f}, {far_target[1]:.3f}, {far_target[2]:.3f}) "
        f"speed={walk_speed:.3f}",
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
