import carb

from isaac_utils.managers.door_manager import DoorManager
from isaac_utils.utils.path import world_path
from pedestrian.simulator.logic.people_manager import PeopleManager

from isaacsim_msgs.srv import DeletePrims

from .utils import Service, on_exception


@on_exception(False)
def remove_person(stage_prefix: str) -> bool:
    # Keep pedestrian deletion idempotent.  Arena cleanup paths may race or
    # retry, and PeopleManager.remove_person() already handles missing entries
    # as a no-op while also clearing manager-owned callbacks/state.
    PeopleManager.get_people_manager().remove_person(world_path(stage_prefix))
    return True


def delete_pedestrians_callback(request: DeletePrims.Request, response: DeletePrims.Response):
    results = []
    for path in request.names:
        results.append(remove_person(path))
    response.ret = results
    try:
        DoorManager.instance().reset_peds()
    except Exception as exc:
        carb.log_warn(f"[DeletePedestrians] Failed to reset pedestrian door state: {exc}")

    carb.log_warn(
        f"[DeletePedestrians] Deleted {sum(1 for ok in response.ret if ok)}/{len(response.ret)} pedestrian(s); "
        "suppressing ROS response to avoid Isaac embedded rclpy response conversion abort"
    )
    raise RuntimeError('DeletePedestrians response intentionally suppressed after deletion')


delete_pedestrians_service = Service(
    srv_type=DeletePrims,
    srv_name='isaac/DeletePedestrians',
    callback=delete_pedestrians_callback
)

__all__ = ['delete_pedestrians_service']
