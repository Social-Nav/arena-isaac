from isaac_utils.managers.door_manager import DoorManager
from isaac_utils.utils.path import world_path
from pedestrian.simulator.logic.people_manager import PeopleManager

from isaacsim_msgs.srv import DeletePrims

from .utils import Service, on_exception


@on_exception(False)
def remove_person(stage_prefix: str) -> bool:
    person = PeopleManager.get_people_manager().get_person(world_path(stage_prefix))
    if person is None:
        raise ValueError(f"Person with stage prefix {stage_prefix} does not exist.")
    person.destroy()
    return True


@on_exception(False)
def delete_pedestrians_callback(request: DeletePrims.Request, response: DeletePrims.Response):
    results = []
    for path in request.names:
        results.append(remove_person(path))
    response.ret = results
    DoorManager.instance().reset_peds()
    return response


delete_pedestrians_service = Service(
    srv_type=DeletePrims,
    srv_name='isaac/DeletePedestrians',
    callback=delete_pedestrians_callback
)

__all__ = ['delete_pedestrians_service']
