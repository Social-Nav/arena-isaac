import collections.abc
import carb

from .utils import Service

services: list[Service] = []

def _safe_import_service(module_name, service_name):
    try:
        from importlib import import_module
        module = import_module(f".{module_name}", package=__name__)
        service = getattr(module, service_name)
        services.append(service)
    except Exception as e:
        carb.log_error(f"Failed to import service {service_name} from {module_name}: {e}")

_safe_import_service("DeletePedestrians", "delete_pedestrians_service")
_safe_import_service("DeletePrims", "delete_prims_service")
_safe_import_service("EditPrims", "edit_prims_service")
_safe_import_service("GetPrims", "get_prims_service")
_safe_import_service("LoadUsdScene", "load_usd_scene_service")
_safe_import_service("NavigatePedestrians", "navigate_pedestrians_service")
_safe_import_service("SpawnDoors", "spawn_doors_service")
_safe_import_service("SpawnElevators", "spawn_elevators_service")
_safe_import_service("SpawnFloors", "spawn_floors_service")
_safe_import_service("SpawnPrims", "spawn_prims_service")
_safe_import_service("SpawnPedestrians", "spawn_pedestrians_service")
_safe_import_service("SpawnUrdf", "spawn_urdf_service")
_safe_import_service("SpawnUsd", "spawn_usd_service")
_safe_import_service("SpawnUsdRobot", "spawn_usd_robot_service")
_safe_import_service("SpawnWalls", "spawn_walls_service")

__all__ = ["services"]
