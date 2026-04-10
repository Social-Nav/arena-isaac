"""
LoadUsdScene service - Load a complete USD scene into Isaac Sim

This service is designed for loading pre-built USD scenes like GRScenes,
which contain complete 3D environments with all geometry, materials, etc.
"""

import carb
import omni.usd
from rclpy.qos import QoSProfile

from isaac_utils.usd_scene_loader import USDSceneLoader
from isaacsim_msgs.srv import LoadUsdScene

from .utils import Service, on_exception

profile = QoSProfile(depth=10)

# Global scene loader instance
_scene_loader: USDSceneLoader = None


def get_scene_loader() -> USDSceneLoader:
    """Get or create the global scene loader instance"""
    global _scene_loader
    if _scene_loader is None:
        _scene_loader = USDSceneLoader()
    return _scene_loader


@on_exception(None)
def load_usd_scene(request: LoadUsdScene.Request, response: LoadUsdScene.Response) -> LoadUsdScene.Response:
    """
    Load a  USD scene (for GRScenes)into Isaac Sim.
    """
    loader = get_scene_loader()

    usd_path = request.usd_path
    scene_prim_path = request.scene_prim_path or "/World/Scene"
    scale = request.scale if request.scale > 0 else 1.0

    # Convert arrays to lists
    position = list(request.position) if any(request.position) else None
    orientation = list(request.orientation) if any(request.orientation) else None

    carb.log_info(f"[LoadUsdScene] Loading: {usd_path}")
    carb.log_info(f"[LoadUsdScene] Target prim: {scene_prim_path}")

    # Load configuration
    load_config = {
        'disable_collision_cooking': request.disable_collision_cooking,
        'add_colliders_automatically': request.add_colliders,
    }

    try:
        success = loader.load_scene(
            usd_path=usd_path,
            scene_prim_path=scene_prim_path,
            scale=scale,
            position=position,
            orientation=orientation,
            load_config=load_config,
        )

        if success:
            response.success = True
            response.message = f"Successfully loaded USD scene: {usd_path}"
            response.scene_prim_path = scene_prim_path
            carb.log_info(f"[LoadUsdScene] Success: {scene_prim_path}")
        else:
            response.success = False
            response.message = f"Failed to load USD scene: {usd_path}"
            response.scene_prim_path = ""
            carb.log_error(f"[LoadUsdScene] Failed to load: {usd_path}")

    except Exception as e:
        import traceback
        response.success = False
        response.message = f"Exception loading USD scene: {str(e)}"
        response.scene_prim_path = ""
        carb.log_error(f"[LoadUsdScene] Exception: {e}")
        carb.log_error(traceback.format_exc())

    return response


load_usd_scene_service = Service(
    srv_type=LoadUsdScene,
    srv_name='isaac/LoadUsdScene',
    callback=load_usd_scene,
)


__all__ = ['load_usd_scene_service']
