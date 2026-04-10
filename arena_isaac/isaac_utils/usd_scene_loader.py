"""
USD Scene Loader for Arena

This module provides functionality to load external USD scenes (e.g., GRScenes)
into Isaac Sim within the Arena framework.

Usage:
    from isaac_utils.usd_scene_loader import USDSceneLoader

    loader = USDSceneLoader()
    loader.load_scene("/path/to/scene.usd", "/World/Scene")
    loader.add_colliders("/World/Scene")
"""

import carb
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import omni.usd

try:
    from isaacsim.core.utils import prims
except ImportError:
    from omni.isaac.core.utils import prims


class USDSceneLoader:
    """
    USD Scene Loader - loads external USD scenes into Isaac Sim.

    Supports loading pre-built USD scenes (e.g., GRScenes) with
    automatic scale detection, physics collider addition, etc.
    """

    def __init__(self, world_prim_path: str = "/World"):
        self.world_prim_path = world_prim_path
        self._loaded_scenes = {}

    def load_scene(
        self,
        usd_path: str,
        scene_prim_path: str = "/World/Scene",
        scale: float = 1.0,
        position: list = None,
        orientation: list = None,
        load_config: dict = None,
    ) -> bool:
        """
        Load a USD scene into Isaac Sim.

        Args:
            usd_path: USD file path (absolute)
            scene_prim_path: Target prim path, default "/World/Scene"
            scale: Scale factor, default 1.0
            position: Position [x, y, z], default [0, 0, 0]
            orientation: Quaternion [x, y, z, w], default [0, 0, 0, 1]
            load_config: Loading configuration options
                - disable_collision_cooking: Disable collision cooking (faster loading)
                - add_colliders_automatically: Auto-add colliders

        Returns:
            bool: Whether loading succeeded
        """
        if load_config is None:
            load_config = {}

        disable_cooking = load_config.get('disable_collision_cooking', True)
        auto_colliders = load_config.get('add_colliders_automatically', True)

        try:
            usd_file_path = Path(usd_path)
            if not usd_file_path.exists():
                carb.log_error(f"[USDSceneLoader] File not found: {usd_path}")
                return False

            carb.log_info(f"[USDSceneLoader] Loading USD scene: {usd_path}")

            # Auto-detect metersPerUnit for scaling (when scale=1.0)
            # GRScenes files use metersPerUnit=0.01 (cm), need conversion to meters
            effective_scale = scale
            if scale == 1.0:
                try:
                    root_layer = Sdf.Layer.FindOrOpen(str(usd_file_path))
                    if root_layer:
                        mpu = root_layer.customLayerData.get('metersPerUnit', 1.0)
                        if mpu != 1.0:
                            effective_scale = mpu
                            carb.log_info(
                                f"[USDSceneLoader] Auto-scale from metersPerUnit={mpu} "
                                f"(cm->m conversion for {usd_file_path.name})"
                            )
                except Exception as _e:
                    carb.log_warn(f"[USDSceneLoader] Failed to auto-detect scale: {_e}")

            if disable_cooking:
                self._disable_collision_cooking()

            # Create root Xform prim with correct transform
            import numpy as np
            _pos = np.array(position if position else [0.0, 0.0, 0.0])
            scene_prim = prims.create_prim(
                scene_prim_path,
                "Xform",
                translation=_pos,
                scale=np.array([effective_scale, effective_scale, effective_scale]),
            )

            # Add USD reference (content inherits parent xformOp scale)
            scene_prim.GetReferences().AddReference(str(usd_file_path))
            carb.log_info(
                f"[USDSceneLoader] Scene referenced with scale={effective_scale:.4f}, "
                f"pos={list(_pos)}"
            )

            if auto_colliders:
                self.add_colliders(scene_prim_path)

            self._loaded_scenes[scene_prim_path] = {
                'usd_path': usd_path,
                'scale': effective_scale,
                'position': position,
                'orientation': orientation,
            }

            carb.log_info(f"[USDSceneLoader] Successfully loaded: {scene_prim_path}")
            return True

        except Exception as e:
            carb.log_error(f"[USDSceneLoader] Failed to load scene: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return False

    def _disable_collision_cooking(self):
        """Disable UJITSO collision cooking for faster loading."""
        try:
            import carb
            import carb.settings
            import omni.physx.bindings._physx as physx_bindings

            settings = carb.settings.get_settings()
            settings.set_bool(physx_bindings.SETTING_UJITSO_COLLISION_COOKING, False)

            carb.log_info("[USDSceneLoader] Disabled UJITSO collision cooking")
        except Exception as e:
            carb.log_warn(f"[USDSceneLoader] Failed to disable collision cooking: {e}")


    def add_colliders(
        self,
        scene_prim_path: str = "/World/Scene",
        approx_type: str = "convexHull",
        excluded_paths: list = None,
        ground_only: bool = True,
    ) -> int:
        """
        Add physics colliders to meshes in the scene.

        Args:
            scene_prim_path: Scene prim path
            approx_type: Collision approximation type
            excluded_paths: Prim paths to exclude
            ground_only: Only add colliders to ground meshes (recommended for GRScenes)

        Returns:
            int: Number of meshes with colliders added
        """
        if excluded_paths is None:
            excluded_paths = []

        # Complex objects to exclude from collision
        default_excluded = [
            '/person/', '/chair/', '/table/', '/cabinet/', '/box/',
            '/book/', '/bottle/', '/cup/', '/keyboard/', '/monitor/',
            '/pillow/', '/bed/', '/sink/', '/tray/', '/pen/', '/pot/',
        ]
        excluded_paths = excluded_paths + default_excluded

        ground_patterns = ['/ground/', '/floor/', '/Ground', '/Floor']

        stage = omni.usd.get_context().get_stage()
        scene_prim = stage.GetPrimAtPath(scene_prim_path)

        if not scene_prim.IsValid():
            carb.log_error(f"[USDSceneLoader] Invalid prim: {scene_prim_path}")
            return 0

        collider_count = 0
        excluded_count = 0
        skipped_count = 0

        for prim in Usd.PrimRange(scene_prim):
            prim_path = str(prim.GetPath())

            if any(excluded in prim_path.lower() for excluded in excluded_paths):
                excluded_count += 1
                continue

            if ground_only:
                if not any(pattern.lower() in prim_path.lower() for pattern in ground_patterns):
                    skipped_count += 1
                    continue

            if prim.IsA(UsdGeom.Mesh):
                try:
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI.Apply(prim)

                    collision_api = UsdPhysics.CollisionAPI(prim)
                    collision_api.CreateCollisionEnabledAttr(True)

                    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                        mesh_collision_api.CreateApproximationAttr().Set(approx_type)

                    collider_count += 1

                except Exception as e:
                    carb.log_warn(f"[USDSceneLoader] Failed to add collider to {prim_path}: {e}")

        carb.log_info(
            f"[USDSceneLoader] Added colliders to {collider_count} meshes "
            f"(excluded {excluded_count}, skipped {skipped_count})"
        )
        return collider_count

    def unload_scene(self, scene_prim_path: str = "/World/Scene") -> bool:
        """Unload a scene by removing its prim."""
        try:
            stage = omni.usd.get_context().get_stage()
            scene_prim = stage.GetPrimAtPath(scene_prim_path)

            if not scene_prim.IsValid():
                carb.log_warn(f"[USDSceneLoader] Scene not found: {scene_prim_path}")
                return False

            stage.RemovePrim(scene_prim.GetPath())

            if scene_prim_path in self._loaded_scenes:
                del self._loaded_scenes[scene_prim_path]

            carb.log_info(f"[USDSceneLoader] Unloaded scene: {scene_prim_path}")
            return True

        except Exception as e:
            carb.log_error(f"[USDSceneLoader] Failed to unload scene: {e}")
            return False


# Convenience function
def load_grscenes_scene(
    scene_id: str,
    grscenes_base_path: str = "/home/pggg/InternUtopia/internutopia/assets/scenes/GRScenes-100",
    scene_type: str = "commercial_scenes",
    variant: str = "navigation",
    scene_prim_path: str = "/World/Scene",
) -> bool:
    """
    Convenience function: load a GRScenes scene.

    Args:
        scene_id: Scene ID (e.g., "MV4AFHQKTKJZ2AABAAAAADQ8_usd")
        grscenes_base_path: GRScenes base path
        scene_type: Scene type ("commercial_scenes" or "home_scenes")
        variant: Scene variant ("navigation" or "interaction")
        scene_prim_path: Scene prim path

    Returns:
        bool: Whether loading succeeded
    """
    from pathlib import Path

    usd_filename = f"start_result_{variant}.usd"
    usd_path = str(
        Path(grscenes_base_path) / scene_type / "scenes" / scene_id / usd_filename
    )

    loader = USDSceneLoader()
    load_config = {
        'disable_collision_cooking': True,
        'add_colliders_automatically': True,
    }

    return loader.load_scene(usd_path, scene_prim_path, load_config=load_config)
