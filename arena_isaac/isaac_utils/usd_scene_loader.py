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

import os
import time

import carb
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import omni.usd
import omni.kit.app

try:
    from isaacsim.core.utils import prims
except ImportError:
    from omni.isaac.core.utils import prims


DEFAULT_CEILING_NAME_PATTERNS = ('ceiling', 'ceil', 'roof')


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


def _is_grscenes_like_usd_path(usd_path: str) -> bool:
    normalized = str(usd_path).replace('\\', '/').lower()
    return (
        'grscenes' in normalized
        or '/commercial_scenes/scenes/' in normalized
        or '/home_scenes/scenes/' in normalized
    ) and normalized.endswith('.usd')


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
                - hide_ceiling_prims: Hide ceiling/roof prims for top-down debug rendering
                - ceiling_name_patterns: Case-insensitive path/name substrings to hide

        Returns:
            bool: Whether loading succeeded
        """
        if load_config is None:
            load_config = {}

        disable_cooking = load_config.get('disable_collision_cooking', True)
        auto_colliders = load_config.get('add_colliders_automatically', True)
        hide_ceiling_config = load_config.get('hide_ceiling_prims', None)
        if hide_ceiling_config is None:
            hide_ceiling_config = os.environ.get('ARENA_ISAAC_HIDE_CEILING_PRIMS')
        if hide_ceiling_config is None:
            hide_ceiling_prims = _is_grscenes_like_usd_path(usd_path)
        else:
            hide_ceiling_prims = _parse_bool(hide_ceiling_config, False)

        ceiling_name_patterns = load_config.get('ceiling_name_patterns') or []
        if not ceiling_name_patterns:
            env_patterns = os.environ.get('ARENA_ISAAC_CEILING_NAME_PATTERNS', '')
            ceiling_name_patterns = [p.strip() for p in env_patterns.split(',') if p.strip()]
        if not ceiling_name_patterns:
            ceiling_name_patterns = list(DEFAULT_CEILING_NAME_PATTERNS)

        post_reference_updates = int(
            load_config.get(
                'post_reference_updates',
                os.environ.get('ARENA_ISAAC_POST_REFERENCE_UPDATES', '1'),
            )
            or 0
        )

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

            # GRScenes-style USDs often defer the expensive composition / asset
            # realization work until the next Kit updates. If we return from the
            # ROS LoadUsdScene service immediately, that hidden cost is paid later
            # by unrelated Pause/Unpause or clock-dependent bringup steps, making
            # those services appear hung. Force a few synchronous Kit updates here
            # so the long scene-load latency is charged to LoadUsdScene itself.
            self._warm_stage_after_reference(scene_prim_path, post_reference_updates)

            hidden_ceiling_count = 0
            if hide_ceiling_prims:
                hidden_ceiling_count = self.hide_ceiling_prims(
                    scene_prim_path=scene_prim_path,
                    name_patterns=ceiling_name_patterns,
                )

            if auto_colliders:
                self.add_colliders(scene_prim_path)

            self._loaded_scenes[scene_prim_path] = {
                'usd_path': usd_path,
                'scale': effective_scale,
                'position': position,
                'orientation': orientation,
                'hidden_ceiling_prims': hidden_ceiling_count,
            }

            carb.log_info(f"[USDSceneLoader] Successfully loaded: {scene_prim_path}")
            return True

        except Exception as e:
            carb.log_error(f"[USDSceneLoader] Failed to load scene: {e}")
            import traceback
            carb.log_error(traceback.format_exc())
            return False

    def _warm_stage_after_reference(self, scene_prim_path: str, updates: int) -> None:
        if updates <= 0:
            return

        app = omni.kit.app.get_app()
        stage = omni.usd.get_context().get_stage()

        for update_idx in range(1, updates + 1):
            started = time.monotonic()
            app.update()
            elapsed = time.monotonic() - started

            child_count = -1
            total_prims = -1
            try:
                if stage is None:
                    stage = omni.usd.get_context().get_stage()
                scene_prim = stage.GetPrimAtPath(scene_prim_path) if stage else None
                if scene_prim and scene_prim.IsValid():
                    child_count = len(list(scene_prim.GetChildren()))
                if stage:
                    total_prims = sum(1 for _ in stage.Traverse())
            except Exception as exc:
                carb.log_warn(f"[USDSceneLoader] Failed to inspect stage after update {update_idx}: {exc}")

            level_fn = carb.log_warn if elapsed >= 5.0 else carb.log_info
            level_fn(
                f"[USDSceneLoader] post-reference update {update_idx}/{updates} took {elapsed:.3f}s "
                f"(children={child_count}, total_prims={total_prims})"
            )

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


    def hide_ceiling_prims(
        self,
        scene_prim_path: str = "/World/Scene",
        name_patterns: list[str] = None,
    ) -> int:
        """Hide ceiling-like prims from rendering without deleting scene geometry.

        This authors a stronger USD visibility opinion (`invisible`) on prims
        whose path/name matches configured substrings.  Physics and collision
        data are left untouched; only Imageable render visibility changes, which
        is enough for the standalone top-down debug camera to see into indoor
        GRScenes rooms.
        """
        if name_patterns is None:
            name_patterns = list(DEFAULT_CEILING_NAME_PATTERNS)

        patterns = [str(pattern).strip().lower() for pattern in name_patterns if str(pattern).strip()]
        if not patterns:
            carb.log_warn("[USDSceneLoader] Ceiling hiding requested with no name patterns; skipped")
            return 0

        stage = omni.usd.get_context().get_stage()
        scene_prim = stage.GetPrimAtPath(scene_prim_path) if stage else None

        if scene_prim is None or not scene_prim.IsValid():
            carb.log_error(f"[USDSceneLoader] Invalid prim for ceiling hiding: {scene_prim_path}")
            return 0

        hidden_count = 0
        matched_examples = []
        for prim in Usd.PrimRange(scene_prim):
            prim_path = str(prim.GetPath())
            prim_path_lower = prim_path.lower()
            prim_name_lower = prim.GetName().lower()
            if not any(pattern in prim_path_lower or pattern in prim_name_lower for pattern in patterns):
                continue
            if not prim.IsA(UsdGeom.Imageable):
                continue

            try:
                imageable = UsdGeom.Imageable(prim)
                imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                hidden_count += 1
                if len(matched_examples) < 5:
                    matched_examples.append(prim_path)
            except Exception as exc:
                carb.log_warn(f"[USDSceneLoader] Failed to hide ceiling prim {prim_path}: {exc}")

        carb.log_info(
            f"[USDSceneLoader] Hid {hidden_count} ceiling-like prims under {scene_prim_path} "
            f"using patterns={patterns}; examples={matched_examples}"
        )
        return hidden_count


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
