"""Dynamic USD camera pose conversion for ROS optical-frame TF evidence."""

from __future__ import annotations

import math

from pxr import Gf, Usd, UsdGeom


def simulation_time_to_sec_nanosec(sim_time: float) -> tuple[int, int]:
    """Convert the actual capture clock without changing its timebase."""
    value = float(sim_time)
    if not math.isfinite(value):
        raise ValueError(f"simulation time must be finite, got {sim_time!r}")
    value = max(value, 0.0)
    sec = int(value)
    nanosec = int((value - sec) * 1_000_000_000)
    # Defensive normalization for a platform whose floating multiplication
    # rounds the largest sub-second value to exactly 1e9.
    if nanosec >= 1_000_000_000:
        sec += nanosec // 1_000_000_000
        nanosec %= 1_000_000_000
    return sec, nanosec


def relative_camera_optical_pose(
    base_world: Gf.Matrix4d, camera_world: Gf.Matrix4d
) -> dict[str, list[float]]:
    """Return base-to-camera ROS optical pose from live USD world matrices."""
    camera_in_base = camera_world * base_world.GetInverse()
    translation = camera_in_base.ExtractTranslation()
    rotation = camera_in_base.ExtractRotation() * Gf.Rotation(
        Gf.Vec3d(1.0, 0.0, 0.0), 180.0
    )
    quaternion = rotation.GetQuat()
    imaginary = quaternion.GetImaginary()
    return {
        "translation": [
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ],
        "rotation_xyzw": [
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(quaternion.GetReal()),
        ],
    }


def stage_camera_optical_pose(
    stage: Usd.Stage, base_prim_path: str, camera_prim_path: str
) -> dict[str, list[float]]:
    if stage is None:
        raise ValueError("USD stage is unavailable")
    base_prim = stage.GetPrimAtPath(base_prim_path)
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not base_prim or not base_prim.IsValid():
        raise ValueError(f"invalid base prim: {base_prim_path}")
    if not camera_prim or not camera_prim.IsValid():
        raise ValueError(f"invalid camera prim: {camera_prim_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return relative_camera_optical_pose(
        cache.GetLocalToWorldTransform(base_prim),
        cache.GetLocalToWorldTransform(camera_prim),
    )
