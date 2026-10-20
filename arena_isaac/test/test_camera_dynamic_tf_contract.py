import math

import pytest
from pxr import Gf

from isaac_utils.camera_dynamic_tf import (
    relative_camera_optical_pose,
    simulation_time_to_sec_nanosec,
)


def test_identity_usd_camera_gets_ros_optical_x_pi_correction():
    pose = relative_camera_optical_pose(Gf.Matrix4d(1.0), Gf.Matrix4d(1.0))
    assert pose["translation"] == [0.0, 0.0, 0.0]
    x, y, z, w = pose["rotation_xyzw"]
    assert math.isclose(abs(x), 1.0, abs_tol=1e-9)
    assert math.isclose(y, 0.0, abs_tol=1e-9)
    assert math.isclose(z, 0.0, abs_tol=1e-9)
    assert math.isclose(w, 0.0, abs_tol=1e-9)


def test_relative_pose_uses_live_world_matrices_and_cancels_base_translation():
    base = Gf.Matrix4d(1.0)
    base.SetTranslate(Gf.Vec3d(10.0, 20.0, 30.0))
    camera = Gf.Matrix4d(1.0)
    camera.SetTranslate(Gf.Vec3d(10.5, 19.5, 31.0))
    pose = relative_camera_optical_pose(base, camera)
    assert all(
        math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(pose["translation"], [0.5, -0.5, 1.0])
    )


def test_dynamic_tf_and_manual_image_share_actual_capture_stamp_conversion():
    assert simulation_time_to_sec_nanosec(4.033333543688059) == (4, 33333543)
    assert simulation_time_to_sec_nanosec(-1.0) == (0, 0)


def test_timestamp_conversion_normalizes_second_boundary_without_overflow():
    before_two = math.nextafter(2.0, 0.0)
    sec, nanosec = simulation_time_to_sec_nanosec(before_two)
    assert sec == 1
    assert 0 <= nanosec < 1_000_000_000
    assert simulation_time_to_sec_nanosec(2.0) == (2, 0)
    with pytest.raises(ValueError, match="must be finite"):
        simulation_time_to_sec_nanosec(float("nan"))
