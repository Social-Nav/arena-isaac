"""
Pixel -> map reprojection for the social-replan pipeline.

Pure numpy, no ROS / no Isaac deps — so it can be unit-tested offline against
snapshot files (<cam>_depth.npy + <cam>_camera.json produced by
snapshot_capturer.py).

Given a pixel (u, v) chosen by the yielding selector on an ego camera, plus that
camera's raw metric depth and intrinsics/extrinsics (all same-frame), compute the
world-space 3D point. In this project the Isaac world frame coincides with the
Nav2 "map" frame (pose_to_tf publishes map->odom straight from the Isaac world
pose), so the returned (x, y, z) is directly a map-frame goal.

Conventions
-----------
Isaac Replicator returns `cameraViewTransform` as a row-major 4x4 that maps a
world-space ROW vector to camera space:  p_cam_row = p_world_row @ V.
Hence camera->world for a row vector is:  p_world_row = p_cam_row @ inv(V).

Isaac/USD camera axes: +X right, +Y up, -Z forward (looking down -Z). A pinhole
ray for pixel (u, v) with depth Z (distance_to_camera, along the optical axis)
is therefore, in camera space:
    x_cam = (u - cx) / fx * Z
    y_cam = -(v - cy) / fy * Z      # image v grows downward, camera +Y is up
    z_cam = -Z                      # camera looks down -Z

The exact sign convention is the most fragile part; `SIGN_MODE` lets the offline
test flip it without touching call sites.
"""

import json

import numpy as np


# The optical-axis / image-axis sign convention. "isaac" matches USD camera
# axes (+X right, +Y up, -Z forward). If offline validation shows the point
# mirrored/behind, switch to "cv" (standard OpenCV: +Z forward, +Y down).
SIGN_MODE = "isaac"


def _load_camera(camera_json_path):
    with open(camera_json_path) as f:
        cam = json.load(f)
    K = np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
    view = np.asarray(cam["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    return K, view, int(cam["width"]), int(cam["height"])


def _depth_at(depth, u, v, win=2):
    """Median finite depth in a small window around (u, v); None if all invalid."""
    h, w = depth.shape[:2]
    u = int(round(u)); v = int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float64)
    finite = patch[np.isfinite(patch) & (patch > 0)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def pixel_to_map(u, v, depth_npy_path, camera_json_path,
                 min_z=0.2, max_z=30.0):
    """Reproject pixel (u, v) to a world/map-frame (x, y, z) point.

    Returns dict {ok, reason, map_point:[x,y,z], depth, cam_point:[...]}.
    ok=False with a reason when depth is invalid/out of range.
    """
    depth = np.load(depth_npy_path)
    K, view, width, height = _load_camera(camera_json_path)

    Z = _depth_at(depth, u, v)
    if Z is None:
        return {"ok": False, "reason": "no valid depth at pixel", "map_point": None}
    if not (min_z <= Z <= max_z):
        return {"ok": False, "reason": f"depth {Z:.2f}m out of [{min_z},{max_z}]",
                "map_point": None, "depth": Z}

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) / fx * Z
    y = (v - cy) / fy * Z
    if SIGN_MODE == "isaac":
        # +X right, +Y up (flip image-down v), -Z forward.
        cam_pt = np.array([x, -y, -Z, 1.0])
    else:  # "cv": +X right, +Y down, +Z forward
        cam_pt = np.array([x, y, Z, 1.0])

    # camera->world for a ROW vector: p_world_row = p_cam_row @ inv(view)
    world_row = cam_pt @ np.linalg.inv(view)
    wp = world_row[:3] / (world_row[3] if world_row[3] != 0 else 1.0)

    return {
        "ok": True,
        "reason": "",
        "depth": Z,
        "cam_point": cam_pt[:3].tolist(),
        "map_point": [float(wp[0]), float(wp[1]), float(wp[2])],
    }


# ── offline self-test / debug ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline pixel->map reprojection test")
    parser.add_argument("--depth", required=True, help="<cam>_depth.npy")
    parser.add_argument("--camera", required=True, help="<cam>_camera.json")
    parser.add_argument("--u", type=int, required=True)
    parser.add_argument("--v", type=int, required=True)
    parser.add_argument("--sign", default=SIGN_MODE, choices=["isaac", "cv"])
    args = parser.parse_args()

    SIGN_MODE = args.sign
    res = pixel_to_map(args.u, args.v, args.depth, args.camera)
    print(json.dumps(res, indent=2, ensure_ascii=False))
