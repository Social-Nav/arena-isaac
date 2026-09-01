#!/usr/bin/env python3
"""Unified pipeline: convert raw npy to PNG/parquet, compute pixel goals, write meta.

Combines raw_data_convert.py and gen_pixel_goal_labels.py into one pass:
  1. Load npy chunks → write RGB/depth PNGs + RGB preview mp4
  2. Compute pixel goals in memory (no PNG re-read)
  3. Write labeled parquet with action/pose/goal columns
  4. Write meta/ (info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl)
  5. Clean up npy chunks

Faster than running the two scripts separately (saves one depth PNG write+read cycle),
and guarantees consistency (same depth data for PNG and labeling).

Camera height/pitch are fixed constants -- see CAMERA_HEIGHT_CM / CAMERA_PITCH_DEG below.

Usage:
    # Every world under the grscenes root, skipping episodes already converted:
    python process_raw_to_dataset.py --data-path src/social_gen/traj_data/grscenes

    # One world:
    python process_raw_to_dataset.py --data-path src/social_gen/traj_data/grscenes/grscenes_1

Re-runnable: an episode whose raw npy are gone is already converted and is skipped, so
pointing this at the root repeatedly only picks up new data. It ends with a per-world
summary and, on failure, names the episode and the reason.
"""

import argparse
import glob
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────────────
# CAMERA MOUNT -- EDIT HERE IF THE ROBOT'S CAMERA MOVES
#
# Where the camera sits on the robot and how it is angled. Fixed on purpose: the mount is
# decided once per robot and does not vary run to run, so these are not CLI flags and are
# not measured from the data.
#
# All values are UNSCALED, exactly as authored in the USD. The robot's `scale` is read from
# arena_robots/robots/<ROBOT_MODEL>/model_params.yaml and applied on top, so it lives in
# exactly one place: change the robot's scale there and this follows. Do NOT copy the scale
# here or pre-multiply the offsets by it -- two copies drift apart the moment one is edited.
#
# Verified against robot_usds/humanoid/Ai2_Bot2/Ai2_Bot2.usda, prim
# head_link2/head_camera/d435/head_camera, on 2026-08-24:
#   translation (0.3506, 0, 1.6453) m  -> the offsets below
#   rotation matched T_ROBOT2CAMERA @ pitch_matrix(15) to 4e-6 once the logger's
#   USD->ROS optical conversion is applied. The USD has since been re-pitched to 30 deg
#   (CAMERA_PITCH_DEG below); re-verify against the prim if the labels look tilted.
#   USD->ROS optical conversion (right-multiply diag([1,-1,-1])) is applied
#   scale 0.8 from model_params.yaml, applied via the parent Xform to the whole reference
#
# ASSUMPTION: head_joint1/head_joint2 do not move during collection, which makes this a
# constant. Both are authored with state:angular:physics:position = 0 and stiff drives. If
# the head is ever actuated mid-episode, the offset stops being constant and has to come
# from the recorded pose instead.
ROBOT_MODEL = "Ai2_Bot2"    # which model_params.yaml to read `scale` from
CAMERA_FORWARD_M = 0.3506   # +x, ahead of the robot origin (unscaled)
CAMERA_LATERAL_M = 0.0      # +y, left of the robot origin (unscaled)
CAMERA_HEIGHT_M = 1.6453    # +z, above the robot origin (unscaled)
CAMERA_PITCH_DEG = 30       # positive downward
# ─────────────────────────────────────────────────────────────────────────────────────


def _read_robot_scale(model=None):
    """`scale` from arena_robots/robots/<model>/model_params.yaml.

    The scale is authored once for the robot and already drives the USD xformOp:scale,
    sensor TFs, footprint and robot_radius, so reading it is what keeps the camera offsets
    consistent with the robot that was actually spawned. Falls back to 1.0 with a warning
    when the file cannot be found or parsed -- a wrong scale is a silent geometry error, so
    it says so rather than guessing 0.8.
    """
    model = model or ROBOT_MODEL
    rel = os.path.join("robots", model, "model_params.yaml")
    candidates = []
    try:  # installed share dir, when running inside a sourced workspace
        from ament_index_python.packages import get_package_share_directory

        candidates.append(os.path.join(get_package_share_directory("arena_robots"), rel))
    except Exception:
        pass
    # source tree: .../Arena/arena_isaac/arena_isaac/arena_isaac/data_logging/postprocess
    here = os.path.dirname(os.path.abspath(__file__))
    arena = os.path.abspath(os.path.join(here, *([os.pardir] * 5)))
    candidates.append(os.path.join(arena, "arena_robots", "arena_robots", rel))

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f) or {}
            scale = float(data.get("scale", 1.0))
            if scale > 0:
                return scale, path
            print(f"  WARN {path} has non-positive scale {scale}; using 1.0", file=sys.stderr)
            return 1.0, path
        except Exception as e:  # noqa: BLE001
            print(f"  WARN could not read scale from {path}: {e}", file=sys.stderr)

    print(f"  WARN no model_params.yaml for '{model}' on any known path; assuming scale 1.0. "
          f"Camera offsets will be wrong if the robot is actually scaled.", file=sys.stderr)
    return 1.0, None


ROBOT_SCALE, _ROBOT_SCALE_SRC = _read_robot_scale()

# Effective height above the floor, used to name output dirs and columns
# (observation.images.rgb.132cm_30deg). Derived, not hand-maintained, so it cannot drift
# from the geometry the labels were computed with.
CAMERA_HEIGHT_CM = round(CAMERA_HEIGHT_M * ROBOT_SCALE * 100)

# Action constants
NO_ACTION, STOP, FORWARD, TURN_LEFT, TURN_RIGHT = -1, 0, 1, 2, 3
ACTION_VALUES = (NO_ACTION, STOP, FORWARD, TURN_LEFT, TURN_RIGHT)

# Coordinate transforms (from gen_pixel_goal_labels.py)
T_ROBOT2CAMERA = np.array(
    [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)

CHUNK_SIZE = 1000
EPISODE_PAT = re.compile(r"episode_(\d+)")
CHUNK_PAT = re.compile(r"chunk_(\d+)\.npy")


def mount_matrix(scale=ROBOT_SCALE):
    """Robot origin -> camera mount point. Translation only, scaled by `scale`.

    This is the piece that makes cam_to_robot_footprint() actually land on the robot rather than on
    the lens. T_ROBOT2CAMERA and pitch_matrix are both pure rotations, so without this the
    inverse is a pure rotation too and leaves the translation untouched -- the "robot"
    pose comes out identical to the camera pose, off by the mount offset (0.28 m forward
    here). Silent, because a pose that is wrong by a fixed offset still looks plausible.
    """
    T = np.eye(4)
    T[0, 3] = CAMERA_FORWARD_M * scale
    T[1, 3] = CAMERA_LATERAL_M * scale
    T[2, 3] = CAMERA_HEIGHT_M * scale
    return T


def pitch_matrix(deg):
    rad = np.radians(deg)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(-rad), -np.sin(-rad), 0.0],
            [0.0, np.sin(-rad), np.cos(-rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def cam_to_robot_footprint(pitch_deg):
    """Camera pose -> robot pose. Includes the mount translation, so the result is the
    robot's own frame, not the lens rotated into robot axes."""
    return np.linalg.inv(mount_matrix() @ T_ROBOT2CAMERA @ pitch_matrix(pitch_deg))


def as_4x4(cell):
    """Parquet cell or flat list → (4,4) float array."""
    arr = np.asarray(cell, dtype=object) if np.asarray(cell).dtype == object else np.asarray(cell)
    if arr.dtype == object:
        arr = np.stack([np.asarray(row, dtype=np.float64) for row in arr])
    return np.asarray(arr, dtype=np.float64).reshape(4, 4)


def load_chunks(subdir: str):
    """Concatenate chunk_XXXX.npy in index order → (frames, paths)."""
    if not os.path.isdir(subdir):
        return None, []
    chunks = sorted(
        (int(m.group(1)), os.path.join(subdir, f))
        for f in os.listdir(subdir)
        if (m := CHUNK_PAT.match(f))
    )
    if not chunks:
        return None, []
    return np.concatenate([np.load(p) for _, p in chunks], axis=0), [p for _, p in chunks]


def load_flat_frames(videos_dir: str, src_ep: int):
    """Load the flat per-frame npy for one episode → (rgb, depth, paths).

    Reads videos/episode_XXXXXX_N.npy written by the logger's world-based layout. Each
    file is a dict {"rgb", "depth"} because the filename carries no modality field. N is
    the frame index within the episode and is ordered numerically, not lexically, so
    frame 10 does not sort before frame 2.

    allow_pickle is required to read a dict out of an npy. Only ever point this at your
    own collected data -- a pickled npy from an untrusted source executes on load.
    """
    if not os.path.isdir(videos_dir):
        return None, None, []
    pat = re.compile(rf"episode_{src_ep:06d}_(\d+)\.npy$")
    found = sorted(
        (int(m.group(1)), os.path.join(videos_dir, f))
        for f in os.listdir(videos_dir)
        if (m := pat.match(f))
    )
    if not found:
        return None, None, []

    rgb, depth = [], []
    for _, p in found:
        rec = np.load(p, allow_pickle=True).item()
        rgb.append(rec["rgb"])
        depth.append(rec["depth"])
    return np.stack(rgb), np.stack(depth), [p for _, p in found]


RGB_EXT = "jpg"       # lossy is fine for appearance; ~6-10x smaller than PNG
RGB_QUALITY = 95      # visually lossless in practice; drop to 85 to halve the size again
DEPTH_EXT = "png"     # MUST stay lossless: uint16 metric depth, JPEG would corrupt it


def write_rgb_images(frames: np.ndarray, out_dir: str, ep_id: int) -> int:
    """Write RGB frames as JPEG.

    JPEG only for RGB. Depth stays PNG (see write_depth_pngs): depth pixels are uint16
    millimetre measurements, and JPEG is both lossy and 8-bit, so encoding depth as JPEG
    would quantise metres into 256 buckets and add ringing around edges -- the reprojection
    would then be wrong everywhere while the image still looked plausible.
    """
    os.makedirs(out_dir, exist_ok=True)
    if frames.dtype != np.uint8:
        hi = float(frames.max())
        frames = (frames * (255.0 if hi <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
    for i, frame in enumerate(frames):
        iio.imwrite(
            os.path.join(out_dir, f"episode_{ep_id:06d}_{i}.{RGB_EXT}"),
            frame,
            quality=RGB_QUALITY,
        )
    return len(frames)


def write_depth_pngs(frames: np.ndarray, out_dir: str, ep_id: int, depth_scale: float) -> int:
    """Write 16-bit depth PNGs, depth_scale PNG units per metre (1000 = millimetres).

    Input may be either float metres or an already-scaled integer. The logger's
    process_depth_for_png hands us uint16 millimetres, so applying depth_scale to that
    would multiply a second time: 2000 mm * 1000 overflows 65535 and every pixel
    saturates, silently flattening the whole depth map. Integer input is therefore taken
    as already being in PNG units and passed through.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames = np.asarray(frames)
    if np.issubdtype(frames.dtype, np.integer):
        scaled = np.clip(frames, 0, 65535).astype(np.uint16)
    else:
        scaled = np.clip(frames.astype(np.float64) * depth_scale, 0, 65535).astype(np.uint16)
    for i, frame in enumerate(scaled):
        iio.imwrite(os.path.join(out_dir, f"episode_{ep_id:06d}_{i}.png"), frame)
    return len(scaled)


def write_preview(frames: np.ndarray, preview_dir: str, ep_id: int, kind: str, fps: int):
    """8-bit mp4 for eyeballing."""
    os.makedirs(preview_dir, exist_ok=True)
    path = os.path.join(preview_dir, f"episode_{ep_id:06d}_{kind}.mp4")
    # Normalize to uint8
    if frames.dtype != np.uint8:
        lo, hi = float(frames.min()), float(frames.max())
        frames = (
            ((frames - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros(frames.shape, dtype=np.uint8)
        )
    if frames.ndim == 3:  # grayscale → 3 channels
        frames = np.stack([frames] * 3, axis=-1)
    iio.imwrite(path, frames, fps=fps, codec="libx264")


def intrinsics(width, height, hfov_deg):
    """fx, fy, cx, cy assumed from image size and FOV. Fallback only.

    Assumes square pixels and the principal point at the exact image centre. Prefer
    intrinsics_from_params(), which reads what the camera actually reported.
    """
    f = 0.5 * width / np.tan(0.5 * np.radians(hfov_deg))
    return f, f, width / 2.0, height / 2.0


def intrinsics_from_params(params_list, width, height, hfov_deg):
    """fx, fy, cx, cy from the recorded observation.camera_intrin (flat 3x3 K).

    The logger derives these from the camera's own cameraProjection matrix, including a
    principal point taken off the matrix rather than assumed to be the image centre --
    with an aperture offset set on the USD camera, cx/cy sit tens of pixels off centre and
    recomputing them from --hfov silently breaks every reprojection while the images still
    look fine. So use the recorded values and keep the --hfov formula only for older data
    that lacks the field.
    """
    K = params_list[0].get("observation.camera_intrin") if params_list else None
    if K is None:
        print("  WARN no observation.camera_intrin, falling back to --hfov "
              f"{hfov_deg} and centre principal point", file=sys.stderr)
        return intrinsics(width, height, hfov_deg)

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if not (fx > 0 and fy > 0):
        print(f"  WARN recorded intrinsics have non-positive focal length "
              f"(fx={fx}, fy={fy}), falling back to --hfov {hfov_deg}", file=sys.stderr)
        return intrinsics(width, height, hfov_deg)
    return float(fx), float(fy), float(cx), float(cy)


def discretize_actions(xyyaw, fwd_thresh=0.05, yaw_thresh=np.radians(5.0)):
    """Label each frame with the motion that ARRIVED at it."""
    n = len(xyyaw)
    actions = np.full(n, NO_ACTION, dtype=np.int64)
    if n < 2:
        return actions

    d_xy = np.diff(xyyaw[:, :2], axis=0)
    d_yaw = np.arctan2(np.sin(np.diff(xyyaw[:, 2])), np.cos(np.diff(xyyaw[:, 2])))

    heading = xyyaw[:-1, 2]
    forward = d_xy[:, 0] * np.cos(heading) + d_xy[:, 1] * np.sin(heading)

    turning = np.abs(d_yaw) > yaw_thresh
    step = np.where(turning, np.where(d_yaw > 0, TURN_LEFT, TURN_RIGHT), FORWARD)
    # Not turning and not translating enough -> STOP. This used to assign FORWARD, which
    # made the whole line a no-op (the not-turning branch above is already FORWARD) and
    # meant STOP was never emitted at all: every stationary frame was labelled FORWARD.
    step = np.where(~turning & (forward <= fwd_thresh), STOP, step)

    actions[1:] = step
    return actions


def farthest_visible_goal(
    extrinsics,
    frame_idx,
    depth,
    fx,
    fy,
    cx,
    cy,
    lookahead=5.0,
    margin=0.2,
    min_ground_dist=1.0,
    ground_z=0.0,
    flip_v=False,
    depth_is_euclidean=False,
    robot_xy=None,
):
    """Farthest visible pixel goal (footprint projection).

    lookahead: how far ahead to search, in METRES of path travelled -- not frames. A frame
    count carries no physical meaning on its own: it has to be multiplied by speed/fps to
    become a distance, so the same number means a different reach at every speed and frame
    rate (64 frames is 0.7 m at 0.3 m/s and 1.7 m at 0.8 m/s, both at 30 fps). The camera's
    near blind spot is a fixed distance, so the search bound has to be one too. Walking the
    path and accumulating real displacement also self-adjusts when the robot slows, stops to
    yield, or accelerates -- frames spent standing still add nothing to the distance instead
    of eating the budget.

    robot_xy: (n, 2) ground positions of the ROBOT, i.e. extrinsics @ cam_to_robot_footprint()
    translations, not the camera's own. Projecting the camera centre instead makes the
    goal drift with mount height and pitch (the two partly cancel), which is the
    difference between a few px of error and ~100 px. It is also the wrong target: the
    label should be a spot the robot can drive onto, not a point in mid-air at lens
    height. Required -- passing None falls back to the camera track and warns.
    """
    h, w = depth.shape
    T_w2c = np.linalg.inv(extrinsics[frame_idx])
    if robot_xy is None:
        print("  WARN farthest_visible_goal got no robot_xy; projecting CAMERA centres, "
              "which biases every goal", file=sys.stderr)
        robot_xy = extrinsics[:, :2, 3]
    here_xy = robot_xy[frame_idx]

    best = None
    travelled = 0.0
    for j in range(frame_idx + 1, len(extrinsics)):
        # Arc length along the path, not straight-line distance from `here_xy`: it is
        # monotonic, so once it passes the budget nothing further can be in range and the
        # loop can stop. Straight-line distance is not monotonic (a robot that curves back
        # gets closer again), which is why the min_ground_dist test below uses `continue`.
        travelled += float(np.hypot(*(robot_xy[j] - robot_xy[j - 1])))
        if travelled > lookahead:
            break
        # Footprint: the robot's ground position, forced onto the floor plane.
        p_world = np.array([robot_xy[j][0], robot_xy[j][1], ground_z, 1.0])

        if np.hypot(*(p_world[:2] - here_xy)) < min_ground_dist:
            continue

        p_cam = T_w2c @ p_world
        z = p_cam[2]
        if z <= 1e-6:
            continue

        u_raw = fx * p_cam[0] / z + cx
        v_raw = fy * p_cam[1] / z + cy
        if flip_v:
            v_raw = h - 1 - v_raw

        u = int(round(u_raw))
        v = int(round(v_raw))
        if not (0 <= u < w and 0 <= v < h):
            continue

        measured = depth[v, u]
        if measured <= 0:
            continue
        dist = np.linalg.norm(p_cam[:3]) if depth_is_euclidean else z
        if dist > measured + margin:
            continue

        best = ((u, v), j - frame_idx)

    return best if best is not None else (None, -1)


def build_schema(setting, peds_type):
    """Parquet schema for labeled data."""
    # int32 for the content columns, int64 for the LeRobot bookkeeping columns below. That
    # split is what the reference format uses; action/goal/frame-id values are all far inside
    # int32 range (pixels < 2^15, frame ids < 2^20).
    fields = [
        pa.field("action", pa.int32()),
        pa.field(f"pose.{setting}", pa.list_(pa.list_(pa.float32(), 4), 4)),
        pa.field(f"goal.{setting}", pa.list_(pa.int32(), 2)),
        pa.field(f"relative_goal_frame_id.{setting}", pa.int32()),
    ]
    if peds_type is not None:
        fields.append(pa.field("pose_peds", peds_type))
    fields += [
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        # frame_index restarts each episode; index is continuous across the dataset. Filled
        # with the per-episode value here and corrected to the dataset-wide offset by
        # _sync_index_column once every episode's length is known.
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ]
    return pa.schema(fields)


def process_episode(ep_dir, ep_id, video_chunk, data_chunk, task_index, cfg, raw=None):
    """Load npy → write PNGs/parquet → compute goals → return stats.

    raw: (videos_dir, src_ep, params_path) for the flat world-based layout, where frames
    live in <scene>/videos/episode_XXXXXX_N.npy and params in
    <scene>/data/episode_XXXXXX.json. None means the legacy per-episode-dir layout
    (ep_dir/{rgb_videos,depth_videos}/chunk_XXXX.npy + ep_dir/params.json).
    """
    # 1. Load npy
    if raw is not None:
        videos_dir, src_ep, params_path = raw
        rgb_frames, depth_frames, frame_npy = load_flat_frames(videos_dir, src_ep)
        rgb_npy, depth_npy = frame_npy, []
    else:
        rgb_frames, rgb_npy = load_chunks(os.path.join(ep_dir, "rgb_videos"))
        depth_frames, depth_npy = load_chunks(os.path.join(ep_dir, "depth_videos"))
        params_path = os.path.join(ep_dir, "params.json")

    if rgb_frames is None or depth_frames is None:
        return None  # already converted

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"missing {params_path}")

    with open(params_path) as f:
        params_list = json.load(f)

    n_frames = len(params_list)
    if len(rgb_frames) != n_frames or len(depth_frames) != n_frames:
        print(
            f"  WARN ep{ep_id}: params {n_frames} vs rgb {len(rgb_frames)} vs depth {len(depth_frames)}",
            file=sys.stderr,
        )

    extrinsics = np.stack([as_4x4(p["observation.camera_state"]) for p in params_list])

    # 2. Write images. rgb and depth go to sibling dirs named for the camera setting, with
    # the rgb preview mp4 alongside them:
    #   videos/chunk-XXX/observation.images.rgb.132cm_30deg/
    #   videos/chunk-XXX/observation.images.depth.132cm_30deg/
    #   videos/chunk-XXX/preview/episode_XXXXXX_rgb.mp4
    # Synchronous: these return only once every file is on disk.
    setting = f"{CAMERA_HEIGHT_CM}cm_{CAMERA_PITCH_DEG}deg"
    rgb_dir = os.path.join(video_chunk, f"observation.images.rgb.{setting}")
    depth_dir = os.path.join(video_chunk, f"observation.images.depth.{setting}")
    preview_dir = os.path.join(video_chunk, "preview")

    n_rgb = write_rgb_images(rgb_frames, rgb_dir, ep_id)
    n_depth = write_depth_pngs(depth_frames, depth_dir, ep_id, cfg.depth_scale)
    write_preview(rgb_frames, preview_dir, ep_id, "rgb", cfg.fps)

    # Verify images were written (labeling below depends on them being on disk for training)
    first_depth_png = os.path.join(depth_dir, f"episode_{ep_id:06d}_0.{DEPTH_EXT}")
    if not os.path.exists(first_depth_png):
        raise FileNotFoundError(
            f"Depth PNG not found after write_depth_pngs: {first_depth_png}. "
            f"Check disk space and permissions. Labeling aborted to prevent inconsistent dataset."
        )
    if n_depth != n_frames:
        raise ValueError(f"Wrote {n_depth} depth PNGs but expected {n_frames}. Dataset incomplete.")

    # 4. Build robot poses (only executes after PNG verification passes)
    robot = extrinsics @ cam_to_robot_footprint(CAMERA_PITCH_DEG)
    xyyaw = np.column_stack([robot[:, 0, 3], robot[:, 1, 3], np.arctan2(robot[:, 1, 0], robot[:, 0, 0])])

    # 4. Discretize actions
    actions = discretize_actions(xyyaw, cfg.fwd_thresh, np.radians(cfg.yaw_thresh_deg))

    # 5. Intrinsics from first depth frame
    h, w = depth_frames[0].shape
    fx, fy, cx, cy = intrinsics_from_params(params_list, w, h, cfg.hfov)
    # Stashed so write_meta can record them in info.json before the raw json is deleted.
    cfg.last_intrinsics = (float(fx), float(fy), float(cx), float(cy))

    # 6. Compute pixel goals
    # Key design: depth_frames (uint16 mm) stays in memory from step 1. We:
    #   - Wrote it to PNG in step 2 (for training dataset)
    #   - Now convert it to float32 m for labeling (in-place, no PNG re-read)
    # This saves one disk I/O cycle compared to the old two-script pipeline.
    # The PNG write is synchronous and verified above, so this is safe.
    depth_m = depth_frames.astype(np.float32) / cfg.depth_scale
    goals, rel_ids = [], []
    for i in range(n_frames):
        goal, rel = farthest_visible_goal(
            extrinsics,
            i,
            depth_m[i],
            fx,
            fy,
            cx,
            cy,
            cfg.lookahead,
            cfg.margin,
            cfg.min_ground_dist,
            cfg.ground_z,
            cfg.flip_v,
            cfg.depth_is_euclidean,
            xyyaw[:, :2],
        )
        if goal is None or rel < 3:
            goals.append([-1, -1])
            rel_ids.append(-1)
        else:
            goals.append([int(goal[0]), int(goal[1])])
            rel_ids.append(int(rel))

    # 7. Build output data
    out_data = {
        "action": actions.tolist(),
        f"pose.{setting}": [m.tolist() for m in extrinsics],
        f"goal.{setting}": goals,
        f"relative_goal_frame_id.{setting}": rel_ids,
    }

    # Pedestrians (if present). The struct is scene-dependent -- {p_1: [16 floats], ...} with
    # however many pedestrians that scene spawned -- so the type is inferred per episode from
    # the first frame. Leaving peds_type None (the old TODO here) meant build_schema never
    # declared the field, and from_pydict silently DROPPED the column: the trajectories were
    # read into out_data and then thrown away at write time. That is the whole social signal
    # of the dataset, and it is only in the raw json, so losing it is unrecoverable once the
    # json is deleted.
    peds_key = "observation.peds_state"
    peds_type = None
    if peds_key in params_list[0]:
        out_data["pose_peds"] = [p[peds_key] for p in params_list]
        first = params_list[0][peds_key]
        if isinstance(first, dict) and first:
            # Every pedestrian is a flat row-major 4x4. Keys are sorted so column order is
            # stable across episodes; a frame missing a pedestrian yields null for it rather
            # than shifting the others.
            widths = {len(v) for p in params_list for v in p[peds_key].values()}
            if widths == {16}:
                peds_type = pa.struct([
                    pa.field(k, pa.list_(pa.float32(), 16)) for k in sorted(first)
                ])
            else:
                print(f"  WARN ep{ep_id}: peds_state entries are {sorted(widths)} floats, "
                      f"expected 16 (4x4); pose_peds omitted", file=sys.stderr)
        if peds_type is None:
            out_data.pop("pose_peds", None)

    frame_idx = np.arange(n_frames)
    out_data["timestamp"] = (frame_idx / cfg.fps).astype(np.float32).tolist()
    out_data["frame_index"] = frame_idx.astype(np.int64).tolist()
    out_data["episode_index"] = [ep_id] * n_frames
    out_data["index"] = frame_idx.astype(np.int64).tolist()  # offset applied in write_meta
    out_data["task_index"] = [task_index] * n_frames

    # 8. Write parquet
    os.makedirs(data_chunk, exist_ok=True)
    pq_path = os.path.join(data_chunk, f"episode_{ep_id:06d}.parquet")
    schema = build_schema(setting, peds_type)
    pq.write_table(pa.Table.from_pydict(out_data, schema=schema), pq_path)

    # 9. Cleanup raw inputs: the npy frames and, for the flat world layout, the params json.
    #
    # This is the only copy of the capture -- deleting it means an Isaac re-run to get it
    # back -- so it happens only after the parquet is on disk AND reads back with the row
    # count we just wrote. A parquet that exists but is truncated (disk full mid-write) would
    # otherwise take the raw data down with it.
    #
    # Deleting the json also removes this episode from flat_episodes(), which is what makes
    # a later rerun skip it instead of re-encoding: the "raw npy are gone" test is no longer
    # the only signal.
    if not cfg.keep_src:
        try:
            written = pq.read_metadata(pq_path).num_rows
        except Exception as e:
            raise RuntimeError(
                f"parquet {pq_path} unreadable right after writing "
                f"({type(e).__name__}: {e}); raw npy/json kept"
            ) from e
        if written != n_frames:
            raise RuntimeError(
                f"parquet {pq_path} has {written} rows but {n_frames} were written; "
                f"raw npy/json kept"
            )
        for p in rgb_npy + depth_npy:
            os.remove(p)
        if raw is not None:
            os.remove(params_path)

    # 10. Return stats
    n_goals = sum(1 for r in rel_ids if r > 0)
    counts = {int(a): int((actions == a).sum()) for a in ACTION_VALUES}
    return {"episode_index": ep_id, "length": n_frames, "n_goals": n_goals, "action_counts": counts}


def read_instruction(ep_dir: str):
    path = os.path.join(ep_dir, "instruction", "instruction.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("parsed_result", {}).get("instruction") or data.get("raw_text", "").strip("```json\n")


def write_meta(scene_path, records, instructions, cfg):
    """Write meta/ (info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl)."""
    meta_dir = os.path.join(scene_path, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    setting = f"{CAMERA_HEIGHT_CM}cm_{CAMERA_PITCH_DEG}deg"

    # Deduplicate instructions → tasks
    task_to_idx, tasks = {}, []
    for r in records:
        instr = instructions.get(str(r["episode_index"]), "PLACEHOLDER_INSTRUCTION")
        if instr not in task_to_idx:
            task_to_idx[instr] = len(tasks)
            tasks.append({"task_index": len(tasks), "task": instr})
        r["task_index"] = task_to_idx[instr]

    # tasks.jsonl
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # episodes.jsonl
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for r in records:
            instr = instructions.get(str(r["episode_index"]), "PLACEHOLDER_INSTRUCTION")
            f.write(json.dumps({"episode_index": r["episode_index"], "tasks": [instr], "length": r["length"]}, ensure_ascii=False) + "\n")

    # The dataset-wide row numbering LeRobot calls `index`, written into each parquet before
    # stats are taken so the two cannot disagree. Done here rather than in process_episode
    # because the offset of an episode depends on the lengths of all episodes before it, which
    # is only known once the whole set is assembled.
    offsets, running = {}, 0
    for r in sorted(records, key=lambda x: x["episode_index"]):
        offsets[r["episode_index"]] = running
        running += r["length"]
    _sync_index_column(scene_path, records, offsets)
    _sync_content_dtypes(scene_path, records, setting)

    # episodes_stats.jsonl
    with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w") as f:
        for r in records:
            ep = r["episode_index"]
            pq_path = os.path.join(
                scene_path, "data", f"chunk-{ep // CHUNK_SIZE:03d}", f"episode_{ep:06d}.parquet")
            if not os.path.exists(pq_path):
                print(f"  WARN episode_{ep:06d}: parquet missing, stats entry omitted", file=sys.stderr)
                continue
            f.write(json.dumps(episode_stats_entry(pq_path, ep, setting, offsets.get(ep, 0))) + "\n")

    # info.json -- exactly the reference key set, in the reference order.
    total_frames = sum(r["length"] for r in records)
    first_pq = None
    for r in sorted(records, key=lambda x: x["episode_index"]):
        cand = os.path.join(scene_path, "data",
                            f"chunk-{r['episode_index'] // CHUNK_SIZE:03d}",
                            f"episode_{r['episode_index']:06d}.parquet")
        if os.path.exists(cand):
            first_pq = cand
            break
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(
            {
                "codebase_version": "v2.1",
                "robot_type": cfg.robot_type,
                "total_episodes": len(records),
                "total_frames": total_frames,
                "total_tasks": len(tasks),
                # No feature has dtype "video" -- rgb/depth are per-frame stills -- so this is
                # 0, which is also what the reference carries.
                "total_videos": 0,
                "total_chunks": len({r["episode_index"] // CHUNK_SIZE for r in records}),
                "chunks_size": CHUNK_SIZE,
                # int, not float: 30.0 would declare a float feature rate where the format
                # uses a plain integer.
                "fps": int(cfg.fps) if float(cfg.fps).is_integer() else cfg.fps,
                "splits": {"train": f"0:{len(records)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": build_features(first_pq, setting) if first_pq else {},
            },
            f,
            indent=4,
        )


def find_sessions(root: str):
    """Session dirs under root, sorted."""
    return sorted(d for d in (os.path.join(root, name) for name in os.listdir(root)) if os.path.isdir(d))


def episode_dirs(session_dir: str):
    """(index, path) for episodes under a session."""
    found = []
    for name in sorted(os.listdir(session_dir)):
        m = EPISODE_PAT.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(session_dir, name)))
    return sorted(found)


class SceneReport:
    """Per-world tally, printed as the run summary so a batch is auditable.

    A batch over many worlds must not lose track of what happened, so every episode lands
    in exactly one of: converted, skipped (already done), or failed with a reason.
    """

    def __init__(self, scene, n_candidates):
        self.scene = scene
        self.n_candidates = n_candidates
        self.skipped = []
        self.failed = []          # (label, reason)
        self.frames = 0
        self.goals = 0
        self.total_in_meta = 0
        self.wrote_meta = False

    @property
    def n_converted(self):
        return self.n_candidates - len(self.skipped) - len(self.failed)

    def finish(self, wrote_meta, total_in_meta=0):
        self.wrote_meta = wrote_meta
        self.total_in_meta = total_in_meta


_ARROW_DTYPE = {
    "int32": "int32", "int64": "int64", "float": "float32", "double": "float64", "bool": "bool",
}


def build_features(pq_path, setting):
    """The info.json `features` dict, derived from the parquet's own schema.

    Read from the file rather than hardcoded so the declaration cannot drift from the data.
    Content columns carry `names` (a label per feature); the LeRobot bookkeeping columns
    (timestamp, frame_index, episode_index, index, task_index) carry names: null.
    """
    schema = pq.read_table(pq_path).schema

    def describe(field):
        t, shape = field.type, [1]
        # Unwrap fixed-size lists to a shape: the 4x4 pose is list[4] of list[4].
        while pa.types.is_fixed_size_list(t) or pa.types.is_list(t):
            shape = ([t.list_size] if pa.types.is_fixed_size_list(t) else [-1]) + (
                shape if shape != [1] else [])
            t = t.value_type
        if pa.types.is_struct(t):
            # No struct dtype exists in the format; describe it as (n_members, member_width)
            # and name the members, which is the closest faithful rendering.
            widths = {f.type.list_size for f in t if pa.types.is_fixed_size_list(f.type)}
            return {
                "dtype": "float32",
                "shape": [len(t), widths.pop() if len(widths) == 1 else -1],
                "names": [f.name for f in t],
            }
        dtype = _ARROW_DTYPE.get(str(t), str(t))
        bookkeeping = field.name in (
            "timestamp", "frame_index", "episode_index", "index", "task_index")
        return {
            "dtype": dtype,
            "shape": shape,
            "names": None if bookkeeping else (
                ["action_index"] if field.name == "action" else [field.name]),
        }

    # Reference order: action, the per-setting group, then the bookkeeping columns. Anything
    # else in the parquet (pose_peds) keeps its schema position ahead of the bookkeeping block.
    head = ["action", f"pose.{setting}", f"goal.{setting}", f"relative_goal_frame_id.{setting}"]
    tail = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    names = [f.name for f in schema]
    order = ([n for n in head if n in names]
             + [n for n in names if n not in head and n not in tail]
             + [n for n in tail if n in names])
    return {n: describe(schema.field(n)) for n in order}


def _sync_content_dtypes(scene_path, records, setting):
    """Cast the content columns of existing parquets to int32 to match the declared schema.

    Episodes converted before the schema switch have them as int64; leaving them would make
    info.json's features block disagree with the files it describes.
    """
    want = {"action": pa.int32(), f"goal.{setting}": pa.list_(pa.int32(), 2),
            f"relative_goal_frame_id.{setting}": pa.int32()}
    fixed = []
    for r in sorted(records, key=lambda x: x["episode_index"]):
        ep = r["episode_index"]
        path = os.path.join(
            scene_path, "data", f"chunk-{ep // CHUNK_SIZE:03d}", f"episode_{ep:06d}.parquet")
        if not os.path.exists(path):
            continue
        table = pq.read_table(path)
        changed = False
        for name, typ in want.items():
            if name in table.column_names and not table.schema.field(name).type.equals(typ):
                i = table.schema.get_field_index(name)
                table = table.set_column(i, pa.field(name, typ), table.column(name).cast(typ))
                changed = True
        if changed:
            tmp = path + ".tmp"
            pq.write_table(table, tmp)
            os.replace(tmp, path)
            fixed.append(ep)
    if fixed:
        print(f"  content columns cast to int32 for episode(s): {fixed}")


def _sync_index_column(scene_path, records, offsets):
    """Add or repair the dataset-wide `index` column in each episode's parquet.

    LeRobot carries two counters: frame_index restarts at 0 each episode, while index runs
    continuously across the dataset. Only the former was written, so `index` was missing
    entirely. Rewrites via a same-directory temp + os.replace so an interruption cannot leave
    a truncated parquet in place of a good one.
    """
    added = []
    for r in sorted(records, key=lambda x: x["episode_index"]):
        ep = r["episode_index"]
        path = os.path.join(
            scene_path, "data", f"chunk-{ep // CHUNK_SIZE:03d}", f"episode_{ep:06d}.parquet")
        if not os.path.exists(path):
            continue
        table = pq.read_table(path)
        want = list(range(offsets[ep], offsets[ep] + table.num_rows))
        if "index" in table.column_names and table.column("index").to_pylist() == want:
            continue
        col = pa.array(want, type=pa.int64())
        if "index" in table.column_names:
            table = table.set_column(table.schema.get_field_index("index"), "index", col)
        else:
            # After episode_index, matching build_schema's tail order
            # (timestamp, frame_index, episode_index, index, task_index).
            pos = (table.schema.get_field_index("episode_index") + 1
                   if "episode_index" in table.column_names else table.num_columns)
            table = table.add_column(pos, pa.field("index", pa.int64()), col)
        tmp = path + ".tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, path)
        added.append(ep)
    if added:
        print(f"  index column written for episode(s): {added}")


def _feature_stats(values, n_rows):
    """min/max/mean/std/count for one feature, in the LeRobot episodes_stats shape.

    Every statistic keeps the feature's own shape and is wrapped in a list, because LeRobot
    treats each feature as an array: a scalar column becomes [v], the 2-element goal becomes
    [u, v], and the 4x4 pose becomes a 4x4 nested list. count is always a single-element list
    holding the row count, not the feature's element count.

    Reductions run elementwise over the frame axis, so a 4x4 pose yields a 4x4 of per-element
    means rather than one number for the whole matrix.
    """
    raw = np.asarray(values)
    is_int = np.issubdtype(raw.dtype, np.integer)
    a = raw.astype(np.float64)
    if a.ndim == 1:
        a = a[:, None]  # scalar column -> (n, 1), so the wrapped output is [v]
    lo, hi = a.min(axis=0), a.max(axis=0)
    if is_int:
        # min/max of an integer feature stay integral; only mean/std become floats. Writing
        # -1.0 where the reference has -1 makes an int feature look like a float one.
        lo, hi = lo.astype(np.int64), hi.astype(np.int64)
    return {
        "min": lo.tolist(),
        "max": hi.tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "count": [int(n_rows)],
    }


def episode_stats_entry(pq_path, ep_id, setting, index_offset):
    """One episodes_stats.jsonl record: {"episode_index": N, "stats": {per-feature ...}}.

    Read from the parquet rather than carried along in `records`, so an episode converted on
    an earlier run gets the same treatment as one converted just now.

    index_offset is where this episode starts in the dataset-wide row numbering, which is what
    the `index` feature counts (frame_index restarts per episode; index does not).
    """
    t = pq.read_table(pq_path)
    cols = t.column_names
    n = t.num_rows
    stats = {}

    if "action" in cols:
        stats["action"] = _feature_stats(t.column("action").to_pylist(), n)
    for key in (f"pose.{setting}", f"goal.{setting}", f"relative_goal_frame_id.{setting}"):
        if key in cols:
            stats[key] = _feature_stats(t.column(key).to_pylist(), n)
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key in cols:
            stats[key] = _feature_stats(t.column(key).to_pylist(), n)
        elif key == "index":
            # Not a column in this dataset's parquet; derive the dataset-wide row numbering
            # so the feature is still described.
            stats[key] = _feature_stats(list(range(index_offset, index_offset + n)), n)

    # Exactly the two keys the format defines. The action histogram that used to ride along
    # here is gone: it is recomputable from the parquet's action column at any time.
    return {"episode_index": ep_id, "stats": stats}


def stats_from_parquet(pq_path, ep_id, setting):
    """Rebuild an episode's stats record from its parquet.

    Lets a rerun regenerate meta/ covering the whole dataset instead of only the episodes
    converted this time. Every field write_meta needs is recoverable: length is the row
    count, action_counts is a value count, n_goals counts positive goal frame ids.
    """
    t = pq.read_table(pq_path, columns=["action", f"relative_goal_frame_id.{setting}"])
    actions = np.asarray(t.column("action").to_pylist())
    rel = np.asarray(t.column(f"relative_goal_frame_id.{setting}").to_pylist())
    return {
        "episode_index": ep_id,
        "length": t.num_rows,
        "n_goals": int((rel > 0).sum()),
        "action_counts": {int(a): int((actions == a).sum()) for a in ACTION_VALUES},
    }


def merge_existing_records(scene_dir, new_records, cfg):
    """new_records plus every already-converted episode's parquet, sorted by episode index.

    meta/ describes the whole dataset. Rebuilding it from only this run's records would
    silently drop earlier episodes from info.json and the jsonl files.
    """
    setting = f"{CAMERA_HEIGHT_CM}cm_{CAMERA_PITCH_DEG}deg"
    have = {r["episode_index"] for r in new_records}
    merged = list(new_records)

    data_root = os.path.join(scene_dir, "data")
    for chunk in sorted(os.listdir(data_root)) if os.path.isdir(data_root) else []:
        chunk_dir = os.path.join(data_root, chunk)
        if not (chunk.startswith("chunk-") and os.path.isdir(chunk_dir)):
            continue
        for name in sorted(os.listdir(chunk_dir)):
            m = re.match(r"episode_(\d+)\.parquet$", name)
            if not m:
                continue
            ep_id = int(m.group(1))
            if ep_id in have:
                continue
            try:
                merged.append(stats_from_parquet(os.path.join(chunk_dir, name), ep_id, setting))
            except Exception as e:
                # Don't fail the run over an unreadable old parquet; just say it is missing
                # from meta/ so the gap is visible.
                print(f"  WARN episode_{ep_id:06d}: existing parquet unreadable, omitted "
                      f"from meta/ ({type(e).__name__}: {e})", file=sys.stderr)

    return sorted(merged, key=lambda r: r["episode_index"])


def flat_episodes(scene_dir: str):
    """(src_ep, params_path) for the world-based layout, sorted by episode index.

    Empty list if this is not a world-layout scene, which sends the caller down the
    legacy session/episode_NN path.
    """
    data_dir = os.path.join(scene_dir, "data")
    try:
        names = os.listdir(data_dir)
    except OSError:
        return []
    found = [
        (int(m.group(1)), os.path.join(data_dir, n))
        for n in names
        if (m := re.match(r"episode_(\d+)\.json$", n))
    ]
    return sorted(found)


def convert_scene_flat(scene_dir, scene, raw_episodes, external_instructions, cfg) -> bool:
    """World-based layout: <scene>/{data/episode_XXXXXX.json, videos/episode_XXXXXX_N.npy}.

    Output goes to the same place as the legacy path -- data/chunk-XXX/*.parquet and
    videos/chunk-XXX/<key>/*.png -- so the two layouts converge after conversion. The
    chunk-XXX subdirs are why raw and converted files can share videos/ and data/ without
    colliding.
    """
    videos_dir = os.path.join(scene_dir, "videos")
    print(f"\n=== {scene}: {len(raw_episodes)} episode(s) with raw data ===")
    instructions, records = {}, []
    report = SceneReport(scene, len(raw_episodes))

    # Episode ids are the logger's own indices, so a rerun that appends episodes does not
    # renumber the ones already converted.
    for ep_id, params_path in raw_episodes:
        chunk = f"chunk-{ep_id // CHUNK_SIZE:03d}"
        label = f"episode_{ep_id:06d}"

        try:
            stats = process_episode(
                scene_dir,
                ep_id,
                os.path.join(scene_dir, "videos", chunk),
                os.path.join(scene_dir, "data", chunk),
                0,  # task_index set later
                cfg,
                raw=(videos_dir, ep_id, params_path),
            )
            if stats is None:
                print(f"  SKIP {label}: no raw frames (already converted)")
                report.skipped.append(label)
                continue

            if str(ep_id) in external_instructions:
                instructions[str(ep_id)] = external_instructions[str(ep_id)]

            records.append(stats)
            report.frames += stats["length"]
            report.goals += stats["n_goals"]
            print(f"  OK   {label}: {stats['length']} rows, {stats['n_goals']} goals")

        except Exception as e:
            # Keep going: one corrupt episode should not cost the whole batch. The reason
            # and the episode are recorded and reprinted in the final summary, and the
            # traceback goes to stderr for the actual line.
            print(f"  FAIL {label}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            report.failed.append((label, f"{type(e).__name__}: {e}"))
            continue

    if not records:
        print(f"{scene}: no episodes converted")
        report.finish(wrote_meta=False)
        return report

    # meta/ describes every episode in the dataset, not just this run's, so it is rebuilt
    # from the parquet already on disk plus what we just added. Without this a second run
    # would shrink meta/ down to only the newly-converted episodes.
    all_records = merge_existing_records(scene_dir, records, cfg)
    write_meta(scene_dir, all_records, instructions, cfg)
    report.finish(wrote_meta=True, total_in_meta=len(all_records))
    return report


def convert_scene(scene_dir: str, external_instructions: dict, cfg) -> bool:
    """Process one scene: all sessions → labeled dataset.

    Args:
        scene_dir: Path to scene directory
        external_instructions: {episode_id: instruction_text} from --instructions flag
        cfg: Config object
    """
    scene = os.path.basename(scene_dir)

    # World-based layout (logger run with --world): no session level at all. Raw frames sit
    # flat in <scene>/videos/episode_XXXXXX_N.npy and params in
    # <scene>/data/episode_XXXXXX.json. Detected by those json, since a converted dataset
    # has data/chunk-XXX/ subdirs instead.
    raw_episodes = flat_episodes(scene_dir)
    if raw_episodes:
        return convert_scene_flat(scene_dir, scene, raw_episodes, external_instructions, cfg)

    # A world whose episodes are all converted has no episode json left (cleanup deletes them),
    # so flat_episodes() is empty and the legacy branch below would treat data/, meta/, videos/
    # and any leftover scenario dir as "sessions". Recognise the converted layout instead and
    # report nothing to do.
    if os.path.isdir(os.path.join(scene_dir, "meta")) or glob.glob(
            os.path.join(scene_dir, "data", "chunk-*")):
        print(f"{scene}: no raw episodes left, already converted")
        return None

    sessions = find_sessions(scene_dir)
    if not sessions:
        print(f"{scene}: no session dirs and no data/episode_*.json, skipped")
        return None  # not a dataset dir at all -- left out of the summary

    print(f"\n=== {scene}: {len(sessions)} session(s) ===")
    instructions, ep_id, records = {}, 0, []
    n_candidates = sum(len(episode_dirs(s)) for s in sessions)
    report = SceneReport(scene, n_candidates)

    for session in sessions:
        for src_idx, ep_dir in episode_dirs(session):
            chunk = f"chunk-{ep_id // CHUNK_SIZE:03d}"
            label = f"{os.path.relpath(ep_dir, scene_dir)} -> episode_{ep_id:06d}"

            try:
                stats = process_episode(
                    ep_dir,
                    ep_id,
                    os.path.join(scene_dir, "videos", chunk),
                    os.path.join(scene_dir, "data", chunk),
                    0,  # task_index set later
                    cfg,
                )
                if stats is None:
                    print(f"  SKIP {label}: already converted")
                    report.skipped.append(label)
                    ep_id += 1
                    continue

                # Priority: --instructions flag > episode/instruction.json > placeholder
                if str(ep_id) in external_instructions:
                    instructions[str(ep_id)] = external_instructions[str(ep_id)]
                else:
                    instr = read_instruction(ep_dir)
                    if instr:
                        instructions[str(ep_id)] = instr

                records.append(stats)
                report.frames += stats["length"]
                report.goals += stats["n_goals"]
                print(f"  OK   {label}: {stats['length']} rows, {stats['n_goals']} goals")
                ep_id += 1

            except Exception as e:
                print(f"  FAIL {label}: {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc()
                report.failed.append((label, f"{type(e).__name__}: {e}"))
                ep_id += 1   # keep ids aligned with the episodes we walked
                continue

    if not records:
        print(f"{scene}: no episodes converted")
        report.finish(wrote_meta=False)
        return report

    write_meta(scene_dir, records, instructions, cfg)
    report.finish(wrote_meta=True, total_in_meta=len(records))
    return report


def print_summary(reports, cfg):
    """Final tally across every world, with each failure's episode and reason.

    Printed even when nothing converted, so "no output" is never ambiguous between "no new
    data" and "everything blew up".
    """
    setting = f"{CAMERA_HEIGHT_CM}cm_{CAMERA_PITCH_DEG}deg"
    print("\n" + "=" * 74)
    print(f"SUMMARY   camera {setting}   (CAMERA_HEIGHT_CM / CAMERA_PITCH_DEG)")
    print("=" * 74)

    if not reports:
        print("No dataset directories found under --data-path. Nothing to do.")
        print(f"  Expected <world>/data/episode_XXXXXX.json under {cfg.data_path}")
        return

    conv = skip = fail = frames = goals = 0
    for r in reports:
        conv += r.n_converted
        skip += len(r.skipped)
        fail += len(r.failed)
        frames += r.frames
        goals += r.goals
        meta = f", meta/ covers {r.total_in_meta}" if r.wrote_meta else ", meta/ not written"
        print(
            f"  {r.scene:<28} converted {r.n_converted:>4}  skipped {len(r.skipped):>4}  "
            f"failed {len(r.failed):>3}{meta}"
        )

    print("-" * 74)
    print(f"  {len(reports)} world(s): {conv} converted, {skip} already done, {fail} failed")
    print(f"  {frames} frames, {goals} pixel goals this run")

    if fail:
        # Repeated at the end so a long scroll of per-episode output cannot bury it.
        print(f"\n  {fail} FAILURE(S) -- episode and reason:")
        for r in reports:
            for label, reason in r.failed:
                print(f"    {r.scene}/{label}: {reason}")
        print("\n  Failed episodes keep their raw npy, so fixing the cause and re-running")
        print("  picks them up again. Tracebacks are above on stderr.")
    print("=" * 74)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True, help="scene root (collected_data/<session>)")
    # Camera height/pitch are not flags -- see CAMERA_HEIGHT_CM / CAMERA_PITCH_DEG at the
    # top of this file.
    p.add_argument("--hfov", type=float, default=90.0, help="horizontal FOV in degrees")
    p.add_argument("--depth-scale", type=float, default=1000.0, help="PNG units per metre (1000 = mm)")
    p.add_argument("--flip-v", action="store_true", help="flip v coord if camera +Y points UP")
    p.add_argument(
        "--fwd-speed-thresh",
        type=float,
        default=0.05,
        help="minimum forward SPEED in m/s to label a frame FORWARD rather than STOP; "
             "converted to a per-frame distance with --fps (default 0.05 m/s)",
    )
    p.add_argument(
        "--fwd-thresh",
        type=float,
        default=None,
        help="deprecated: per-frame forward distance in m. Frame-rate dependent, so the same "
             "value means different speeds at different --fps. Overrides --fwd-speed-thresh "
             "when given.",
    )
    p.add_argument(
        "--yaw-speed-thresh-deg",
        type=float,
        default=8.0,
        help="minimum turn RATE in deg/s to label a frame TURN_LEFT/TURN_RIGHT; converted to a "
             "per-frame angle with --fps (default 8 deg/s). Replaces the frame-rate dependent "
             "--yaw-thresh-deg, whose 5 deg/FRAME default needed 150 deg/s -- about 1.7x the "
             "robot's own 86 deg/s ceiling -- so TURN was never emitted at 30 fps.",
    )
    p.add_argument(
        "--yaw-thresh-deg",
        type=float,
        default=None,
        help="deprecated: per-frame turn angle in degrees. Frame-rate dependent, so the same "
             "value means different rates at different --fps. Overrides --yaw-speed-thresh-deg "
             "when given.",
    )
    p.add_argument(
        "--lookahead",
        type=float,
        default=5.0,
        help="how far ahead to search for a goal, in METRES of path travelled. Not frames: "
             "a frame count means a different distance at every speed and fps, while the "
             "camera's near blind spot is a fixed distance (default 5.0 m)",
    )
    p.add_argument("--margin", type=float, default=0.2, help="depth occlusion margin (m)")
    p.add_argument(
        "--min-ground-dist",
        type=float,
        default=1.0,
        help="reject footprints closer than this, measured robot-origin to robot-origin (m). "
             "With the current mount (h=1.32 m, pitch 30 deg, 480 px tall) the ground enters "
             "the frame at 0.84 m from the robot origin, so this threshold is now the binding "
             "one -- at the old 15 deg pitch the blind spot reached 1.31 m and the projection "
             "test rejected everything nearer first (default 1.0 m)",
    )
    p.add_argument("--ground-z", type=float, default=0.0, help="world z of the floor plane")
    p.add_argument(
        "--depth-is-euclidean",
        action="store_true",
        help="set if depth stores radial distance (distance_to_camera) instead of z-depth (distance_to_image_plane)",
    )
    p.add_argument("--fps", type=float, default=30.0, help="capture rate (fps)")
    p.add_argument("--robot-type", default="social_gen", help="robot_type field in info.json")
    p.add_argument(
        "--instructions",
        help='JSON file or directory with instructions. File: {"0": "instruction", ...}. '
        'Dir: contains <scene>.json per scene. Overwrites any instructions found in episode dirs. '
        'Use this to inject VLM-generated instructions after data collection.',
    )
    p.add_argument(
        "--keep-src",
        action="store_true",
        help="keep the raw inputs after conversion. By default a successfully converted "
             "episode has its npy frames and its data/episode_XXXXXX.json deleted, which is "
             "also what makes a later rerun skip it. The raw data cannot be regenerated "
             "without re-running Isaac, so deletion is gated on the parquet reading back with "
             "the expected row count.",
    )
    p.add_argument("--workers", type=int, default=1, help="parallel workers (currently unused, TODO)")
    cfg = p.parse_args()

    # Resolve the forward threshold to the per-frame distance discretize_actions wants.
    # It compares against a single frame-to-frame step, so a distance given directly is
    # only meaningful alongside a frame rate: 0.05 m/frame is 1.5 m/s at 30 fps, i.e. far
    # above any indoor robot, which would label every real frame STOP. Deriving it from a
    # speed keeps the meaning stable when --fps changes.
    if cfg.fwd_thresh is None:
        cfg.fwd_thresh = cfg.fwd_speed_thresh / cfg.fps
    else:
        print(f"  NOTE --fwd-thresh {cfg.fwd_thresh} m/frame given directly "
              f"(= {cfg.fwd_thresh * cfg.fps:.3f} m/s at {cfg.fps} fps); "
              f"ignoring --fwd-speed-thresh", file=sys.stderr)
    # Same treatment for yaw. Warn when the derived gate exceeds what the robot can physically
    # turn, because that silently makes TURN_* unreachable rather than merely rare.
    if cfg.yaw_thresh_deg is None:
        cfg.yaw_thresh_deg = cfg.yaw_speed_thresh_deg / cfg.fps
    else:
        print(f"  NOTE --yaw-thresh-deg {cfg.yaw_thresh_deg} deg/frame given directly "
              f"(= {cfg.yaw_thresh_deg * cfg.fps:.1f} deg/s at {cfg.fps} fps); "
              f"ignoring --yaw-speed-thresh-deg", file=sys.stderr)
    _yaw_rate = cfg.yaw_thresh_deg * cfg.fps
    if _yaw_rate > 86.0:  # velocity_smoother_max_velocity[2] = 1.5 rad/s
        print(f"  WARNING turn gate is {_yaw_rate:.0f} deg/s but the robot's ceiling is ~86 deg/s "
              f"-- TURN_LEFT/TURN_RIGHT can NEVER be emitted. Lower --yaw-speed-thresh-deg.",
              file=sys.stderr)
    print(f"  action threshold: forward > {cfg.fwd_thresh:.5f} m/frame "
          f"(= {cfg.fwd_thresh * cfg.fps:.3f} m/s at {cfg.fps} fps), "
          f"turn > {cfg.yaw_thresh_deg:.4f} deg/frame (= {_yaw_rate:.1f} deg/s)")

    # --data-path may be the grscenes root (subdirs are worlds) or a single world dir.
    # A world dir is recognised by its own data/episode_XXXXXX.json; without this check
    # its data/, videos/ and meta/ would each be walked as if they were scenes.
    # meta/ or data/chunk-*/ means this is a world that has already been converted -- its
    # episode json are gone, so flat_episodes() cannot recognise it any more and the else
    # branch would walk data/, meta/ and videos/ as if each were a separate world.
    if flat_episodes(cfg.data_path) or os.path.isdir(
            os.path.join(cfg.data_path, "meta")) or glob.glob(
            os.path.join(cfg.data_path, "data", "chunk-*")):
        scenes = [cfg.data_path]
    else:
        scenes = sorted(
            os.path.join(cfg.data_path, d) for d in os.listdir(cfg.data_path) if os.path.isdir(os.path.join(cfg.data_path, d))
        )
        if not scenes:
            scenes = [cfg.data_path]  # treat data_path itself as a scene

    # Load external instructions (per-scene or single file)
    def load_instructions(scene_name):
        if not cfg.instructions:
            return {}
        path = cfg.instructions
        if os.path.isdir(path):
            path = os.path.join(path, f"{scene_name}.json")
            if not os.path.exists(path):
                return {}
        with open(path) as f:
            data = json.load(f)
            # Ensure keys are strings (episode_id as string)
            return {str(k): v for k, v in data.items()}

    # One world failing must not abandon the rest of the batch, so the exception is caught
    # per world and turned into a report entry.
    reports = []
    for scene in scenes:
        name = os.path.basename(scene)
        try:
            external_instr = load_instructions(name)
            rep = convert_scene(scene, external_instr, cfg)
            if rep is not None:
                reports.append(rep)
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            rep = SceneReport(name, 0)
            rep.failed.append((f"{name} (whole world)", f"{type(e).__name__}: {e}"))
            rep.finish(wrote_meta=False)
            reports.append(rep)

    print_summary(reports, cfg)

    # Reminder if no instructions provided
    if not cfg.instructions:
        print(
            "\n[INFO] No --instructions provided. Episodes use placeholder instructions."
            "\nRun annotate_with_instructions.py next to fill them in:"
            "\n  export API_KEY=sk-...   # required by --use-vlm"
            f"\n  python annotate_with_instructions.py --data-path {cfg.data_path} --use-vlm"
        )


if __name__ == "__main__":
    main()
