#!/usr/bin/env python3
"""
Post-process VLN dataset: convert JSON → Parquet and npy → mp4.
Run with system Python (requires pandas, imageio[ffmpeg]) after IsaacSim recording.

Directory structure:
  collected_data/
  └── 2026-04-23_14-30-00/          ← session (one per launch)
      └── episode_000000/
          ├── data/params.json
          ├── rgb_videos/chunk_0000.npy ...
          └── depth_videos/chunk_0000.npy ...

Usage:
    # Convert all sessions
    python3 convert_json_to_parquet.py collected_data [--delete-src] [--fps 30]

    # Convert one specific session
    python3 convert_json_to_parquet.py collected_data/2026-04-23_14-30-00 [--delete-src]
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import imageio.v3 as iio

SESSION_PAT = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
EPISODE_PAT = re.compile(r'^episode_(\d+)$')
CHUNK_PAT   = re.compile(r'^chunk_(\d+)\.npy$')


def _session_dirs(root: str):
    """Yield session dirs (timestamp-named) under root, sorted."""
    for name in sorted(os.listdir(root)):
        if SESSION_PAT.match(name):
            yield os.path.join(root, name)


def _episode_dirs(session_dir: str):
    """Yield (episode_idx, episode_dir) under a session dir, sorted."""
    for name in sorted(os.listdir(session_dir)):
        m = EPISODE_PAT.match(name)
        if m:
            yield int(m.group(1)), os.path.join(session_dir, name)


def _npy_to_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    lo, hi = frames.min(), frames.max()
    frames = ((frames - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo \
             else np.zeros_like(frames, dtype=np.uint8)
    if frames.ndim == 3:
        frames = np.stack([frames] * 3, axis=-1)
    return frames


def _load_chunks(subdir: str):
    """Load and concatenate all chunk_XXXX.npy files in subdir, sorted by index."""
    chunks = sorted(
        (int(m.group(1)), os.path.join(subdir, f))
        for f in os.listdir(subdir)
        if (m := CHUNK_PAT.match(f))
    )
    if not chunks:
        return None, []
    return np.concatenate([np.load(p) for _, p in chunks], axis=0), [p for _, p in chunks]


def convert_session(session_dir: str, delete_src: bool, fps: int):
    session_name = os.path.basename(session_dir)
    print(f"\n--- Session: {session_name} ---")

    for ep_idx, ep_dir in _episode_dirs(session_dir):
        ep_label = f"episode_{ep_idx:06d}"

        # JSON → Parquet
        json_path    = os.path.join(ep_dir, "data", "params.json")
        parquet_path = os.path.join(ep_dir, "data", "params.parquet")
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            pd.DataFrame(data).to_parquet(parquet_path)
            print(f"  JSON→Parquet: {ep_label}/data/params")
            if delete_src:
                os.remove(json_path)

        # RGB npy chunks → mp4
        rgb_dir = os.path.join(ep_dir, "rgb_videos")
        if os.path.isdir(rgb_dir):
            frames, src_paths = _load_chunks(rgb_dir)
            if frames is not None:
                mp4_path = os.path.join(rgb_dir, "video.mp4")
                iio.imwrite(mp4_path, _npy_to_uint8(frames), fps=fps, codec="libx264")
                print(f"  npy→mp4 (RGB):   {ep_label} ({len(frames)} frames, {len(src_paths)} chunks)")
                if delete_src:
                    for p in src_paths:
                        os.remove(p)

        # Depth npy chunks → mp4
        depth_dir = os.path.join(ep_dir, "depth_videos")
        if os.path.isdir(depth_dir):
            frames, src_paths = _load_chunks(depth_dir)
            if frames is not None:
                mp4_path = os.path.join(depth_dir, "video.mp4")
                iio.imwrite(mp4_path, _npy_to_uint8(frames), fps=fps, codec="libx264")
                print(f"  npy→mp4 (Depth): {ep_label} ({len(frames)} frames, {len(src_paths)} chunks)")
                if delete_src:
                    for p in src_paths:
                        os.remove(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process VLN dataset")
    parser.add_argument("path",
                        help="Root collected_data dir (processes all sessions) "
                             "or a specific session dir (e.g. collected_data/2026-04-23_14-30-00)")
    parser.add_argument("--delete-src", action="store_true",
                        help="Delete source JSON/npy files after conversion")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default 30)")
    args = parser.parse_args()

    path = args.path.rstrip("/")
    if not os.path.isdir(path):
        print(f"Error: {path} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Fix permissions in case files were written by root (Isaac container)
    os.system(f"chmod -R a+rwX {path} 2>/dev/null")

    # Determine if path is a session dir or the root collected_data dir
    if SESSION_PAT.match(os.path.basename(path)):
        sessions = [path]
    else:
        sessions = list(_session_dirs(path))
        if not sessions:
            print(f"No session directories found under {path}", file=sys.stderr)
            sys.exit(1)

    for session_dir in sessions:
        convert_session(session_dir, delete_src=args.delete_src, fps=args.fps)

    print("\nDone.")
