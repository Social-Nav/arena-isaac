"""
Batch instruction generator for a grscenes data directory.

For every rgb_video.mp4 found under the given root, generates R2R-format
instructions and saves the result to a sibling `instruction/instruction.json`
alongside the existing rgb_videos / depth_videos / data folders.

Usage (Arena container):
    /opt/venv/bin/python -m arena_isaac.data_logging.postprocess.batch_instructions \
        --root /opt/arena_ws/src/Arena/collected_data/grscenes_1
"""

import argparse
import json
from pathlib import Path

from arena_isaac.data_logging.postprocess.instruction_generator import analyze_video_for_vln


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/opt/arena_ws/src/Arena/collected_data/grscenes_1",
        help="Root data directory to scan for rgb_video.mp4 files",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    videos = sorted(root.rglob("rgb_videos/rgb_video.mp4"))

    if not videos:
        print(f"No rgb_video.mp4 files found under: {root}")
        return

    print(f"Found {len(videos)} video(s) under {root}\n")

    ok, failed = 0, []

    for i, video_path in enumerate(videos, 1):
        episode_dir = video_path.parent.parent   # rgb_videos/ -> episode_XX/
        out_dir = episode_dir / "instruction"
        out_file = out_dir / "instruction.json"

        print(f"[{i}/{len(videos)}] {video_path.relative_to(root)}")

        if out_file.exists():
            print(f"  Already exists, skipping.\n")
            ok += 1
            continue

        result = analyze_video_for_vln(str(video_path), save_output=False)

        if result is None:
            print(f"  FAILED\n")
            failed.append(str(video_path.relative_to(root)))
            continue

        out_dir.mkdir(exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  Saved -> {out_file.relative_to(root)}\n")
        ok += 1

    print("=" * 60)
    print(f"Done  {ok}/{len(videos)} succeeded")
    if failed:
        print(f"Failed:")
        for p in failed:
            print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
