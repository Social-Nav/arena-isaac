#!/usr/bin/env python3
"""Batch generate instructions using VLM and write tasks.jsonl + episodes.jsonl.

This is Stage 2 of the data pipeline:
  Stage 1: convert_and_label.py → PNG + parquet + info.json + episodes_stats.jsonl
  Stage 2: batch_instructions.py (THIS SCRIPT) → tasks.jsonl + episodes.jsonl

Reads RGB preview videos (mp4), calls Gemini VLM to generate navigation instructions,
deduplicates them into tasks, and writes meta/tasks.jsonl and meta/episodes.jsonl.

Usage:
    python batch_instructions.py \\
        --data-path social_gen/traj_data/grscenes/grscenes_1
"""

import argparse
import base64
import json
import os
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

# Camera mount and chunk size come from the conversion script so the two can never
# disagree about directory names. Edit CAMERA_HEIGHT_CM / CAMERA_PITCH_DEG there.
try:
    from .process_raw_to_dataset import CAMERA_HEIGHT_CM, CAMERA_PITCH_DEG, CHUNK_SIZE
except ImportError:  # run as a plain script rather than a package module
    from process_raw_to_dataset import CAMERA_HEIGHT_CM, CAMERA_PITCH_DEG, CHUNK_SIZE


# ========================================
# VLM Configuration
# ========================================

MODEL = "gemini-2.5-pro"
API_URL = f"https://jkwl.dmxapi.cn/v1beta/models/{MODEL}:generateContent"
API_KEY = os.environ.get("API_KEY", "")

VLN_PROMPT = """
You are an expert VLN dataset annotator familiar with the R2R (Room-to-Room) dataset format.
Analyze this egocentric robot video and generate navigation instructions that conform to R2R conventions.

R2R instruction style requirements:
- Write in fluent, natural English.
- Include explicit directional cues (turn left/right, go straight, go through).
- Reference visible landmarks (furniture, architectural features, colors, textures).
- Describe room transitions and environmental changes (floor type, wall color, doorways).
- Clearly mark the endpoint ("Stop at...", "Your destination is...").

Output strictly as JSON with no extra text:
{
  "instruction": "Full natural-language navigation instruction as a single paragraph in English"
}
"""


# ========================================
# VLM Utility Functions (from instruction_generator.py)
# ========================================

def encode_video(video_path: str) -> str | None:
    """Encode a video file to Base64 string for Gemini API."""
    try:
        with open(video_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"    Encoded video: {size_mb:.1f} MB")
        return data
    except Exception as e:
        print(f"    Encoding failed: {e}")
        return None


def get_mime_type(video_path: str) -> str:
    """Return MIME type for video file extension."""
    ext = Path(video_path).suffix.lower()
    mime_map = {
        ".mp4":  "video/mp4",
        ".mov":  "video/quicktime",
        ".avi":  "video/avi",
        ".mkv":  "video/x-matroska",
        ".webm": "video/webm",
    }
    return mime_map.get(ext, "video/mp4")


def call_gemini_api(
    video_data: str,
    mime_type: str,
    prompt: str,
    max_retries: int = 3
) -> dict | None:
    """Send video analysis request to DMXAPI Gemini endpoint."""
    if not API_KEY:
        print("    ERROR: API_KEY not set. Export API_KEY=sk-... before running.")
        return None

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": video_data
                    }
                },
                {"text": prompt}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "maxOutputTokens": 2048
        }
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(f"    API call attempt {attempt}/{max_retries}...")
            response = requests.post(
                API_URL,
                headers=headers,
                params={"key": API_KEY},
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            print(f"    Timeout, retrying...")
            time.sleep(5 * attempt)
        except requests.exceptions.RequestException as e:
            print(f"    Request failed: {e}")
            if hasattr(e, "response") and e.response:
                print(f"       Status: {e.response.status_code}")
                print(f"       Response: {e.response.text[:200]}")
            if attempt < max_retries:
                time.sleep(3 * attempt)

    return None


def extract_text_from_response(response: dict) -> str | None:
    """Extract generated text from Gemini API response."""
    try:
        candidates = response.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
    except Exception as e:
        print(f"    Response parsing error: {e}")
    return None


def parse_json_from_text(text: str) -> dict | None:
    """Fault-tolerant JSON extraction from model output."""
    # Strategy 1: clean JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: fenced code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: first { ... } in text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ========================================
# Episode Discovery and Instruction Generation
# ========================================

def discover_episodes(scene_path):
    """Find all episodes by scanning preview videos across every chunk.

    Walks videos/chunk-*/preview/ rather than only chunk-000: chunks hold CHUNK_SIZE
    episodes each, so a world past that many would otherwise go silently unannotated.

    Returns:
        Dict {episode_index: (length, video_path)}
    """
    videos_root = os.path.join(scene_path, "videos")
    if not os.path.isdir(videos_root):
        raise FileNotFoundError(f"No videos/ dir in {scene_path}")

    chunk_dirs = sorted(
        os.path.join(videos_root, d)
        for d in os.listdir(videos_root)
        if d.startswith("chunk-") and os.path.isdir(os.path.join(videos_root, d))
    )
    if not chunk_dirs:
        raise FileNotFoundError(
            f"No videos/chunk-*/ in {scene_path}. Run process_raw_to_dataset.py first."
        )

    pattern = re.compile(r"episode_(\d+)_rgb\.mp4")
    setting = f"{CAMERA_HEIGHT_CM}cm_{CAMERA_PITCH_DEG}deg"
    episodes = {}

    for chunk_dir in chunk_dirs:
        preview_dir = os.path.join(chunk_dir, "preview")
        if not os.path.isdir(preview_dir):
            continue

        for f in os.listdir(preview_dir):
            m = pattern.match(f)
            if not m:
                continue
            ep_id = int(m.group(1))
            video_path = os.path.join(preview_dir, f)

            # Length from the parquet, which is authoritative.
            parquet_path = os.path.join(
                scene_path,
                f"data/chunk-{ep_id // CHUNK_SIZE:03d}",
                f"episode_{ep_id:06d}.parquet",
            )
            if os.path.exists(parquet_path):
                length = pq.read_metadata(parquet_path).num_rows
            else:
                # Fallback: count RGB frames in the same chunk as the preview. Accept both
                # extensions -- RGB is written as JPEG now, but datasets converted before
                # that change hold PNG, and this fallback is exactly the path an older
                # dataset takes.
                rgb_dir = os.path.join(chunk_dir, f"observation.images.rgb.{setting}")
                png_pattern = re.compile(rf"episode_{ep_id:06d}_(\d+)\.(?:jpg|jpeg|png)$")
                frames = (
                    [int(pm.group(1)) for g in os.listdir(rgb_dir) if (pm := png_pattern.match(g))]
                    if os.path.isdir(rgb_dir)
                    else []
                )
                length = max(frames) + 1 if frames else 0

            episodes[ep_id] = (length, video_path)

    return episodes


def generate_instruction_from_video(video_path: str, ep_id: int) -> str | None:
    """Generate instruction by calling Gemini VLM on preview video.

    Args:
        video_path: Path to episode_XXXXXX_rgb.mp4
        ep_id: Episode index (for logging)

    Returns:
        Generated instruction string, or None if failed
    """
    print(f"    Encoding video...")
    video_data = encode_video(video_path)
    if not video_data:
        return None

    mime_type = get_mime_type(video_path)

    response = call_gemini_api(video_data, mime_type, VLN_PROMPT)
    if not response:
        print(f"    API call failed after retries")
        return None

    text = extract_text_from_response(response)
    if not text:
        print(f"    Could not extract text from response")
        return None

    # Print token usage
    usage = response.get("usageMetadata", {})
    if usage:
        print(f"    Tokens: {usage.get('totalTokenCount', 0)} total "
              f"({usage.get('promptTokenCount', 0)} in, "
              f"{usage.get('candidatesTokenCount', 0)} out)")

    parsed = parse_json_from_text(text)
    if parsed and "instruction" in parsed:
        instruction = parsed["instruction"].strip()
        print(f"    ✓ Generated: {instruction[:80]}...")
        return instruction
    else:
        print(f"    Parsing failed. Raw output: {text[:200]}")
        return None


PLACEHOLDER_PREFIX = "Navigate from start to destination"
FAILED_PREFIX = "FAILED_GENERATION_EP"


def load_existing_instructions(scene_path):
    """{episode_index: instruction} already in meta/episodes.jsonl.

    Each VLM call is a paid request, so a rerun that adds 5 episodes to a world of 200
    must not re-annotate the 200. Placeholders and past failures are treated as absent so
    they do get retried.
    """
    path = os.path.join(scene_path, "meta", "episodes.jsonl")
    if not os.path.exists(path):
        return {}

    done = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tasks = rec.get("tasks") or []
            if not tasks:
                continue
            instr = tasks[0]
            if instr.startswith(PLACEHOLDER_PREFIX) or instr.startswith(FAILED_PREFIX):
                continue
            if instr == "PLACEHOLDER_INSTRUCTION":
                continue
            done[int(rec["episode_index"])] = instr
    return done


def batch_generate_instructions(scene_path, episodes, use_vlm, redo=False):
    """Generate instructions for all episodes, one at a time.

    Serial on purpose: each episode's request completes (returns text, or exhausts its
    retries and returns None) before the next begins, so a rate limit or a hang affects
    only the episode it happens on.

    Args:
        scene_path: Path to scene directory
        episodes: Dict {episode_index: (length, video_path)}
        use_vlm: If False, use placeholder instructions
        redo: Regenerate even episodes that already have a real instruction

    Returns:
        (instructions, n_reused, failures) -- failures is [(ep_id, reason)]
    """
    existing = {} if redo else load_existing_instructions(scene_path)
    instructions = dict(existing)
    failures = []
    n_reused = 0

    todo = [e for e in sorted(episodes) if e not in existing]
    if existing:
        n_reused = len(existing)
        print(f"  Reusing {n_reused} existing instruction(s); {len(todo)} to generate")

    for n, ep_id in enumerate(todo, 1):
        length, video_path = episodes[ep_id]
        print(f"\n  [{n}/{len(todo)}] Episode {ep_id} ({length} frames):")

        if not use_vlm:
            instructions[ep_id] = f"{PLACEHOLDER_PREFIX} (episode {ep_id})"
            continue

        try:
            instr = generate_instruction_from_video(video_path, ep_id)
        except Exception as e:
            # One bad episode must not end the batch: the rest still get annotated and
            # this one is named in the summary.
            print(f"    FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            instructions[ep_id] = f"{FAILED_PREFIX}{ep_id}"
            failures.append((ep_id, f"{type(e).__name__}: {e}"))
            continue

        if instr:
            instructions[ep_id] = instr
        else:
            instructions[ep_id] = f"{FAILED_PREFIX}{ep_id}"
            failures.append((ep_id, "VLM returned no usable instruction after retries"))

        time.sleep(2)  # rate limit

    return instructions, n_reused, failures


# ========================================
# Write Meta Files
# ========================================

def write_tasks_and_episodes(scene_path, episodes, instructions):
    """Write meta/tasks.jsonl and meta/episodes.jsonl.

    Args:
        scene_path: Path to scene directory
        episodes: Dict {episode_index: (length, video_path)}
        instructions: Dict {episode_index: instruction_text}
    """
    meta_dir = os.path.join(scene_path, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # Deduplicate instructions → tasks
    task_to_idx = {}
    tasks = []
    episode_to_task = {}

    for ep_id in sorted(episodes.keys()):
        length, _ = episodes[ep_id]
        instr = instructions.get(ep_id, f"PLACEHOLDER_EP{ep_id}")

        if instr not in task_to_idx:
            task_idx = len(tasks)
            task_to_idx[instr] = task_idx
            tasks.append({"task_index": task_idx, "task": instr})

        episode_to_task[ep_id] = task_to_idx[instr]

    # Write tasks.jsonl
    tasks_path = os.path.join(meta_dir, "tasks.jsonl")
    with open(tasks_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\n✓ Wrote {len(tasks)} unique tasks to {tasks_path}")

    # Write episodes.jsonl
    episodes_path = os.path.join(meta_dir, "episodes.jsonl")
    with open(episodes_path, "w", encoding="utf-8") as f:
        for ep_id in sorted(episodes.keys()):
            length, _ = episodes[ep_id]
            instr = instructions.get(ep_id, f"PLACEHOLDER_EP{ep_id}")
            f.write(json.dumps({
                "episode_index": ep_id,
                "tasks": [instr],  # List format per R2R convention
                "length": length
            }, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {len(episodes)} episodes to {episodes_path}")

    # Update info.json with correct total_tasks
    info_path = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        info["total_tasks"] = len(tasks)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"✓ Updated {info_path} with total_tasks={len(tasks)}")

    # Push task_index into the parquet and episodes_stats.jsonl. Stage 1 writes 0 for every
    # row because instructions do not exist yet, so without this every episode keeps pointing
    # at task 0 -- episode 1 would resolve to episode 0's instruction. episode_to_task was
    # computed above and previously went unused.
    _sync_task_index(scene_path, episode_to_task)


def _sync_task_index(scene_path, episode_to_task):
    """Rewrite the task_index column of each episode's parquet, and its stats entry."""
    data_root = os.path.join(scene_path, "data")
    fixed = []
    for ep_id, task_idx in sorted(episode_to_task.items()):
        chunk = f"chunk-{ep_id // 1000:03d}"
        path = os.path.join(data_root, chunk, f"episode_{ep_id:06d}.parquet")
        if not os.path.exists(path):
            continue
        table = pq.read_table(path)
        if "task_index" not in table.column_names:
            continue
        current = table.column("task_index").to_pylist()
        if current and all(v == task_idx for v in current):
            continue
        col = pa.array([task_idx] * table.num_rows, type=table.schema.field("task_index").type)
        table = table.set_column(table.schema.get_field_index("task_index"), "task_index", col)
        # Same-directory temp then replace, so an interrupted write cannot leave a truncated
        # parquet where the real one was.
        tmp = path + ".tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, path)
        fixed.append((ep_id, task_idx))

    stats_path = os.path.join(scene_path, "meta", "episodes_stats.jsonl")
    if os.path.exists(stats_path):
        rows, changed = [], False
        for line in open(stats_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ti = episode_to_task.get(r.get("episode_index"))
            if ti is not None and r.get("task_index") != {"min": ti, "max": ti}:
                r["task_index"] = {"min": ti, "max": ti}
                changed = True
            rows.append(r)
        if changed:
            with open(stats_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"✓ Updated task_index in {stats_path}")

    if fixed:
        print("✓ Synced task_index into parquet: "
              + ", ".join(f"ep{e}->{t}" for e, t in fixed))


# ========================================
# Main Entry Point
# ========================================

def print_summary(tally, cfg):
    """Closing tally across worlds, naming every episode that failed and why."""
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)

    if not tally:
        print("No annotatable worlds found under --data-path.")
        print(f"  Expected <world>/videos/chunk-*/preview/episode_*_rgb.mp4 under {cfg.data_path}")
        print("  Run process_raw_to_dataset.py first -- it writes those previews.")
        return

    gen = reuse = fail = eps = 0
    for name, n_eps, n_gen, n_reuse, failures in tally:
        eps += n_eps
        gen += n_gen
        reuse += n_reuse
        fail += len(failures)
        print(f"  {name:<28} {n_eps:>4} episodes: generated {n_gen:>4}, "
              f"reused {n_reuse:>4}, failed {len(failures):>3}")

    print("-" * 74)
    print(f"  {len(tally)} world(s), {eps} episodes: {gen} generated, {reuse} reused, {fail} failed")

    if fail:
        print(f"\n  {fail} FAILURE(S) -- episode and reason:")
        for name, _, _, _, failures in tally:
            for ep_id, reason in failures:
                print(f"    {name}/episode_{ep_id:06d}: {reason}")
        print("\n  Failed episodes are written as FAILED_GENERATION_EP<id>, which counts as")
        print("  absent, so a plain rerun retries exactly those and leaves the rest alone.")
    print("=" * 74)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-path", required=True,
                   help="grscenes root (all worlds) or one world dir")
    # Camera height/pitch are not flags -- see CAMERA_HEIGHT_CM / CAMERA_PITCH_DEG in
    # process_raw_to_dataset.py, which this imports.
    p.add_argument(
        "--instructions",
        help="Optional: pre-generated instructions JSON (skip VLM). Format: {episode_id: instruction}"
    )
    p.add_argument(
        "--redo", action="store_true",
        help="Regenerate instructions that already exist (default: reuse them, so a rerun "
             "only pays for new episodes)"
    )
    p.add_argument(
        "--use-vlm",
        action="store_true",
        help="Call Gemini VLM to generate instructions (requires API_KEY). If not set, uses placeholders."
    )
    cfg = p.parse_args()

    # Handle multiple scenes
    if os.path.isdir(os.path.join(cfg.data_path, "videos")):
        # Single scene
        scenes = [cfg.data_path]
    else:
        # Multiple scenes
        scenes = [
            os.path.join(cfg.data_path, d)
            for d in sorted(os.listdir(cfg.data_path))
            if os.path.isdir(os.path.join(cfg.data_path, d))
        ]

    # Per-world tallies for the closing summary.
    tally = []      # (scene_name, n_episodes, n_generated, n_reused, failures)

    for scene in scenes:
        scene_name = os.path.basename(scene)
        print(f"\n{'='*60}")
        print(f"Processing scene: {scene_name}")
        print(f"{'='*60}")

        # Discover episodes from preview videos
        try:
            episodes = discover_episodes(scene)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        if not episodes:
            print(f"[SKIP] No episodes found in {scene}")
            continue

        print(f"Found {len(episodes)} episodes")

        # Generate or load instructions
        if cfg.instructions:
            # Load pre-generated instructions
            instr_path = cfg.instructions
            if os.path.isdir(instr_path):
                instr_path = os.path.join(instr_path, f"{scene_name}.json")

            if os.path.exists(instr_path):
                with open(instr_path, encoding="utf-8") as f:
                    instructions_raw = json.load(f)
                # Ensure keys are int
                instructions = {int(k): v for k, v in instructions_raw.items()}
                reused, failures = 0, []
                print(f"Loaded {len(instructions)} instructions from {instr_path}")
            else:
                print(f"[WARN] {instr_path} not found")
                if cfg.use_vlm:
                    print("Generating with VLM instead...")
                    instructions, reused, failures = batch_generate_instructions(
                        scene, episodes, True, cfg.redo)
                else:
                    print("Using placeholders (pass --use-vlm to generate)")
                    instructions, reused, failures = batch_generate_instructions(
                        scene, episodes, False, cfg.redo)
        elif cfg.use_vlm:
            # Generate with VLM
            if not API_KEY:
                print("[ERROR] --use-vlm requires API_KEY environment variable")
                print("   export API_KEY=sk-...")
                continue
            instructions, reused, failures = batch_generate_instructions(
                scene, episodes, True, cfg.redo)
        else:
            # Use placeholders
            print("Using placeholder instructions (pass --use-vlm or --instructions to generate)")
            instructions, reused, failures = batch_generate_instructions(
                scene, episodes, False, cfg.redo)

        # Write tasks.jsonl and episodes.jsonl
        write_tasks_and_episodes(scene, episodes, instructions)

        n_generated = len(episodes) - reused - len(failures)
        tally.append((scene_name, len(episodes), n_generated, reused, failures))

        print(f"\n{'='*60}")
        print(f"✓ {scene_name} complete")
        print(f"{'='*60}")

    print_summary(tally, cfg)


if __name__ == "__main__":
    main()
