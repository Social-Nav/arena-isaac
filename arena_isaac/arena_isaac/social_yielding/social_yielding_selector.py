"""
========================================
Social Yielding Pixel-Goal Selector
Pick a yielding waypoint (pixel goal) for a robot to avoid pedestrians,
using a vision model (GPT-5.4 via DMXAPI Responses API) over the robot's snapshots.
========================================

Context / pipeline
-------------------
When the proactive-yielding trigger fires (pedestrian blocks the corridor,
no feasible global path), the sim is paused and three snapshots are captured:

    head_rgb.png     forward ego view  (usually blocked by the pedestrian)
    back_rgb.png     rear ego view     (usually where free space is)
    topdown_rgb.png  bird's-eye view centred on the robot (auxiliary context)

This script sends the three images to GPT-4o and asks it to choose ONE pixel
goal: a point in a chosen ego image (head or back) that the robot should drive
toward to yield / clear the social conflict, then return to normal navigation.

Outputs (this stage):
    1. pixel goal (u, v) + which ego camera it belongs to  (JSON)
    2. an annotated copy of that ego RGB with the pixel goal marked  (PNG, drawn
       locally with PIL from the coordinate GPT returns)

NOT done here (belongs on the Isaac side, needs depth + camera intrinsics/
extrinsics): converting the pixel goal to a map-frame coordinate.

Design mirrors data_logging/postprocess/instruction_generator.py:
config block, small utility functions, fault-tolerant JSON parsing, CLI entry.
"""

import base64
import json
import os
import re
import time
from pathlib import Path

import requests


# ========================================
# Configuration  (edit before running)
# ========================================

# GPT-5.4 vision via the DMXAPI gateway, Responses API (/v1/responses).
MODEL = "gpt-5.4"
API_URL = "https://jkwl.dmxapi.cn/v1/responses"
# Read the DMXAPI key from the environment; never hard-code it in source.
#   export API_KEY=sk-...
API_KEY = os.environ.get("API_KEY", "")

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "yielding_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Snapshot filenames inside a snapshot dir (must match snapshot_capturer.py).
HEAD_RGB = "head_rgb.png"
BACK_RGB = "back_rgb.png"
TOPDOWN_RGB = "topdown_rgb.png"


# ========================================
# Prompt
# ========================================

# The prompt is intentionally verbose about the coordinate convention and the
# JSON schema — these are the parts most likely to need tuning. Image sizes are
# injected at call time so GPT returns coordinates in the correct pixel space.
PROMPT_TEMPLATE = """
You are a social-navigation assistant for a mobile robot. The robot's forward
path is blocked by pedestrian(s) and there is no feasible forward path. You must
pick a YIELDING waypoint that lets the robot genuinely move OUT OF THE
PEDESTRIAN'S WAY — clearing the corridor so the pedestrian(s) can pass — and only
then resume its original navigation. A tiny step backward is NOT enough; the goal
should meaningfully open up space.

You are given three images from the robot at the moment of conflict:
  - Image 1 "head": forward ego RGB ({head_w}x{head_h}). Usually blocked by the pedestrian.
  - Image 2 "back": rear ego RGB ({back_w}x{back_h}). Usually where free space is.
  - Image 3 "topdown": bird's-eye view centred on the robot. USE THIS to reason
    about where the pedestrian(s) are relative to the robot and which direction
    has enough open floor to actually step aside / retreat into.

Reasoning steps:
  1. From the topdown, locate the pedestrian(s) and the robot (image centre), and
     identify the direction with the most open, drivable floor to yield into.
  2. Choose the ego camera ("head" or "back") that looks toward that open
     direction. Front is usually blocked, so it is often "back".
  3. In that ego image, pick ONE pixel goal (u, v) on the open floor that is:
     - clearly FURTHER away, not just in front of the robot — aim for a spot that
       is several metres away (roughly the mid-to-far floor region of the image,
       i.e. a point noticeably above the very bottom edge), so the robot actually
       vacates the corridor;
     - still on visible, reachable floor — NOT on a wall, furniture, person, or
       the far horizon/ceiling (those give unreliable depth). A good target sits
       on the floor plane, mid-height in the image.

Coordinate convention:
  - u = horizontal pixel, 0 = left edge, increasing right.
  - v = vertical pixel, 0 = top edge, increasing down.
  - (u, v) MUST lie inside the chosen image's resolution.
  - Balance: far enough to truly yield, but on solid visible floor (not a wall/far
    background) so the 3D position is reliable.

Output STRICTLY as JSON, no extra text:
{{
  "camera": "head" | "back",
  "pixel_goal": [u, v],
  "reason": "one concise sentence: where the pedestrian is and why this yields",
  "confidence": 0.0-1.0
}}
"""


# ========================================
# Utility Functions
# ========================================

def encode_image(image_path: str) -> str | None:
    """Base64-encode an image as a full data URI (for Responses API input_image)."""
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    try:
        ext = Path(image_path).suffix.lower()
        mime = mime_map.get(ext, "image/png")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"  Encoding failed for {image_path}: {e}")
        return None


def image_size(image_path: str) -> tuple[int, int]:
    """Return (width, height) of an image, or (0, 0) on failure."""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.width, im.height
    except Exception as e:
        print(f"  Could not read size of {image_path}: {e}")
        return 0, 0


def call_vision_api(
    images_uri: dict,
    prompt: str,
    max_retries: int = 3,
) -> dict | None:
    """
    Send the three snapshots + prompt to the model via DMXAPI Responses API.

    Args:
        images_uri: {"head": data_uri, "back": data_uri, "topdown": data_uri}
                    (any may be None).
        prompt:     the fully-formatted instruction prompt.
        max_retries: retry attempts on timeout / transient errors.

    Returns:
        Parsed JSON response dict, or None if all attempts fail.
    """
    # DMXAPI gateway expects the raw key in Authorization (no "Bearer " prefix).
    if not API_KEY:
        print("[Selector] API_KEY not set; export API_KEY=sk-... before running.")
        return None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"{API_KEY}",
    }

    # Responses API format: input=[{role, content:[input_text / input_image]}].
    # Order matters: the prompt refers to Image 1/2/3 as head/back/topdown.
    content = [{"type": "input_text", "text": prompt}]
    for tag in ("head", "back", "topdown"):
        uri = images_uri.get(tag)
        if uri:
            content.append({"type": "input_image", "image_url": uri})

    payload = {
        "model": MODEL,
        "input": [{"role": "user", "content": content}],
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Sending request (attempt {attempt}/{max_retries})...")
            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=180,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print("  Request timed out, retrying...")
            time.sleep(5 * attempt)
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"     Status code : {e.response.status_code}")
                print(f"     Response    : {e.response.text[:300]}")
            if attempt < max_retries:
                time.sleep(3 * attempt)

    return None


def extract_text_from_response(response: dict) -> str | None:
    """Extract assistant text from a Responses API response.

    Shape: {status, output: [ ... {type:"message", content:[{type:"output_text",
    text:...}]} ... ]}. Falls back gracefully across minor format variants.
    """
    try:
        output = response.get("output", [])
        if not output:
            return None
        # Prefer the message-type item; else first item.
        message = next((it for it in output if it.get("type") == "message"), output[0])
        content = message.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                return first.get("text", "")
            if isinstance(first, str):
                return first
        elif isinstance(content, str):
            return content
    except Exception as e:
        print(f"  Response parsing error: {e}")
    return None


def parse_json_from_text(text: str) -> dict | None:
    """Fault-tolerant JSON extraction (clean -> fenced block -> first {...})."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def print_usage_stats(response: dict):
    # Responses API uses input_tokens/output_tokens; fall back to chat keys.
    usage = response.get("usage", {})
    if usage:
        total = usage.get("total_tokens", 0)
        inp = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        out = usage.get("output_tokens", usage.get("completion_tokens", 0))
        print("\n  Token usage:")
        print(f"     Total  : {total}")
        print(f"     Input  : {inp}")
        print(f"     Output : {out}")


def annotate_pixel_goal(
    ego_image_path: str,
    pixel_goal: tuple[int, int],
    out_path: str,
) -> bool:
    """Draw the chosen pixel goal on a copy of the ego image (crosshair + dot)."""
    try:
        from PIL import Image, ImageDraw
        im = Image.open(ego_image_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        u, v = int(pixel_goal[0]), int(pixel_goal[1])
        r = max(6, im.width // 100)          # marker radius scales with image
        draw.ellipse([u - r, v - r, u + r, v + r], outline=(255, 0, 0), width=3)
        draw.line([u - 2 * r, v, u + 2 * r, v], fill=(255, 0, 0), width=2)
        draw.line([u, v - 2 * r, u, v + 2 * r], fill=(255, 0, 0), width=2)
        im.save(out_path)
        print(f"  Annotated image saved: {out_path}")
        return True
    except Exception as e:
        print(f"  Annotation failed: {e}")
        return False


# ========================================
# Core Function
# ========================================

def select_yielding_goal(
    snapshot_dir: str,
    save_output: bool = True,
) -> dict | None:
    """
    Pick a yielding pixel goal from a snapshot directory (head/back/topdown).

    Args:
        snapshot_dir: dir containing head_rgb.png / back_rgb.png / topdown_rgb.png.
                      Relative paths resolve from SCRIPT_DIR.
        save_output:  if True, save result JSON + annotated ego image.

    Returns:
        Result dict {camera, pixel_goal, reason, confidence, annotated_image, ...}
        or None on failure.
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = SCRIPT_DIR / snapshot_dir

    print(f"\n{'='*60}")
    print(f"Snapshot dir : {snapshot_dir}")
    print(f"{'='*60}")

    paths = {
        "head": snapshot_dir / HEAD_RGB,
        "back": snapshot_dir / BACK_RGB,
        "topdown": snapshot_dir / TOPDOWN_RGB,
    }
    # head + back are required; topdown is auxiliary.
    for tag in ("head", "back"):
        if not paths[tag].exists():
            print(f"Required image missing: {paths[tag]}")
            return None

    sizes = {tag: image_size(str(p)) for tag, p in paths.items() if p.exists()}
    images_uri = {tag: encode_image(str(p)) for tag, p in paths.items() if p.exists()}

    head_w, head_h = sizes.get("head", (0, 0))
    back_w, back_h = sizes.get("back", (0, 0))
    prompt = PROMPT_TEMPLATE.format(
        head_w=head_w, head_h=head_h, back_w=back_w, back_h=back_h,
    )

    response = call_vision_api(images_uri, prompt)
    if not response:
        print("API call failed after all retries.")
        return None

    # Responses API reports a status; anything other than "completed" means the
    # output may be missing/partial (e.g. failed, incomplete due to token limit).
    status = response.get("status")
    if status and status != "completed":
        print(f"Response status = {status} (not completed).")
        if response.get("error"):
            print(f"  API error: {response.get('error')}")

    text = extract_text_from_response(response)
    if not text:
        print("Could not extract text from response.")
        return None

    print_usage_stats(response)
    parsed = parse_json_from_text(text)

    if not parsed or "pixel_goal" not in parsed or "camera" not in parsed:
        print("JSON parse failed or missing fields. Raw output:")
        print(text)
        return None

    camera = parsed["camera"]
    u, v = parsed["pixel_goal"]
    print(f"\n  Chosen camera : {camera}")
    print(f"  Pixel goal    : ({u}, {v})")
    print(f"  Reason        : {parsed.get('reason', '')}")

    # Local annotation of the chosen ego image.
    annotated_path = None
    ego_path = paths.get(camera)
    if ego_path and ego_path.exists():
        annotated_path = str(snapshot_dir / f"{camera}_pixel_goal.png")
        annotate_pixel_goal(str(ego_path), (u, v), annotated_path)
    else:
        print(f"  Warning: chosen camera '{camera}' image not found, skipped annotation")

    result = {
        "snapshot_dir": str(snapshot_dir),
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "camera": camera,
        "pixel_goal": [u, v],
        "reason": parsed.get("reason", ""),
        "confidence": parsed.get("confidence"),
        "annotated_image": annotated_path,
        "raw_text": text,
        # NOTE: pixel -> map conversion is done on the Isaac side (needs depth +
        # camera intrinsics/extrinsics); intentionally not computed here.
        "map_goal": None,
    }

    if save_output:
        out_path = snapshot_dir / "yielding_goal.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {out_path}")

    return result


# ========================================
# Entry Point
# ========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Social Yielding Pixel-Goal Selector")
    parser.add_argument(
        "--snapshot-dir", required=True,
        help="Directory with head_rgb.png / back_rgb.png / topdown_rgb.png",
    )
    args = parser.parse_args()

    select_yielding_goal(snapshot_dir=args.snapshot_dir)
