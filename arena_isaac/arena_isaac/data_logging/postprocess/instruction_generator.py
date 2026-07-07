"""
========================================
VLN Instruction Generator
Vision-Language Navigation tool for egocentric robot videos
Powered by DMXAPI Gemini Video Analysis API
========================================

Description:
    Analyzes egocentric robot navigation videos and automatically
    generates natural language instructions in R2R (Room-to-Room) format.
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

MODEL = "gemini-2.5-pro"
API_URL = f"https://jkwl.dmxapi.cn/v1beta/models/{MODEL}:generateContent"
# Read the DMXAPI key from the environment; never hard-code it in source.
#   export API_KEY=sk-...   (auto-loaded from /opt/arena_ws/.env when sourced)
API_KEY = os.environ.get("API_KEY", "")

# All paths are anchored to this script's directory, so you can call the
# script from any working directory and paths will still resolve correctly.
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ========================================
# VLN Prompt
# ========================================

PROMPT = """
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
# Utility Functions
# ========================================

def encode_video(video_path: str) -> str | None:
    """
    Encode a video file to a Base64 string.

    The Gemini API expects video data embedded directly in the JSON request body
    (inlineData). HTTP and JSON can only carry text, not raw binary, so we use
    Base64 encoding to convert the binary video bytes into a text-safe ASCII string.
    The trade-off is a ~33% increase in payload size.

    Args:
        video_path: Path to the video file.

    Returns:
        Base64-encoded string, or None on failure.
    """
    try:
        with open(video_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  Encoded successfully | File size: {size_mb:.1f} MB")
        return data
    except Exception as e:
        print(f"  Encoding failed: {e}")
        return None


def get_mime_type(video_path: str) -> str:
    """Return the MIME type for a given video file extension."""
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
    """
    Send a video analysis request to the DMXAPI Gemini endpoint.

    Retries automatically on timeout or transient network errors.

    Args:
        video_data:  Base64-encoded video string.
        mime_type:   MIME type of the video (e.g. "video/mp4").
        prompt:      VLN instruction prompt to send alongside the video.
        max_retries: Maximum number of retry attempts.

    Returns:
        Parsed JSON response dict, or None if all attempts fail.
    """
    headers = {"Content-Type": "application/json"}
    if not API_KEY:
        print("  API_KEY not set; export API_KEY=sk-... before running.")
        return None
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
            "temperature": 0.2,     # Low temperature for consistent, deterministic instructions
            "topP": 0.8,
            "maxOutputTokens": 2048
        }
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Sending request (attempt {attempt}/{max_retries})...")
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
            print(f"  Request timed out, retrying...")
            time.sleep(5 * attempt)
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}")
            if hasattr(e, "response") and e.response:
                print(f"     Status code : {e.response.status_code}")
                print(f"     Response    : {e.response.text[:300]}")
            if attempt < max_retries:
                time.sleep(3 * attempt)

    return None


def extract_text_from_response(response: dict) -> str | None:
    """Extract the generated text from a Gemini API response dict."""
    try:
        candidates = response.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
    except Exception as e:
        print(f"  Response parsing error: {e}")
    return None


def parse_json_from_text(text: str) -> dict | None:
    """
    Fault-tolerant JSON extraction from raw model output.

    Tries three strategies in order:
      1. Direct parse (model returned clean JSON).
      2. Extract from ```json ... ``` fenced code block.
      3. Extract the first {...} block found anywhere in the text.
    """
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

    # Strategy 3: first { ... } in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def print_usage_stats(response: dict):
    """Print token usage statistics from the API response."""
    usage = response.get("usageMetadata", {})
    if usage:
        print(f"\n  Token usage:")
        print(f"     Total  : {usage.get('totalTokenCount', 0)}")
        print(f"     Input  : {usage.get('promptTokenCount', 0)}")
        print(f"     Output : {usage.get('candidatesTokenCount', 0)}")


# ========================================
# Core Analysis Function
# ========================================

def analyze_video_for_vln(
    video_path: str,
    save_output: bool = True
) -> dict | None:
    """
    Analyze a single egocentric robot video and generate R2R-format VLN instructions.

    Path resolution:
        Relative paths are resolved from SCRIPT_DIR (the folder containing
        this file), NOT from your current working directory.

    Args:
        video_path:  Path to the video file.
                       Relative: "test_videos/nav_01.mp4"  ->  <SCRIPT_DIR>/test_videos/nav_01.mp4
                       Absolute: "/data/robot/nav_01.mp4"  ->  used as-is
        save_output: If True, save the result JSON to OUTPUT_DIR.

    Returns:
        Dict containing raw text and parsed JSON result, or None on failure.
    """
    # Resolve relative paths relative to this script's directory
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = SCRIPT_DIR / video_path

    print(f"\n{'='*60}")
    print(f"Video   : {video_path.name}")
    print(f"{'='*60}")

    if not video_path.exists():
        print(f"File not found: {video_path}")
        return None

    print("Encoding video...")
    video_data = encode_video(str(video_path))
    if not video_data:
        return None

    mime_type = get_mime_type(str(video_path))

    response = call_gemini_api(video_data, mime_type, PROMPT)
    if not response:
        print("API call failed after all retries.")
        return None

    text = extract_text_from_response(response)
    if not text:
        print("Could not extract text from response.")
        return None

    print_usage_stats(response)

    parsed = parse_json_from_text(text)

    result = {
        "video_file": video_path.name,
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_text": text,
        "parsed_result": parsed
    }

    print(f"\n{'='*60}")
    print("VLN Result")
    print(f"{'='*60}")
    if parsed:
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    else:
        print("JSON parsing failed. Raw output:")
        print(text)

    if save_output:
        stem = video_path.stem
        out_path = OUTPUT_DIR / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {out_path}")

    return result


def batch_process(
    video_dir: str,
    extensions: list = None,
    output_jsonl: str = None
) -> list[dict]:
    """
    Process all videos in a directory and optionally export a JSONL dataset.

    Args:
        video_dir:    Directory containing video files.
                      Relative paths are resolved from SCRIPT_DIR.
        extensions:   List of accepted file extensions. Defaults to common video formats.
        output_jsonl: If provided, exports results as a .jsonl file in OUTPUT_DIR.

    Returns:
        List of result dicts for all successfully processed videos.
    """
    if extensions is None:
        extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    video_dir = Path(video_dir)
    if not video_dir.is_absolute():
        video_dir = SCRIPT_DIR / video_dir

    videos = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in extensions])

    if not videos:
        print(f"No video files found in: {video_dir}")
        return []

    print(f"\n{'='*60}")
    print(f"Batch mode")
    print(f"   Directory : {video_dir}")
    print(f"   Found     : {len(videos)} video(s)")
    print(f"{'='*60}")

    results, failed = [], []

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing...")
        result = analyze_video_for_vln(str(video))

        if result:
            results.append(result)
        else:
            failed.append(video.name)

        # Brief pause between requests to respect API rate limits
        if i < len(videos):
            time.sleep(2)

    # Export JSONL training dataset
    if output_jsonl and results:
        fname = output_jsonl if output_jsonl.endswith(".jsonl") else output_jsonl + ".jsonl"
        jsonl_path = OUTPUT_DIR / fname
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                if r.get("parsed_result"):
                    line = {
                        "video": r["video_file"],
                        "instruction": r["parsed_result"],
                        "model": r["model"],
                        "timestamp": r["timestamp"]
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"\nJSONL dataset exported: {jsonl_path}")

    print(f"\n{'='*60}")
    print(f"Batch complete")
    print(f"   Succeeded : {len(results)}/{len(videos)}")
    if failed:
        print(f"   Failed    : {failed}")
    print(f"{'='*60}")

    return results


# ========================================
# Entry Point — uncomment an example to run
# ========================================

if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(description="VLN Instruction Generator")
    _parser.add_argument("--video", required=True, help="Path to the rgb_video.mp4 to analyze")
    _parser.add_argument("--tag", default="", help="Session tag used as output filename prefix")
    _args = _parser.parse_args()

    _result = analyze_video_for_vln(
        video_path=_args.video,
    )

    # Also write a tag-prefixed copy of the result alongside the video
    if _result and _args.tag:
        _video_dir = Path(_args.video).parent
        _out = _video_dir / f"{_args.tag}_r2r.json"
        with open(_out, "w", encoding="utf-8") as _f:
            json.dump(_result, _f, indent=2, ensure_ascii=False)
        print(f"Tagged result saved to: {_out}")