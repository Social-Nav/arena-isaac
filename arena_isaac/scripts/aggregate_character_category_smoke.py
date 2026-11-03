#!/usr/bin/env python3
"""Aggregate the four per-category character-smoke receipts.

Engineering-only verdict.  A PASS here means the spawn/control/video wiring
worked for all four scenario-facing categories; it is NOT a scientific result.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CATEGORIES = ('pedestrian', 'doctor', 'construction_worker', 'police')
MIN_UNIQUE_FRACTION = 0.99


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(category: str, receipt: dict) -> list[str]:
    blockers = []
    if receipt.get('category') != category:
        blockers.append(f'CATEGORY_MISMATCH:{receipt.get("category")}')
    sel = receipt.get('selection') or {}
    if sel.get('asset_exists') is not True:
        blockers.append(f'ASSET_EXISTENCE_NOT_CONFIRMED:{sel.get("asset_stat_status")}')
    if not sel.get('stage_prim_exists'):
        blockers.append('STAGE_PRIM_MISSING')
    if sel.get('identity_id') is None:
        blockers.append('IDENTITY_UNRESOLVED')
    if int(receipt.get('received_count') or 0) < 1:
        blockers.append('CONTROL_NOT_RECEIVED')
    if int(receipt.get('published_count') or 0) < 1:
        blockers.append('CONTROL_NOT_PUBLISHED')
    captured = int(receipt.get('captured_frames') or 0)
    unique = int(receipt.get('unique_captured_frames') or 0)
    steps = int(receipt.get('steps_requested') or 0)
    if steps and captured < steps:
        blockers.append(f'CAPTURED_FRAMES_BELOW_REQUESTED:{captured}/{steps}')
    if captured and unique < int(captured * MIN_UNIQUE_FRACTION):
        blockers.append(f'DUPLICATED_RENDERED_FRAMES:{unique}/{captured}')
    decode = receipt.get('decode') or {}
    if int(decode.get('decoded_frames') or 0) != captured:
        blockers.append(f'DECODE_COUNT_MISMATCH:{decode.get("decoded_frames")}/{captured}')
    if int(decode.get('unique_decoded_frames') or 0) != captured:
        blockers.append(f'DECODE_UNIQUE_MISMATCH:{decode.get("unique_decoded_frames")}/{captured}')
    if not receipt.get('sim_time_monotonic'):
        blockers.append('SIM_TIME_NOT_MONOTONIC')
    if float(receipt.get('moved_distance_m') or 0.0) <= 0.0:
        blockers.append('AGENT_DID_NOT_MOVE')
    xn = receipt.get('xN_sim_over_video')
    if not isinstance(xn, (int, float)) or xn <= 0.0:
        blockers.append(f'XN_NOT_POSITIVE_FINITE:{xn}')
    video = Path(str(receipt.get('video_path', '')))
    if not video.is_file():
        blockers.append('VIDEO_MISSING')
    elif receipt.get('video_sha256') != _sha256(video):
        blockers.append('VIDEO_SHA_MISMATCH')
    return blockers


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: aggregate_character_category_smoke.py <out-root>', file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    rows = []
    all_blockers = []
    for category in CATEGORIES:
        receipt_path = root / category / f'SMOKE_RECEIPT_{category}.json'
        if not receipt_path.is_file():
            all_blockers.append(f'{category}:RECEIPT_MISSING')
            rows.append({'category': category, 'status': 'MISSING',
                         'receipt_path': str(receipt_path)})
            continue
        receipt = json.loads(receipt_path.read_text())
        blockers = _check(category, receipt)
        status = 'PASS' if not blockers else 'BLOCKED'
        rows.append({
            'category': category,
            'status': status,
            'receipt_path': str(receipt_path),
            'receipt_sha256': _sha256(receipt_path),
            'identity_id': (receipt.get('selection') or {}).get('identity_id'),
            'resolved_model': (receipt.get('selection') or {}).get('resolved_model'),
            'relative_usd': (receipt.get('selection') or {}).get('relative_usd'),
            'asset_exists': (receipt.get('selection') or {}).get('asset_exists'),
            'asset_stat_status': (receipt.get('selection') or {}).get('asset_stat_status'),
            'stage_prim_exists': (receipt.get('selection') or {}).get('stage_prim_exists'),
            'published_count': receipt.get('published_count'),
            'received_count': receipt.get('received_count'),
            'captured_frames': receipt.get('captured_frames'),
            'unique_captured_frames': receipt.get('unique_captured_frames'),
            'decoded_frames': (receipt.get('decode') or {}).get('decoded_frames'),
            'unique_decoded_frames': (receipt.get('decode') or {}).get('unique_decoded_frames'),
            'video_fps': receipt.get('video_fps'),
            'sim_elapsed_sec': receipt.get('sim_elapsed_sec'),
            'wall_elapsed_sec': receipt.get('wall_elapsed_sec'),
            'video_duration_sec': receipt.get('video_duration_sec'),
            'xN_sim_over_video': receipt.get('xN_sim_over_video'),
            'sim_time_monotonic': receipt.get('sim_time_monotonic'),
            'moved_distance_m': receipt.get('moved_distance_m'),
            'video_path': receipt.get('video_path'),
            'video_sha256': receipt.get('video_sha256'),
            'blockers': blockers,
        })
        all_blockers.extend(f'{category}:{b}' for b in blockers)
    status = 'SMOKE_PASS_ENGINEERING' if not all_blockers else 'SMOKE_PARTIAL'
    aggregate = {
        'schema': 'arena.isaac.character_category_smoke.aggregate.v1',
        'status': status,
        'engineering_only': True,
        'categories': list(CATEGORIES),
        'rows': rows,
        'blockers': all_blockers,
    }
    out = root / 'AGGREGATE.json'
    out.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + '\n')
    print(f'AGGREGATE={out}')
    print(f'STATUS={status}')
    print(f'BLOCKERS={len(all_blockers)}')
    return 0 if not all_blockers else 1


if __name__ == '__main__':
    sys.exit(main())
