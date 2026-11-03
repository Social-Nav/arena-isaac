# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU contract tests for the four-category character smoke aggregator."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / 'scripts' / 'aggregate_character_category_smoke.py')
CATEGORIES = ('pedestrian', 'doctor', 'construction_worker', 'police')


def _load_module():
    spec = importlib.util.spec_from_file_location(
        'character_smoke_aggregate', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_receipt(root: Path, category: str, **overrides):
    video = root / category / f'character_spawn_smoke_{category}.mp4'
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(f'fake-video-{category}'.encode())
    receipt = {
        'category': category,
        'selection': {
            'asset_exists': True,
            'asset_stat_status': 'Result.OK',
            'stage_prim_exists': True,
            'identity_id': f'classic_{category}_01',
            'resolved_model': f'classic_{category}_01',
            'relative_usd': f'Isaac/People/{category}.usd',
        },
        'published_count': 4,
        'received_count': 4,
        'captured_frames': 120,
        'unique_captured_frames': 120,
        'steps_requested': 120,
        'decode': {'decoded_frames': 120, 'unique_decoded_frames': 120},
        'sim_time_monotonic': True,
        'moved_distance_m': 1.2,
        'xN_sim_over_video': 0.168,
        'video_path': str(video),
        'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
    }
    receipt.update(overrides)
    path = root / category / f'SMOKE_RECEIPT_{category}.json'
    path.write_text(json.dumps(receipt))
    return receipt


@pytest.fixture()
def module():
    return _load_module()


@pytest.mark.parametrize('category', CATEGORIES)
def test_valid_receipt_has_no_blockers(module, tmp_path, category):
    receipt = _make_receipt(tmp_path, category)
    assert module._check(category, receipt) == []


def test_duplicated_frames_block(module, tmp_path):
    receipt = _make_receipt(tmp_path, 'pedestrian', unique_captured_frames=60)
    blockers = module._check('pedestrian', receipt)
    assert any(b.startswith('DUPLICATED_RENDERED_FRAMES') for b in blockers)


def test_missing_control_receive_blocks(module, tmp_path):
    receipt = _make_receipt(tmp_path, 'doctor', received_count=0)
    blockers = module._check('doctor', receipt)
    assert 'CONTROL_NOT_RECEIVED' in blockers


def test_unconfirmed_asset_existence_blocks(module, tmp_path):
    receipt = _make_receipt(
        tmp_path, 'police',
        selection={
            'asset_exists': None,
            'asset_stat_status': 'STAT_ERROR:TimeoutError',
            'stage_prim_exists': True,
            'identity_id': 'classic_police_01',
        })
    blockers = module._check('police', receipt)
    assert any(
        b.startswith('ASSET_EXISTENCE_NOT_CONFIRMED') for b in blockers)


def test_video_sha_mismatch_blocks(module, tmp_path):
    receipt = _make_receipt(
        tmp_path, 'construction_worker', video_sha256='0' * 64)
    blockers = module._check('construction_worker', receipt)
    assert 'VIDEO_SHA_MISMATCH' in blockers


def test_aggregate_fails_closed_when_one_receipt_missing(tmp_path):
    for category in CATEGORIES[:-1]:
        _make_receipt(tmp_path, category)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True, text=True)
    assert result.returncode == 1
    assert 'STATUS=SMOKE_PARTIAL' in result.stdout
    aggregate = json.loads((tmp_path / 'AGGREGATE.json').read_text())
    assert aggregate['status'] == 'SMOKE_PARTIAL'
    assert any('RECEIPT_MISSING' in b for b in aggregate['blockers'])


def test_aggregate_passes_with_four_valid_receipts(tmp_path):
    for category in CATEGORIES:
        _make_receipt(tmp_path, category)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert 'STATUS=SMOKE_PASS_ENGINEERING' in result.stdout
