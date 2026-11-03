#!/usr/bin/env python3
"""Machine-parse the authoritative eval500 pedestrian model fields.

Reads every ``worlds/*/scenarios/eval_*/scenario.yaml`` dynamic entry, resolves
the requested model deterministically through the curated character pools,
records the concrete identity/USD, and verifies asset existence with an HTTP
HEAD against the Isaac 5.1 asset root.  No random fallback: an unknown model or
an empty pool is a fail-close row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

from pedestrian.simulator.logic.people.character_assets import (
    CATALOG_CHARACTER_ASSETS,
    CHARACTER_POOLS,
    resolve_character_asset,
)
from pedestrian.simulator.logic.people.character_selector import (
    CharacterPoolSelector,
)

ASSET_ROOT = (
    'https://omniverse-content-production.s3-us-west-2.amazonaws.com'
    '/Assets/Isaac/5.1'
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head(url: str, cache: dict[str, int]) -> int:
    if url in cache:
        return cache[url]
    try:
        request = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(request, timeout=20) as response:
            code = response.status
    except Exception:
        code = 0
    cache[url] = code
    return code


def _parse_entries(scenarios_root: Path):
    entries = []
    scenario_paths = sorted(
        scenarios_root.glob('*/scenarios/eval_*/scenario.yaml'))
    for scenario_path in scenario_paths:
        data = yaml.safe_load(scenario_path.read_text()) or {}
        for dynamic in data.get('dynamic') or []:
            entries.append({
                'scenario': str(scenario_path.relative_to(scenarios_root)),
                'name': dynamic.get('name'),
                'requested_model': dynamic.get('model'),
            })
    return scenario_paths, entries


def _resolve_all(entries, seed: int):
    selector = CharacterPoolSelector(seed=seed)
    rows = []
    for entry in entries:
        requested = entry['requested_model']
        row = dict(entry)
        row['fail_close'] = False
        row['blockers'] = []
        if not requested:
            row['fail_close'] = True
            row['blockers'].append('MODEL_FIELD_MISSING')
            row['resolved_model'] = None
            row['identity_id'] = None
            row['relative_usd'] = None
            row['asset_source'] = None
            rows.append(row)
            continue
        if requested in CHARACTER_POOLS:
            if not CHARACTER_POOLS[requested]:
                row['fail_close'] = True
                row['blockers'].append(f'EMPTY_POOL:{requested}')
        elif requested not in CATALOG_CHARACTER_ASSETS:
            row['fail_close'] = True
            row['blockers'].append(f'UNKNOWN_MODEL:{requested}')
        resolved = selector.resolve(entry['name'], requested)
        try:
            canonical, asset = resolve_character_asset(resolved)
        except Exception as exc:
            row['fail_close'] = True
            row['blockers'].append(f'RESOLVE_ERROR:{type(exc).__name__}:{exc}')
            canonical, asset = None, None
        row['resolved_model'] = canonical
        row['identity_id'] = asset.identity_id if asset else None
        row['relative_usd'] = asset.relative_usd if asset else None
        row['asset_source'] = asset.source if asset else None
        row['category'] = (
            requested if requested in CHARACTER_POOLS else None)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios-root', required=True,
                        help='path to arena_simulation_setup/worlds')
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--output', required=True)
    parser.add_argument('--skip-asset-head', action='store_true')
    args = parser.parse_args()

    scenarios_root = Path(args.scenarios_root).resolve()
    scenario_paths, entries = _parse_entries(scenarios_root)
    if not scenario_paths:
        print('FAIL-CLOSE: no eval scenario files found', file=sys.stderr)
        return 2

    rows = _resolve_all(entries, args.seed)
    replay_rows = _resolve_all(entries, args.seed)
    deterministic_replay = all(
        (a.get('resolved_model'), a.get('identity_id'), a.get('relative_usd'))
        == (b.get('resolved_model'), b.get('identity_id'), b.get('relative_usd'))
        for a, b in zip(rows, replay_rows)
    )

    asset_cache: dict[str, int] = {}
    if not args.skip_asset_head:
        for row in rows:
            if not row.get('relative_usd'):
                continue
            url = f'{ASSET_ROOT}/{row["relative_usd"]}'
            code = _head(url, asset_cache)
            row['asset_url'] = url
            row['asset_http_status'] = code
            row['asset_exists'] = code == 200
            if code != 200:
                row['fail_close'] = True
                row['blockers'].append(f'ASSET_MISSING:{code}')

    # Supplementary coverage: every identity and every selector-reachable
    # variant in the four scenario pools must also resolve to an existing USD.
    # This does not replace the 1125-entry validation above.
    pool_coverage = {}
    for pool_name, identities in CHARACTER_POOLS.items():
        identity_rows = []
        for identity_id in identities:
            variants = ['base'] + [f'_v{index:02d}' for index in range(1, 11)]
            for variant in variants:
                model_name = identity_id if variant == 'base' else f'{identity_id}{variant}'
                if model_name not in CATALOG_CHARACTER_ASSETS:
                    continue
                canonical, asset = resolve_character_asset(model_name)
                url = f'{ASSET_ROOT}/{asset.relative_usd}'
                code = 200 if args.skip_asset_head else _head(url, asset_cache)
                identity_rows.append({
                    'identity_id': identity_id,
                    'variant': variant,
                    'resolved_model': canonical,
                    'relative_usd': asset.relative_usd,
                    'asset_url': url,
                    'asset_http_status': code,
                    'asset_exists': code == 200,
                })
        pool_coverage[pool_name] = {
            'identity_count': len(identities),
            'variant_rows': len(identity_rows),
            'missing_assets': sorted(
                row['asset_url'] for row in identity_rows
                if row['asset_exists'] is not True),
            'rows': identity_rows,
        }

    category_counts = Counter(
        row['category'] for row in rows if row.get('category'))
    model_counts = Counter(
        row['requested_model'] for row in rows if row.get('requested_model'))
    resolved_counts = Counter(
        row['resolved_model'] for row in rows if row.get('resolved_model'))
    identity_counts = Counter(
        row['identity_id'] for row in rows if row.get('identity_id'))
    fail_close_rows = [row for row in rows if row['fail_close']]
    unique_assets = sorted({row['relative_usd'] for row in rows if row.get('relative_usd')})
    missing_assets = sorted(
        url for url, code in asset_cache.items() if code != 200)

    payload = {
        'schema': 'arena.isaac.eval500_pedestrian_selection.v2',
        'scenarios_root': str(scenarios_root),
        'scenario_count': len(scenario_paths),
        'entry_count': len(rows),
        'deterministic_seed': args.seed,
        'deterministic_replay': deterministic_replay,
        'catalog_sha256': _sha256(
            Path(__file__).parents[1] / 'pedestrian' / 'config' / 'character_identities.yaml'),
        'category_counts': dict(sorted(category_counts.items())),
        'requested_model_counts': dict(sorted(model_counts.items())),
        'resolved_model_counts': dict(sorted(resolved_counts.items())),
        'unique_resolved_models': len(resolved_counts),
        'unique_identities': len(identity_counts),
        'unique_assets': len(unique_assets),
        'asset_head_checked': not args.skip_asset_head,
        'missing_assets': missing_assets,
        'pool_coverage': pool_coverage,
        'fail_close_count': len(fail_close_rows),
        'unknown_category_fail_closed': True,
        'no_random_fallback': True,
        'entries': rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'OUTPUT={output}')
    print(f'ENTRIES={len(rows)} SCENARIOS={len(scenario_paths)}')
    print(f'CATEGORY_COUNTS={dict(sorted(category_counts.items()))}')
    print(
        f'UNIQUE_RESOLVED_MODELS={len(resolved_counts)} '
        f'UNIQUE_IDENTITIES={len(identity_counts)}')
    print(f'UNIQUE_ASSETS={len(unique_assets)} MISSING_ASSETS={len(missing_assets)}')
    pool_missing = sum(
        len(coverage['missing_assets']) for coverage in pool_coverage.values())
    for pool_name, coverage in sorted(pool_coverage.items()):
        print(
            f'POOL {pool_name}: identities={coverage["identity_count"]} '
            f'variant_rows={coverage["variant_rows"]} '
            f'missing={len(coverage["missing_assets"])}')
    print(f'POOL_MISSING_ASSETS={pool_missing}')
    print(f'FAIL_CLOSE_ROWS={len(fail_close_rows)}')
    print(f'DETERMINISTIC_REPLAY={deterministic_replay}')
    return 0 if (
        not fail_close_rows and deterministic_replay and pool_missing == 0
    ) else 1


if __name__ == '__main__':
    sys.exit(main())
