#!/usr/bin/env python3
"""Minimal per-category Isaac asset-selection/spawn + control + video smoke.

One invocation spawns exactly one pedestrian for one requested category.  The
outer driver runs four fresh Isaac processes sequentially (pedestrian, doctor,
construction_worker, police) under a single shared flock, so no two categories
share a stage or a process.

This is an engineering smoke only: it proves deterministic selection, concrete
asset resolution, stage spawn, a real ROS control publish/receive, minimal
waypoint execution, real rendered frames, and a playable MP4.  It does not
start Qwen, Nav2, or a scientific episode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

CATEGORIES = ('pedestrian', 'doctor', 'construction_worker', 'police')

MODULE_ROLES = {
    'selector': 'pedestrian.simulator.logic.people.character_selector',
    'catalog': 'pedestrian.simulator.logic.people.character_assets',
    'person': 'pedestrian.simulator.logic.people.person',
    'spawn': 'arena_isaac.services.SpawnPedestrians',
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_commit() -> str:
    """Resolve the enclosing arena_isaac repo commit at runtime."""
    configured = os.environ.get('ARENA_ISAAC_SOURCE_COMMIT')
    if configured:
        return configured.strip()
    repo_root = Path(__file__).parents[2]
    try:
        return subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'UNKNOWN'


def _read_mountinfo():
    mounts = []
    try:
        text = Path('/proc/self/mountinfo').read_text()
    except OSError as exc:
        return {'status': 'UNKNOWN', 'error': str(exc), 'mounts': []}
    for line in text.splitlines():
        fields = line.split(' - ', 1)
        if len(fields) != 2:
            continue
        left = fields[0].split()
        right = fields[1].split()
        if len(left) < 5 or len(right) < 2:
            continue
        mounts.append({
            'mount_point': left[4],
            'source': right[1],
            'fstype': right[0],
            'options': left[5] if len(left) > 5 else '',
        })
    return {'status': 'OK', 'error': '', 'mounts': mounts}


def _container_id():
    try:
        text = Path('/proc/self/cgroup').read_text()
    except OSError:
        return 'UNKNOWN'
    for line in text.splitlines():
        parts = line.split(':')
        if len(parts) >= 3:
            value = parts[2].strip()
            if '/docker/' in value:
                return value.rsplit('/docker/', 1)[1].strip()
            if value.count('/') == 1 and len(value.strip('/')) == 64:
                return value.strip('/')
    return os.uname().nodename


def _provenance():
    expected_root = Path(os.environ.get(
        'ARENA_EXPECTED_SOURCE_ROOT', str(Path(__file__).parents[2]))).resolve()
    manifest_path = os.environ.get('ARENA_PROVENANCE_MANIFEST', '')
    manifest = None
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
    ledger = []
    failures = []
    for role, module_name in MODULE_ROLES.items():
        try:
            spec = importlib.util.find_spec(module_name)
            origin = spec.origin if spec is not None else None
        except Exception as exc:
            failures.append(f'{role}:FIND_SPEC_ERROR:{type(exc).__name__}:{exc}')
            ledger.append({'role': role, 'module': module_name,
                           'origin': None, 'error': f'{type(exc).__name__}:{exc}'})
            continue
        if origin is None:
            failures.append(f'{role}:NO_ORIGIN')
            ledger.append({'role': role, 'module': module_name, 'origin': None})
            continue
        origin_path = Path(origin).resolve()
        sha = _sha256_bytes(origin_path.read_bytes())
        under = origin_path == expected_root or expected_root in origin_path.parents
        expected_sha = None
        if manifest:
            expected_sha = (manifest.get('modules', {}).get(role) or {}).get('sha256')
        ok = under and (expected_sha is None or sha == expected_sha)
        if not under:
            failures.append(f'{role}:OUTSIDE_EXPECTED_ROOT:{origin_path}')
        if expected_sha is not None and sha != expected_sha:
            failures.append(f'{role}:SHA_MISMATCH:{sha}!={expected_sha}')
        ledger.append({
            'role': role, 'module': module_name, 'origin': str(origin_path),
            'sha256': sha, 'expected_sha256': expected_sha,
            'under_expected_root': under,
        })
    harness_path = Path(__file__).resolve()
    harness_sha = _sha256_bytes(harness_path.read_bytes())
    harness_under = (
        harness_path == expected_root or expected_root in harness_path.parents)
    expected_harness = None
    if manifest:
        expected_harness = (manifest.get('modules', {}).get('harness') or {}).get('sha256')
    if not harness_under:
        failures.append(f'harness:OUTSIDE_EXPECTED_ROOT:{harness_path}')
    if expected_harness is not None and harness_sha != expected_harness:
        failures.append(f'harness:SHA_MISMATCH:{harness_sha}!={expected_harness}')
    ledger.append({
        'role': 'harness', 'module': 'validate_character_video_smoke',
        'origin': str(harness_path), 'sha256': harness_sha,
        'expected_sha256': expected_harness,
        'under_expected_root': harness_under,
    })
    mounts = _read_mountinfo()
    relevant = [
        m for m in mounts['mounts']
        if any(token in m['source'] or token in m['mount_point']
               for token in ('final_arena_isaac', 'arena_isaac_clean_integration',
                             'outputs', '/opt/arena_ws/install', '/root/.cache',
                             '/isaac-sim/kit/cache'))
    ]
    payload = {
        'status': 'PASS' if not failures else 'FAIL',
        'expected_commit': os.environ.get('ARENA_ISAAC_SOURCE_COMMIT', 'UNKNOWN'),
        'observed_commit': _source_commit(),
        'expected_source_root': str(expected_root),
        'harness_source_root': str(Path(__file__).parents[2].resolve()),
        'python_executable': sys.executable,
        'pythonpath': os.environ.get('PYTHONPATH', ''),
        'container_id': _container_id(),
        'container_image': os.environ.get(
            'ARENA_CONTAINER_IMAGE', os.environ.get('ARENA_IMAGE', 'UNKNOWN')),
        'manifest_path': manifest_path,
        'modules': ledger,
        'mounts': {'status': mounts['status'], 'error': mounts['error'],
                   'relevant': relevant},
        'failures': failures,
    }
    if manifest and failures:
        raise RuntimeError(
            'PROVENANCE_FAIL_CLOSED: ' + '; '.join(failures))
    return payload


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def _stat_asset(url: str):
    """Return (exists, status, size) for a remote Isaac asset URL."""
    try:
        import omni.client

        result, entry = omni.client.stat(url)
        code = str(result)
        size = int(getattr(entry, 'size', 0) or 0)
        exists = code.endswith('OK')
        return exists, code, size
    except Exception as exc:  # pragma: no cover - depends on Kit runtime
        return None, f'STAT_ERROR:{type(exc).__name__}:{exc}', 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--category', required=True, choices=CATEGORIES)
    parser.add_argument(
        '--model', default=None,
        help='Concrete catalog model name; defaults to the category pool name.')
    parser.add_argument('--entity', default='smoke_00')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--video-fps', type=float, default=10.0)
    parser.add_argument('--ros-domain-id', type=int, default=107)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    category = args.category
    frames_dir = out / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = out / f'character_spawn_smoke_{category}.mp4'
    receipt_path = out / f'SMOKE_RECEIPT_{category}.json'
    pts_path = out / f'video_frame_pts_{category}.json'
    pre_ledger_path = out / f'SELECTION_PRE_LEDGER_{category}.json'

    requested_model_arg = args.model or category

    app = SimulationApp({'headless': True, 'width': args.width, 'height': args.height})
    rclpy_node = None
    publisher = None
    subscriber = None
    ros_context = None
    try:
        from isaacsim.core.utils.extensions import enable_extension

        for extension in (
            'omni.anim.people',
            'omni.anim.navigation.bundle',
            'omni.anim.timeline',
            'omni.anim.graph.bundle',
            'omni.anim.retarget.bundle',
            'omni.kit.scripting',
            'isaacsim.ros2.bridge',
        ):
            enable_extension(extension)
        app.update()

        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import GroundPlane
        from isaacsim.core.utils.prims import create_prim, define_prim
        from pedestrian.simulator.logic.people.character_assets import (
            CHARACTER_POOLS,
            resolve_character_asset,
        )
        from pedestrian.simulator.logic.people.character_selector import (
            CharacterPoolSelector,
        )
        from pedestrian.simulator.logic.people.person import Person

        catalog_path = (
            Path(__file__).parents[1]
            / 'pedestrian' / 'config' / 'character_identities.yaml')
        catalog_sha = _sha256_bytes(catalog_path.read_bytes())
        source_commit = _source_commit()
        provenance = _provenance()
        omni_server = os.environ.get(
            'OMNI_SERVER',
            'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1')

        # ---- deterministic selection + pre-ledger BEFORE any spawn ----
        if category not in CHARACTER_POOLS or not CHARACTER_POOLS[category]:
            raise RuntimeError(f'FAIL-CLOSE empty/unknown category: {category}')
        selector = CharacterPoolSelector(seed=args.seed)
        resolved = selector.resolve(args.entity, requested_model_arg)
        canonical, asset = resolve_character_asset(resolved)
        if asset.identity_id not in CHARACTER_POOLS[category]:
            raise RuntimeError(
                f'FAIL-CLOSE category pool mismatch: requested={requested_model_arg} '
                f'category={category} resolved={resolved} identity={asset.identity_id}')
        asset_url = f'{omni_server}/{asset.relative_usd}'
        asset_exists, asset_stat_status, asset_size = _stat_asset(asset_url)
        selection_ledger = {
            'schema': 'arena.isaac.character_smoke.selection_pre_ledger.v1',
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'requested_category': category,
            'requested_model_arg': requested_model_arg,
            'resolved_model': canonical,
            'identity_id': asset.identity_id,
            'asset_source': asset.source,
            'relative_usd': asset.relative_usd,
            'asset_url': asset_url,
            'asset_exists': asset_exists,
            'asset_stat_status': asset_stat_status,
            'asset_size_bytes': asset_size,
            'catalog_sha256': catalog_sha,
            'source_commit': source_commit,
            'seed': args.seed,
            'entity': args.entity,
            'spawn_started': False,
        }
        _write_json(pre_ledger_path, selection_ledger)

        os.environ['ROS_DOMAIN_ID'] = str(args.ros_domain_id)
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        rclpy.init()
        ros_context = rclpy.ok()
        rclpy_node = Node('character_spawn_smoke')
        publisher = rclpy_node.create_publisher(String, '/character_smoke/target', 10)
        received = {'count': 0, 'last_data': ''}

        def _on_control(msg):
            received['count'] += 1
            received['last_data'] = msg.data

        subscriber = rclpy_node.create_subscription(
            String, '/character_smoke/target', _on_control, 10)

        omni.usd.get_context().new_stage()
        app.update()
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
        world.scene.add(GroundPlane(prim_path='/World/GroundPlane', size=30.0))
        define_prim('/World/Pedestrians', 'Xform')
        create_prim('/World/ShowcaseLight', 'DomeLight',
                    attributes={'inputs:intensity': 1100.0})

        position = [0.0, 0.0, 0.0]
        left = np.array([-1.5, 0.0, 0.0])
        right = np.array([1.5, 0.0, 0.0])
        endpoints = (right, left)
        stage_path = '/World/Pedestrians/smoke_00'
        person = Person(world, stage_path, canonical, position, 0.0)
        person.update_target_positions(list(endpoints) * 200, walk_speed=0.65)

        selection_ledger['spawn_started'] = True
        selection_ledger['stage_path'] = stage_path
        selection_ledger['targets'] = [list(map(float, e)) for e in endpoints]
        selection_ledger['walk_speed_mps'] = 0.65
        _write_json(pre_ledger_path, selection_ledger)

        control_payload = json.dumps({
            'entity': args.entity,
            'category': category,
            'model': canonical,
            'targets': [list(map(float, e)) for e in endpoints],
        }, sort_keys=True)
        msg = String()
        msg.data = control_payload
        publisher.publish(msg)
        published_count = 1

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(stage_path)
        stage_prim_exists = bool(prim and prim.IsValid())
        child_prim_count = len(prim.GetChildren()) if stage_prim_exists else 0

        world.reset()
        world.play()
        initial_position = [float(x) for x in person.position]

        camera = rep.create.camera(
            position=(0.0, -9.0, 5.0), look_at=(0.0, 0.0, 1.0),
            focal_length=30.0)
        render_product = rep.create.render_product(
            camera, (args.width, args.height))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach([render_product])

        frame_rows = []
        frame_hashes = []
        wall_start = time.monotonic()
        sim_start = float(world.current_time)
        for step in range(args.steps):
            world.step(render=True)
            rep.orchestrator.step(rt_subframes=1)
            rclpy.spin_once(rclpy_node, timeout_sec=0.0)
            if step in (30, 60, 90):
                publisher.publish(msg)
                published_count += 1
            data = rgb.get_data()
            if data is None:
                continue
            frame = np.asarray(data)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frame = frame[:, :, ::-1].copy()  # RGB -> BGR
            frame_hash = _sha256_bytes(frame.tobytes())
            frame_path = frames_dir / f'frame_{len(frame_rows):05d}.png'
            cv2.imwrite(str(frame_path), frame)
            wall_elapsed = time.monotonic() - wall_start
            sim_elapsed = float(world.current_time) - sim_start
            frame_rows.append({
                'frame': len(frame_rows),
                'sim_elapsed_sec': sim_elapsed,
                'wall_elapsed_sec': wall_elapsed,
                'frame_sha256': frame_hash,
                'png': str(frame_path),
            })
            frame_hashes.append(frame_hash)

        final_position = [float(x) for x in person.position]
        moved = [
            final_position[i] - initial_position[i] for i in range(3)
        ]
        moved_distance = float(np.linalg.norm(np.array(moved)))

        wall_elapsed = time.monotonic() - wall_start
        sim_elapsed = float(world.current_time) - sim_start
        sim_times = [row['sim_elapsed_sec'] for row in frame_rows]
        sim_time_monotonic = all(
            b >= a for a, b in zip(sim_times, sim_times[1:]))
        capture_fps = (len(frame_rows) / wall_elapsed) if wall_elapsed > 0 else 0.0
        video_fps = float(args.video_fps)
        video_duration = len(frame_rows) / video_fps if video_fps > 0 else 0.0
        xN = (sim_elapsed / video_duration) if video_duration > 0 else float('nan')

        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), video_fps,
            (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f'VideoWriter failed: {video_path}')
        for row in frame_rows:
            frame = cv2.imread(row['png'])
            if frame is None:
                raise RuntimeError(f'frame read failed: {row["png"]}')
            local_xN = (
                row['sim_elapsed_sec'] / (row['frame'] / video_fps)
                if row['frame'] > 0 and video_fps > 0 else float('nan'))
            label = (
                f'{category}  xN(sim/video)={local_xN:.3f}  '
                f'frame={row["frame"]}/{len(frame_rows)}  '
                f'sim={row["sim_elapsed_sec"]:.3f}s  wall={row["wall_elapsed_sec"]:.3f}s')
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (args.width - tw - 18, 8),
                          (args.width - 8, 16 + th + 8), (0, 0, 0), -1)
            cv2.putText(frame, label, (args.width - tw - 12, 12 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            writer.write(frame)
        writer.release()

        cap = cv2.VideoCapture(str(video_path))
        decode = {
            'opened': bool(cap.isOpened()),
            'reported_fps': float(cap.get(cv2.CAP_PROP_FPS)),
            'reported_frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'reported_duration_sec': (
                float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                / float(cap.get(cv2.CAP_PROP_FPS))
                if cap.get(cv2.CAP_PROP_FPS) else 0.0),
            'decoded_frames': 0,
            'unique_decoded_frames': 0,
        }
        decoded_hashes = []
        while True:
            ok, decoded = cap.read()
            if not ok:
                break
            decode['decoded_frames'] += 1
            decoded_hashes.append(_sha256_bytes(decoded.tobytes()))
        cap.release()
        decode['unique_decoded_frames'] = len(set(decoded_hashes))

        pts_payload = {
            'schema': 'arena.isaac.character_smoke.video_pts.v2',
            'category': category,
            'video_fps': video_fps,
            'capture_fps': capture_fps,
            'sim_elapsed_sec': sim_elapsed,
            'wall_elapsed_sec': wall_elapsed,
            'video_duration_sec': video_duration,
            'xN_sim_over_video': xN,
            'sim_time_monotonic': sim_time_monotonic,
            'unique_captured_frames': len(set(frame_hashes)),
            'captured_frames': len(frame_rows),
            'frames': frame_rows,
        }
        _write_json(pts_path, pts_payload)

        receipt = {
            'schema': 'arena.isaac.character_spawn_smoke.v2',
            'category': category,
            'source_commit': source_commit,
            'catalog_sha256': catalog_sha,
            'script_path': str(Path(__file__).resolve()),
            'script_sha256': _sha256_bytes(Path(__file__).read_bytes()),
            'provenance': provenance,
            'isaac_version': Path('/isaac-sim/VERSION').read_text().strip()
            if Path('/isaac-sim/VERSION').is_file() else '',
            'hostname': os.uname().nodename,
            'python_executable': sys.executable,
            'omni_server': omni_server,
            'ros_domain_id': args.ros_domain_id,
            'control_topic': '/character_smoke/target',
            'control_type': 'std_msgs/msg/String',
            'control_payload': control_payload,
            'published_count': published_count,
            'received_count': received['count'],
            'received_last_data': received['last_data'],
            'seed': args.seed,
            'steps_requested': args.steps,
            'selection': {
                'requested_category': category,
                'requested_model_arg': requested_model_arg,
                'resolved_model': canonical,
                'identity_id': asset.identity_id,
                'asset_source': asset.source,
                'relative_usd': asset.relative_usd,
                'asset_url': asset_url,
                'asset_exists': asset_exists,
                'asset_stat_status': asset_stat_status,
                'asset_size_bytes': asset_size,
                'stage_path': stage_path,
                'stage_prim_exists': stage_prim_exists,
                'stage_child_prim_count': child_prim_count,
                'targets': [list(map(float, e)) for e in endpoints],
                'walk_speed_mps': 0.65,
            },
            'initial_position': initial_position,
            'final_position': final_position,
            'moved_delta': moved,
            'moved_distance_m': moved_distance,
            'captured_frames': len(frame_rows),
            'unique_captured_frames': len(set(frame_hashes)),
            'video_fps': video_fps,
            'capture_fps': capture_fps,
            'sim_elapsed_sec': sim_elapsed,
            'wall_elapsed_sec': wall_elapsed,
            'video_duration_sec': video_duration,
            'xN_sim_over_video': xN,
            'sim_time_monotonic': sim_time_monotonic,
            'video_path': str(video_path),
            'video_sha256': _sha256_bytes(video_path.read_bytes()),
            'pts_path': str(pts_path),
            'pre_ledger_path': str(pre_ledger_path),
            'decode': decode,
            'cleanup': [
                'writer.release',
                'cap.release',
                'rgb.detach',
                'render_product.destroy',
                'subscriber.destroy',
                'publisher.destroy',
                'rclpy_node.destroy_node',
                'rclpy.shutdown',
                'app.close',
            ],
        }
        _write_json(receipt_path, receipt)
        print('SMOKE_CATEGORY=' + category, flush=True)
        print('SMOKE_RECEIPT=' + str(receipt_path), flush=True)
        print('SMOKE_VIDEO=' + str(video_path), flush=True)
        print('SMOKE_XN=' + str(xN), flush=True)
        print('SMOKE_FRAMES=' + str(len(frame_rows)), flush=True)
        print('SMOKE_RECEIVED=' + str(received['count']), flush=True)
        print('SMOKE_MOVED_M=' + str(moved_distance), flush=True)
        return 0
    finally:
        try:
            if subscriber is not None:
                subscriber.destroy()
            if publisher is not None:
                publisher.destroy()
            if rclpy_node is not None:
                rclpy_node.destroy_node()
            if ros_context:
                import rclpy
                rclpy.shutdown()
        except Exception:
            pass
        app.close()


if __name__ == '__main__':
    sys.exit(main())
