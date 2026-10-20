"""Episode-frozen, mutually exclusive camera image backend selection."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ENV_NAME = "ARENA_CAMERA_PUBLISH_BACKEND"
LEGACY_ENV_NAME = "ARENA_LISN_CAMERA_PUBLISH_BACKEND"
MANIFEST_ENV_NAME = "ARENA_CAMERA_BACKEND_MANIFEST"
REQUIRE_MANIFEST_ENV_NAME = "ARENA_REQUIRE_CAMERA_BACKEND_MANIFEST"
VALID_BACKENDS = {"native", "manual"}
ALIASES = {"spawnusdrobot": "native"}
_PROCESS_LOCKED_BACKEND: str | None = None


def _env_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _selected_value(environment: Mapping[str, str]) -> tuple[str, str]:
    if ENV_NAME in environment:
        return str(environment[ENV_NAME]), ENV_NAME
    if LEGACY_ENV_NAME in environment:
        return str(environment[LEGACY_ENV_NAME]), LEGACY_ENV_NAME
    return "native", "project_default"


def _canonical_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in VALID_BACKENDS:
        raise ValueError(
            f"camera backend must be one of {sorted(VALID_BACKENDS)}; "
            f"dual/auto-switch are forbidden, got {value!r}"
        )
    return normalized


def camera_publish_backend(environment: Mapping[str, str] | None = None) -> str:
    if environment is None and _PROCESS_LOCKED_BACKEND is not None:
        return _PROCESS_LOCKED_BACKEND
    environment = os.environ if environment is None else environment
    value, _source = _selected_value(environment)
    return _canonical_backend(value)


def backend_contract(backend: str) -> dict[str, Any]:
    backend = _canonical_backend(backend)
    if backend == "native":
        return {
            "backend": "native",
            "image_producer": "Replicator ROS2 native writers",
            "depth_semantics": "DistanceToImagePlane",
            "intrinsics_source": "isaacsim.ros2.bridge.read_camera_info",
            "tf_producer": "DynamicCameraTfPublisher",
            "model_input_policy": "candidate_requires_case_runtime_qualification",
        }
    return {
        "backend": "manual",
        "image_producer": "ManualReplicatorCameraPublisher",
        "depth_semantics": "distance_to_camera; nonfinite values replaced with 0.0",
        "intrinsics_source": "Replicator camera_params; fx=fy=500 fallback invalid",
        "tf_producer": "DynamicCameraTfPublisher",
        "model_input_policy": "deprecated_diagnostic_legacy_only_pending_removal",
    }


def spawnusdrobot_publishers_enabled(backend: str) -> bool:
    return _canonical_backend(backend) == "native"


def manual_publishers_enabled(backend: str) -> bool:
    return _canonical_backend(backend) == "manual"


def _atomic_json(
    path: Path, value: dict[str, Any], *, require_new: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_value = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if require_new:
            # Linking a fully flushed same-directory temporary file makes the
            # final path visible atomically and fails with EEXIST instead of
            # overwriting evidence created after the Gate0 precheck.
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass
class EpisodeCameraBackend:
    backend: str
    selection_source: str
    manifest_path: Path | None = None
    state: str = "FROZEN_BEFORE_EPISODE"
    episodes: list[str] = field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "arena.camera_backend_manifest.v1",
            **backend_contract(self.backend),
            "selection_source": self.selection_source,
            "state": self.state,
            "backend_locked": True,
            "auto_switch_allowed": False,
            "episodes": list(self.episodes),
        }

    def write_manifest(self, *, require_new: bool = False) -> dict[str, Any]:
        payload = self.manifest()
        if self.manifest_path is not None:
            _atomic_json(self.manifest_path, payload, require_new=require_new)
        return payload

    def start_episode(self, episode_id: str | int) -> dict[str, Any]:
        episode = str(episode_id)
        if episode not in self.episodes:
            self.episodes.append(episode)
        self.state = "EPISODE_ACTIVE_BACKEND_LOCKED"
        return self.write_manifest()

    def request_switch(self, _backend: str) -> None:
        raise RuntimeError("camera backend auto-switch is forbidden after freeze")


def freeze_camera_backend(
    environment: Mapping[str, str] | None = None,
) -> EpisodeCameraBackend:
    global _PROCESS_LOCKED_BACKEND
    environment = os.environ if environment is None else environment
    value, source = _selected_value(environment)
    backend = _canonical_backend(value)
    if _PROCESS_LOCKED_BACKEND is not None and _PROCESS_LOCKED_BACKEND != backend:
        raise RuntimeError(
            f"camera backend already frozen as {_PROCESS_LOCKED_BACKEND}; "
            f"cannot switch to {backend}"
        )
    manifest_value = str(environment.get(MANIFEST_ENV_NAME, "")).strip()
    manifest_path = Path(manifest_value) if manifest_value else None
    manifest_required = _env_bool(environment.get(REQUIRE_MANIFEST_ENV_NAME))
    if manifest_required:
        if manifest_path is None:
            raise RuntimeError(
                f"{REQUIRE_MANIFEST_ENV_NAME}=1 requires {MANIFEST_ENV_NAME}"
            )
        if not manifest_path.is_absolute():
            raise RuntimeError(
                f"qualified {MANIFEST_ENV_NAME} must be an absolute path"
            )
        if manifest_path.exists():
            raise RuntimeError(
                f"qualified {MANIFEST_ENV_NAME} must be new: {manifest_path}"
            )
    session = EpisodeCameraBackend(
        backend=backend,
        selection_source=source,
        manifest_path=manifest_path,
    )
    session.write_manifest(require_new=manifest_required)
    # Lock only after the required manifest has been durably written.  A
    # manifest validation/write failure therefore prevents the case without
    # poisoning a later clean startup in the same process.
    _PROCESS_LOCKED_BACKEND = backend
    return session


def _reset_camera_backend_for_tests() -> None:
    global _PROCESS_LOCKED_BACKEND
    _PROCESS_LOCKED_BACKEND = None
