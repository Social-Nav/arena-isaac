"""Mutually exclusive odometry ownership and offline qualification evidence.

Exactly one backend may publish a robot's namespaced ``odom`` topic:

``manual_step``
    The Python simulation-step loop owns Odometry and world->odom->base TF.
    This is the project default because it advances with every explicit Isaac
    physics step, including headless/non-render steps.

``isaac_graph``
    The SpawnUsdRobot-created OmniGraph owns Odometry and the corresponding TF.
    The Python step publisher remains clock-only.

The registry is Isaac-independent so replacement/removal and qualification
rules can be exercised without starting Kit, ROS, DDS or a GPU.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


ENV_NAME = "ARENA_ODOM_PUBLISH_BACKEND"
LEGACY_GRAPH_ENV_NAME = "ARENA_SPAWN_USD_ROBOT_ENABLE_ISAAC_ODOM_GRAPH"
VALID_BACKENDS = {"manual_step", "isaac_graph"}
ALIASES = {
    "manual": "manual_step",
    "python": "manual_step",
    "graph": "isaac_graph",
    "omnigraph": "isaac_graph",
}
_PROCESS_LOCKED_BACKEND: str | None = None


def odom_owner_token(robot_prim_path: str, namespace: str) -> str:
    robot = str(robot_prim_path).rstrip("/") or "/"
    return f"{robot}|{str(namespace).strip('/')}"


def canonical_topic_name(value: str) -> str:
    normalized = str(value).strip()
    return "/" + normalized.lstrip("/") if normalized else ""


def _canonical_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in VALID_BACKENDS:
        raise ValueError(
            f"odom backend must be one of {sorted(VALID_BACKENDS)}; "
            f"dual/auto ownership is forbidden, got {value!r}"
        )
    return normalized


def _selected_backend(environment: Mapping[str, str]) -> tuple[str, str]:
    if ENV_NAME in environment:
        return _canonical_backend(environment[ENV_NAME]), ENV_NAME
    # The legacy launch variable historically enabled a second graph while the
    # Python publisher stayed live.  It can no longer select ownership; keeping
    # it in an old launch environment must not silently resurrect dual output.
    if LEGACY_GRAPH_ENV_NAME in environment:
        return "manual_step", f"project_default_ignores_{LEGACY_GRAPH_ENV_NAME}"
    return "manual_step", "project_default"


def odom_publish_backend(environment: Mapping[str, str] | None = None) -> str:
    if environment is None and _PROCESS_LOCKED_BACKEND is not None:
        return _PROCESS_LOCKED_BACKEND
    selected, _source = _selected_backend(os.environ if environment is None else environment)
    return selected


def freeze_odom_backend(environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    global _PROCESS_LOCKED_BACKEND
    selected, source = _selected_backend(os.environ if environment is None else environment)
    if _PROCESS_LOCKED_BACKEND is not None and _PROCESS_LOCKED_BACKEND != selected:
        raise RuntimeError(
            f"odom backend already frozen as {_PROCESS_LOCKED_BACKEND}; cannot switch to {selected}"
        )
    _PROCESS_LOCKED_BACKEND = selected
    return {
        "schema": "arena.odom_ownership_backend.v1",
        "backend": selected,
        "selection_source": source,
        "backend_locked": True,
        "dual_publish_allowed": False,
        "topic_owner": (
            "ManualStepStatePublisher"
            if selected == "manual_step"
            else "SpawnUsdRobot Isaac OmniGraph"
        ),
    }


def manual_step_enabled(backend: str | None = None) -> bool:
    return _canonical_backend(backend or odom_publish_backend()) == "manual_step"


def isaac_graph_enabled(backend: str | None = None) -> bool:
    return _canonical_backend(backend or odom_publish_backend()) == "isaac_graph"


@dataclass
class OdomOwnershipRegistry:
    records_by_owner: dict[str, dict[str, Any]] = field(default_factory=dict)
    owner_by_topic: dict[str, str] = field(default_factory=dict)
    cleanup_by_owner: dict[str, list[Callable[[], None]]] = field(default_factory=dict)

    def register(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(spec)
        owner = str(record.get("owner_token") or "")
        robot = str(record.get("robot_prim_path") or "").rstrip("/")
        topic = str(record.get("topic") or "")
        backend = _canonical_backend(str(record.get("backend") or ""))
        required = ("world_frame", "odom_frame", "base_frame", "source_prim_path")
        if not owner or not robot or not topic.startswith("/"):
            raise ValueError("odom record requires owner_token, robot_prim_path and absolute topic")
        if any(not str(record.get(key) or "") for key in required):
            raise ValueError(f"odom record missing required frame/source fields: {record!r}")
        if owner != odom_owner_token(robot, str(record.get("namespace") or "")):
            raise ValueError("odom owner_token does not match robot path and namespace")
        existing_topic_owner = self.owner_by_topic.get(topic)
        if existing_topic_owner is not None and existing_topic_owner != owner:
            raise ValueError(
                f"odom topic {topic!r} already owned by {existing_topic_owner!r}"
            )
        existing = self.records_by_owner.get(owner)
        if existing is not None and existing != record:
            raise ValueError(
                f"odom owner {owner!r} must be unregistered before replacement"
            )
        record["backend"] = backend
        self.records_by_owner[owner] = record
        self.owner_by_topic[topic] = owner
        return dict(record)

    def records(self, *, backend: str | None = None) -> list[dict[str, Any]]:
        records = [dict(record) for record in self.records_by_owner.values()]
        if backend is not None:
            canonical = _canonical_backend(backend)
            records = [record for record in records if record["backend"] == canonical]
        return sorted(records, key=lambda item: item["owner_token"])

    def add_cleanup(self, owner_token: str, callback: Callable[[], None]) -> None:
        owner = str(owner_token)
        if owner not in self.records_by_owner:
            raise ValueError(f"cannot attach cleanup to unknown odom owner {owner!r}")
        if not callable(callback):
            raise TypeError("odom cleanup handle must be callable")
        self.cleanup_by_owner.setdefault(owner, []).append(callback)

    def unregister_owner(self, owner_token: str) -> dict[str, Any]:
        owner = str(owner_token)
        record = self.records_by_owner.get(owner)
        callbacks = list(self.cleanup_by_owner.get(owner, []))
        errors: list[str] = []
        completed = 0
        remaining: list[Callable[[], None]] = []
        for callback in callbacks:
            try:
                callback()
                completed += 1
            except Exception as exc:
                errors.append(f"{type(exc).__name__}:{exc}")
                remaining.append(callback)
        if errors:
            self.cleanup_by_owner[owner] = remaining
            return {
                "owner_token": owner,
                "removed": False,
                "cleanup_completed": completed,
                "errors": errors,
                "pass": False,
            }
        self.cleanup_by_owner.pop(owner, None)
        self.records_by_owner.pop(owner, None)
        if record is not None:
            self.owner_by_topic.pop(str(record["topic"]), None)
        return {
            "owner_token": owner,
            "removed": record is not None,
            "cleanup_completed": completed,
            "errors": [],
            "pass": True,
        }

    def unregister_robot(self, robot_prim_path: str) -> dict[str, Any]:
        robot = str(robot_prim_path).rstrip("/")
        owners = sorted(
            owner
            for owner, record in self.records_by_owner.items()
            if str(record["robot_prim_path"]).rstrip("/") == robot
        )
        results = [self.unregister_owner(owner) for owner in owners]
        return {
            "robot_prim_path": robot,
            "owners": owners,
            "results": results,
            "pass": all(item["pass"] for item in results),
        }

    def unregister_robot_after_graph_disable(
        self, robot_prim_path: str, graph_disable: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Keep the owner intact unless every odom/TF graph node is dead."""

        robot = str(robot_prim_path).rstrip("/")
        owners = sorted(
            owner
            for owner, record in self.records_by_owner.items()
            if str(record["robot_prim_path"]).rstrip("/") == robot
        )
        if graph_disable.get("pass") is not True:
            return {
                "robot_prim_path": robot,
                "owners": owners,
                "results": [],
                "graph_disable": dict(graph_disable),
                "registry_cleanup_skipped": True,
                "owner_retained": bool(owners),
                "pass": False,
            }
        result = self.unregister_robot(robot)
        result["graph_disable"] = dict(graph_disable)
        result["registry_cleanup_skipped"] = False
        result["owner_retained"] = not result["pass"] and bool(owners)
        return result


_RUNTIME_REGISTRY = OdomOwnershipRegistry()


def register_odom_owner(**spec: Any) -> dict[str, Any]:
    return _RUNTIME_REGISTRY.register(spec)


def get_odom_ownership_specs(*, backend: str | None = None) -> list[dict[str, Any]]:
    return _RUNTIME_REGISTRY.records(backend=backend)


def add_odom_owner_cleanup(owner_token: str, callback: Callable[[], None]) -> None:
    _RUNTIME_REGISTRY.add_cleanup(owner_token, callback)


def unregister_odom_robot(
    robot_prim_path: str, *, fail_on_error: bool = True
) -> dict[str, Any]:
    result = _RUNTIME_REGISTRY.unregister_robot(robot_prim_path)
    if fail_on_error and not result["pass"]:
        raise RuntimeError(f"odom ownership cleanup failed for {robot_prim_path}: {result}")
    return result


def assess_graph_disable_nodes(nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a structured fail-closed ledger for targeted odom/TF publishers."""

    normalized: list[dict[str, Any]] = []
    for value in nodes:
        record = dict(value)
        dead_topic = str(record.get("dead_topic") or "")
        authored_after = str(record.get("authored_topic_after") or "")
        runtime_after_value = record.get("runtime_topic_after")
        runtime_after = (
            None if runtime_after_value is None else str(runtime_after_value)
        )
        live_node_value = record.get("live_node_present")
        live_attr_value = record.get("live_attr_present")
        presence_recorded = isinstance(live_node_value, bool) and isinstance(
            live_attr_value, bool
        )
        live_node_present = live_node_value is True
        live_attr_present = live_attr_value is True
        errors = [str(item) for item in record.get("errors", [])]
        set_succeeded = record.get("set_succeeded") is True
        authored_verified = bool(dead_topic) and authored_after == dead_topic
        if live_node_present:
            runtime_verified = live_attr_present and runtime_after == dead_topic
        else:
            runtime_verified = not live_attr_present and runtime_after is None
        runtime_verified = bool(presence_recorded and runtime_verified)
        record["errors"] = errors
        record["dead_topic"] = dead_topic
        record["authored_topic_after"] = authored_after
        record["runtime_topic_after"] = runtime_after
        record["presence_recorded"] = presence_recorded
        record["live_node_present"] = live_node_present
        record["live_attr_present"] = live_attr_present
        record["authored_verified"] = authored_verified
        record["runtime_verified"] = runtime_verified
        record["pass"] = bool(
            set_succeeded and authored_verified and runtime_verified and not errors
        )
        normalized.append(record)
    failed = [record for record in normalized if not record["pass"]]
    return {
        "schema": "arena.odom_graph_disable_ledger.v1",
        "targeted_node_count": len(normalized),
        "disabled_and_verified_count": len(normalized) - len(failed),
        "failed_node_count": len(failed),
        "nodes": normalized,
        "pass": not failed,
    }


def unregister_odom_robot_after_graph_disable(
    robot_prim_path: str,
    graph_disable: Mapping[str, Any],
    *,
    fail_on_error: bool = True,
) -> dict[str, Any]:
    result = _RUNTIME_REGISTRY.unregister_robot_after_graph_disable(
        robot_prim_path, graph_disable
    )
    if fail_on_error and not result["pass"]:
        raise RuntimeError(
            f"odom/TF graph cleanup failed for {robot_prim_path}: {result}"
        )
    return result


def validate_odom_evidence(
    *,
    samples: Sequence[Mapping[str, Any]],
    publisher_gids: Sequence[str],
    tf_edges: Sequence[Sequence[str]],
    expected_topic: str,
    expected_odom_frame: str,
    expected_base_frame: str,
    expected_world_frame: str,
) -> dict[str, Any]:
    """Return per-predicate qualification evidence without relaxing any gate."""

    normalized_samples = [dict(sample) for sample in samples]
    expected_topic = canonical_topic_name(expected_topic)
    frame_pairs = sorted(
        {
            (str(item.get("frame_id") or ""), str(item.get("child_frame_id") or ""))
            for item in normalized_samples
        }
    )
    stamps = [int(item.get("stamp_ns", -1)) for item in normalized_samples]
    gids = sorted({str(value) for value in publisher_gids if str(value)})
    edges = {(str(edge[0]), str(edge[1])) for edge in tf_edges if len(edge) == 2}
    predicates = {
        "samples_present": {
            "pass": bool(normalized_samples),
            "reason": "samples present" if normalized_samples else "no odom samples",
        },
        "topic_exact": {
            "pass": all(
                canonical_topic_name(str(item.get("topic") or "")) == expected_topic
                for item in normalized_samples
            ),
            "reason": f"expected only {expected_topic}",
        },
        "single_frame_pair": {
            "pass": frame_pairs == [(expected_odom_frame, expected_base_frame)],
            "reason": f"observed frame pairs: {frame_pairs!r}",
        },
        "strictly_monotonic_stamp": {
            "pass": bool(stamps) and all(after > before for before, after in zip(stamps, stamps[1:])),
            "reason": "header stamps must strictly increase in bag order",
        },
        "unique_publisher_gid": {
            "pass": len(gids) == 1,
            "reason": f"observed publisher GIDs: {gids!r}",
        },
        "tf_world_to_odom": {
            "pass": (expected_world_frame, expected_odom_frame) in edges,
            "reason": f"missing TF edge {expected_world_frame}->{expected_odom_frame}",
        },
        "tf_odom_to_base": {
            "pass": (expected_odom_frame, expected_base_frame) in edges,
            "reason": f"missing TF edge {expected_odom_frame}->{expected_base_frame}",
        },
    }
    return {
        "schema": "arena.odom_qualification_evidence.v1",
        "pass": all(item["pass"] for item in predicates.values()),
        "expected": {
            "topic": expected_topic,
            "world_frame": expected_world_frame,
            "odom_frame": expected_odom_frame,
            "base_frame": expected_base_frame,
        },
        "sample_count": len(normalized_samples),
        "frame_pair_counts": {
            f"{parent}|{child}": sum(
                1
                for item in normalized_samples
                if str(item.get("frame_id") or "") == parent
                and str(item.get("child_frame_id") or "") == child
            )
            for parent, child in frame_pairs
        },
        "publisher_gids": gids,
        "tf_edges": sorted([list(edge) for edge in edges]),
        "predicates": predicates,
    }


def _reset_odom_ownership_for_tests() -> None:
    global _PROCESS_LOCKED_BACKEND, _RUNTIME_REGISTRY
    _PROCESS_LOCKED_BACKEND = None
    _RUNTIME_REGISTRY = OdomOwnershipRegistry()
