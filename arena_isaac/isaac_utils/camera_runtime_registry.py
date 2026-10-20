"""Ownership-scoped camera image/TF specs and native writer handles.

The registry is deliberately independent from Isaac/Replicator imports so its
replacement and removal semantics can be tested without starting Kit.  A
record is owned by the pair ``(robot_prim_path, namespace)``; removing one
robot must never clear another live robot's camera state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


def camera_owner_token(robot_prim_path: str, namespace: str) -> str:
    robot = str(robot_prim_path).rstrip("/") or "/"
    return f"{robot}|{str(namespace).strip('/')}"


@dataclass
class CameraRuntimeRegistry:
    image_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    tf_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    handles_by_owner: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    robot_by_owner: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _robot_path(value: Any) -> str:
        return str(value).rstrip("/") or "/"

    def _record_owner(self, spec: dict[str, Any]) -> str:
        owner = str(spec.get("owner_token") or "")
        robot = self._robot_path(spec.get("robot_prim_path") or "")
        if not owner or robot == "/":
            raise ValueError("camera registry records require owner_token and robot_prim_path")
        existing = self.robot_by_owner.get(owner)
        if existing is not None and existing != robot:
            raise ValueError(
                f"camera owner {owner!r} already belongs to {existing!r}, not {robot!r}"
            )
        self.robot_by_owner[owner] = robot
        return owner

    @staticmethod
    def _storage_key(key: str, owner_token: str) -> str:
        return f"{owner_token}|{str(key)}"

    def register_image(self, key: str, spec: dict[str, Any]) -> None:
        record = dict(spec)
        owner = self._record_owner(record)
        self.image_specs[self._storage_key(key, owner)] = record

    def register_tf(self, key: str, spec: dict[str, Any]) -> None:
        record = dict(spec)
        owner = self._record_owner(record)
        self.tf_specs[self._storage_key(key, owner)] = record

    def add_handles(
        self,
        owner_token: str,
        robot_prim_path: str,
        handles: list[dict[str, Any]],
    ) -> None:
        owner = str(owner_token)
        robot = self._robot_path(robot_prim_path)
        if not owner or robot == "/":
            raise ValueError("camera handles require owner_token and robot_prim_path")
        existing = self.robot_by_owner.get(owner)
        if existing is not None and existing != robot:
            raise ValueError(
                f"camera owner {owner!r} already belongs to {existing!r}, not {robot!r}"
            )
        self.robot_by_owner[owner] = robot
        records = [dict(record) for record in handles]
        self.handles_by_owner.setdefault(owner, []).extend(records)

    def image_records(self) -> list[dict[str, Any]]:
        return [dict(spec) for spec in self.image_specs.values()]

    def tf_records(self) -> list[dict[str, Any]]:
        return [dict(spec) for spec in self.tf_specs.values()]

    def _owners_for_robot(self, robot_prim_path: str) -> set[str]:
        robot = self._robot_path(robot_prim_path)
        return {
            owner
            for owner, owned_robot in self.robot_by_owner.items()
            if owned_robot == robot
        }

    def unregister_owner(
        self,
        owner_token: str,
        *,
        detach_writer: Callable[[Any], None],
        destroy_render_product: Callable[[Any], None],
    ) -> dict[str, Any]:
        owner = str(owner_token)
        handles = list(self.handles_by_owner.get(owner, []))
        errors = []
        failed_handles = []
        detached = 0
        destroyed = 0
        # Writers must stop producing before their render products are
        # destroyed.  Registration order is not relied upon here.
        ordered = sorted(
            handles,
            key=lambda record: 0 if record.get("kind") == "writer" else 1,
        )
        for record in ordered:
            try:
                if record["kind"] == "writer":
                    detach_writer(record["value"])
                    detached += 1
                elif record["kind"] == "render_product":
                    destroy_render_product(record["value"])
                    destroyed += 1
                else:
                    raise ValueError(f"unknown camera handle kind: {record['kind']}")
            except Exception as exc:
                errors.append(f"{record.get('kind')}:{type(exc).__name__}:{exc}")
                failed_handles.append(record)
        if failed_handles:
            # Retain only failed handles and all specs.  The caller can retry
            # exact cleanup, and delete/replace paths fail closed instead of
            # losing ownership evidence for still-live resources.
            self.handles_by_owner[owner] = failed_handles
            return {
                "owner_token": owner,
                "removed_image_keys": [],
                "removed_tf_keys": [],
                "writers_detached": detached,
                "render_products_destroyed": destroyed,
                "errors": errors,
                "pass": False,
            }
        self.handles_by_owner.pop(owner, None)
        removed_image = [
            key
            for key, spec in self.image_specs.items()
            if spec.get("owner_token") == owner
        ]
        removed_tf = [
            key for key, spec in self.tf_specs.items() if spec.get("owner_token") == owner
        ]
        for key in removed_image:
            self.image_specs.pop(key, None)
        for key in removed_tf:
            self.tf_specs.pop(key, None)
        self.robot_by_owner.pop(owner, None)
        return {
            "owner_token": owner,
            "removed_image_keys": sorted(removed_image),
            "removed_tf_keys": sorted(removed_tf),
            "writers_detached": detached,
            "render_products_destroyed": destroyed,
            "errors": errors,
            "pass": not errors,
        }

    def unregister_robot(
        self,
        robot_prim_path: str,
        *,
        detach_writer: Callable[[Any], None],
        destroy_render_product: Callable[[Any], None],
    ) -> dict[str, Any]:
        owners = sorted(self._owners_for_robot(robot_prim_path))
        results = [
            self.unregister_owner(
                owner,
                detach_writer=detach_writer,
                destroy_render_product=destroy_render_product,
            )
            for owner in owners
        ]
        return {
            "robot_prim_path": str(robot_prim_path),
            "owners": owners,
            "results": results,
            "pass": all(item["pass"] for item in results),
        }
