import pytest

from isaac_utils.camera_runtime_registry import (
    CameraRuntimeRegistry,
    camera_owner_token,
)


def _register_camera(
    registry: CameraRuntimeRegistry,
    *,
    robot: str,
    namespace: str,
    key: str,
    camera_prim: str,
    writer=None,
    render_product=None,
) -> str:
    owner = camera_owner_token(robot, namespace)
    common = {
        "key": key,
        "owner_token": owner,
        "robot_prim_path": robot,
        "camera_prim_path": camera_prim,
    }
    registry.register_tf(key, {**common, "frame": f"{namespace}/camera"})
    registry.register_image(key, {**common, "topic_base": f"/{namespace}/camera"})
    handles = []
    if writer is not None:
        handles.append({"kind": "writer", "value": writer, "key": key})
    if render_product is not None:
        handles.append(
            {"kind": "render_product", "value": render_product, "key": key}
        )
    registry.add_handles(owner, robot, handles)
    return owner


def _cleanup(registry: CameraRuntimeRegistry, robot: str, events: list[str]):
    return registry.unregister_robot(
        robot,
        detach_writer=lambda value: events.append(f"detach:{value}"),
        destroy_render_product=lambda value: events.append(f"destroy:{value}"),
    )


def test_same_robot_respawn_replaces_owned_records_and_old_prim_paths():
    registry = CameraRuntimeRegistry()
    events = []
    robot = "/World/Robots/Ai2_Bot2"
    _register_camera(
        registry,
        robot=robot,
        namespace="old_ns",
        key="head",
        camera_prim=f"{robot}/old_head_camera",
        writer="old-writer",
        render_product="old-render-product",
    )
    result = _cleanup(registry, robot, events)
    assert result["pass"] is True
    assert events == ["detach:old-writer", "destroy:old-render-product"]

    _register_camera(
        registry,
        robot=robot,
        namespace="new_ns",
        key="head",
        camera_prim=f"{robot}/new_head_camera",
        writer="new-writer",
        render_product="new-render-product",
    )
    records = registry.image_records() + registry.tf_records()
    assert records
    assert all("old_head_camera" not in item["camera_prim_path"] for item in records)
    assert all("new_head_camera" in item["camera_prim_path"] for item in records)


def test_robot_removal_leaves_no_tf_or_publisher_spec():
    registry = CameraRuntimeRegistry()
    robot = "/World/Robots/Ai2_Bot2"
    _register_camera(
        registry,
        robot=robot,
        namespace="robot1",
        key="head",
        camera_prim=f"{robot}/head_camera",
    )
    assert _cleanup(registry, robot, [])["pass"] is True
    assert registry.image_records() == []
    assert registry.tf_records() == []
    assert registry.handles_by_owner == {}
    assert registry.robot_by_owner == {}


def test_two_robots_remove_only_the_owned_camera_state():
    registry = CameraRuntimeRegistry()
    robot_a = "/World/Robots/A"
    robot_b = "/World/Robots/B"
    _register_camera(
        registry,
        robot=robot_a,
        namespace="a",
        key="head",
        camera_prim=f"{robot_a}/head_camera",
        writer="writer-a",
    )
    owner_b = _register_camera(
        registry,
        robot=robot_b,
        namespace="b",
        key="head",
        camera_prim=f"{robot_b}/head_camera",
        writer="writer-b",
    )
    events = []
    assert _cleanup(registry, robot_a, events)["pass"] is True
    assert events == ["detach:writer-a"]
    assert {item["owner_token"] for item in registry.tf_records()} == {owner_b}
    assert {item["owner_token"] for item in registry.image_records()} == {owner_b}
    assert list(registry.handles_by_owner) == [owner_b]


def test_cleanup_failure_is_fail_closed_and_retains_exact_handle_for_retry():
    registry = CameraRuntimeRegistry()
    robot = "/World/Robots/A"
    owner = _register_camera(
        registry,
        robot=robot,
        namespace="a",
        key="head",
        camera_prim=f"{robot}/head_camera",
        writer="slow-writer",
        render_product="render-product",
    )

    def fail_detach(_value):
        raise RuntimeError("writer still live")

    first = registry.unregister_robot(
        robot,
        detach_writer=fail_detach,
        destroy_render_product=lambda _value: None,
    )
    assert first["pass"] is False
    assert registry.image_records()
    assert registry.tf_records()
    assert registry.handles_by_owner[owner] == [
        {"kind": "writer", "value": "slow-writer", "key": "head"}
    ]

    events = []
    second = _cleanup(registry, robot, events)
    assert second["pass"] is True
    assert events == ["detach:slow-writer"]
    assert registry.image_records() == []
    assert registry.tf_records() == []


def test_owner_token_cannot_be_rebound_to_another_robot():
    registry = CameraRuntimeRegistry()
    owner = camera_owner_token("/World/Robots/A", "shared")
    registry.register_tf(
        "head",
        {
            "owner_token": owner,
            "robot_prim_path": "/World/Robots/A",
            "camera_prim_path": "/World/Robots/A/head_camera",
        },
    )
    with pytest.raises(ValueError, match="already belongs"):
        registry.register_image(
            "head",
            {
                "owner_token": owner,
                "robot_prim_path": "/World/Robots/B",
                "camera_prim_path": "/World/Robots/B/head_camera",
            },
        )
