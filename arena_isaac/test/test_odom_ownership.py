import pytest

from isaac_utils.odom_ownership import (
    ENV_NAME,
    LEGACY_GRAPH_ENV_NAME,
    OdomOwnershipRegistry,
    _reset_odom_ownership_for_tests,
    assess_graph_disable_nodes,
    freeze_odom_backend,
    isaac_graph_enabled,
    manual_step_enabled,
    odom_owner_token,
    odom_publish_backend,
    validate_odom_evidence,
)


@pytest.fixture(autouse=True)
def reset_global_backend():
    _reset_odom_ownership_for_tests()
    yield
    _reset_odom_ownership_for_tests()


def _spec(robot="/World/Robots/A", namespace="task_generator_node/A"):
    name = namespace.rsplit("/", 1)[-1]
    return {
        "owner_token": odom_owner_token(robot, namespace),
        "robot_prim_path": robot,
        "namespace": namespace,
        "topic": f"/{namespace}/odom",
        "world_frame": f"{name}/world",
        "odom_frame": f"{name}/odom",
        "base_frame": f"{name}/base_footprint",
        "source_prim_path": f"{robot}/Ai2_Bot2_Chassis/base_footprint",
        "backend": "manual_step",
    }


def test_project_default_is_manual_step_and_mutually_exclusive():
    assert odom_publish_backend({}) == "manual_step"
    assert manual_step_enabled("manual_step") is True
    assert isaac_graph_enabled("manual_step") is False


def test_graph_backend_disables_manual_owner_path():
    assert odom_publish_backend({ENV_NAME: "isaac_graph"}) == "isaac_graph"
    assert manual_step_enabled("isaac_graph") is False
    assert isaac_graph_enabled("isaac_graph") is True


def test_legacy_graph_switch_cannot_reenable_second_owner():
    assert odom_publish_backend({LEGACY_GRAPH_ENV_NAME: "1"}) == "manual_step"
    assert odom_publish_backend(
        {ENV_NAME: "manual_step", LEGACY_GRAPH_ENV_NAME: "1"}
    ) == "manual_step"
    assert odom_publish_backend(
        {ENV_NAME: "isaac_graph", LEGACY_GRAPH_ENV_NAME: "0"}
    ) == "isaac_graph"


@pytest.mark.parametrize("value", ["dual", "auto", "both", ""])
def test_unknown_or_dual_backend_is_rejected(value):
    with pytest.raises(ValueError, match="dual/auto ownership is forbidden"):
        odom_publish_backend({ENV_NAME: value})


def test_backend_is_process_frozen():
    contract = freeze_odom_backend({ENV_NAME: "manual_step"})
    assert contract["topic_owner"] == "ManualStepStatePublisher"
    with pytest.raises(RuntimeError, match="already frozen"):
        freeze_odom_backend({ENV_NAME: "isaac_graph"})


def test_same_topic_cannot_have_two_robot_owners():
    registry = OdomOwnershipRegistry()
    first = _spec()
    registry.register(first)
    second = _spec(robot="/World/Robots/B", namespace="task_generator_node/B")
    second["topic"] = first["topic"]
    with pytest.raises(ValueError, match="already owned"):
        registry.register(second)


def test_same_robot_respawn_requires_unregister_and_removes_stale_owner():
    registry = OdomOwnershipRegistry()
    first = _spec()
    registry.register(first)
    changed = dict(first, source_prim_path="/World/Robots/A/new_base")
    with pytest.raises(ValueError, match="must be unregistered"):
        registry.register(changed)
    result = registry.unregister_robot(first["robot_prim_path"])
    assert result["pass"] is True
    assert registry.records() == []
    registry.register(changed)
    assert registry.records()[0]["source_prim_path"].endswith("new_base")


def test_two_robots_remove_only_owned_record_and_cleanup():
    registry = OdomOwnershipRegistry()
    a = _spec()
    b = _spec(robot="/World/Robots/B", namespace="task_generator_node/B")
    registry.register(a)
    registry.register(b)
    events = []
    registry.add_cleanup(a["owner_token"], lambda: events.append("destroy-a"))
    registry.add_cleanup(b["owner_token"], lambda: events.append("destroy-b"))
    result = registry.unregister_robot(a["robot_prim_path"])
    assert result["pass"] is True
    assert events == ["destroy-a"]
    assert registry.records() == [b]


def test_cleanup_failure_retains_owner_and_blocks_replacement():
    registry = OdomOwnershipRegistry()
    spec = _spec()
    registry.register(spec)

    def fail():
        raise RuntimeError("publisher GID still live")

    registry.add_cleanup(spec["owner_token"], fail)
    result = registry.unregister_robot(spec["robot_prim_path"])
    assert result["pass"] is False
    assert registry.records() == [spec]
    with pytest.raises(ValueError, match="must be unregistered"):
        registry.register(dict(spec, source_prim_path="/new"))


def _graph_node(
    *,
    path="/World/Robot/Graph/PublishTf",
    node_type="isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
    had_topic_name=True,
    set_succeeded=True,
    dead_topic="/__arena_disabled_odom_tf/node",
    authored_after="/__arena_disabled_odom_tf/node",
    runtime_after="/__arena_disabled_odom_tf/node",
    errors=None,
):
    return {
        "node_path": path,
        "node_type": node_type,
        "had_topic_name": had_topic_name,
        "dead_topic": dead_topic,
        "set_succeeded": set_succeeded,
        "authored_topic_after": authored_after,
        "runtime_topic_after": runtime_after,
        "errors": list(errors or []),
    }


def test_tf_without_topic_name_passes_only_after_dead_topic_readback():
    result = assess_graph_disable_nodes(
        [_graph_node(had_topic_name=False, runtime_after=None)]
    )
    assert result["pass"] is True
    assert result["nodes"][0]["authored_verified"] is True
    assert result["nodes"][0]["runtime_verified"] is True


def test_tf_disable_exception_retains_owner_and_blocks_replacement():
    registry = OdomOwnershipRegistry()
    spec = _spec()
    registry.register(spec)
    graph_result = assess_graph_disable_nodes(
        [
            _graph_node(
                set_succeeded=False,
                authored_after="/tf",
                runtime_after="/tf",
                errors=["set:RuntimeError:fixture"],
            )
        ]
    )
    result = registry.unregister_robot_after_graph_disable(
        spec["robot_prim_path"], graph_result
    )
    assert result["pass"] is False
    assert result["registry_cleanup_skipped"] is True
    assert result["owner_retained"] is True
    assert registry.records() == [spec]
    with pytest.raises(ValueError, match="must be unregistered"):
        registry.register(dict(spec, source_prim_path="/new"))


def test_partial_odom_tf_disable_retains_owner():
    registry = OdomOwnershipRegistry()
    spec = _spec()
    registry.register(spec)
    graph_result = assess_graph_disable_nodes(
        [
            _graph_node(
                path="/World/Robot/Graph/PublishOdom",
                node_type="isaacsim.ros2.bridge.ROS2PublishOdometry",
            ),
            _graph_node(
                path="/World/Robot/Graph/PublishTf",
                set_succeeded=True,
                authored_after="/__arena_disabled_odom_tf/node",
                runtime_after="/tf",
            ),
        ]
    )
    assert graph_result["disabled_and_verified_count"] == 1
    assert graph_result["failed_node_count"] == 1
    result = registry.unregister_robot_after_graph_disable(
        spec["robot_prim_path"], graph_result
    )
    assert result["pass"] is False
    assert registry.records() == [spec]


def test_all_odom_tf_nodes_verified_then_exact_owner_cleanup_runs():
    registry = OdomOwnershipRegistry()
    spec = _spec()
    registry.register(spec)
    events = []
    registry.add_cleanup(spec["owner_token"], lambda: events.append("destroy-gid"))
    graph_result = assess_graph_disable_nodes(
        [
            _graph_node(node_type="isaacsim.ros2.bridge.ROS2PublishOdometry"),
            _graph_node(had_topic_name=False, runtime_after=None),
        ]
    )
    result = registry.unregister_robot_after_graph_disable(
        spec["robot_prim_path"], graph_result
    )
    assert result["pass"] is True
    assert result["registry_cleanup_skipped"] is False
    assert events == ["destroy-gid"]
    assert registry.records() == []


def test_v3_dual_frame_fixture_fails_deterministically():
    samples = [
        {
            "topic": "/task_generator_node/Ai2_Bot2/odom",
            "frame_id": "Ai2_Bot2/odom",
            "child_frame_id": "Ai2_Bot2/base_link",
            "stamp_ns": index + 1,
        }
        for index in range(1015)
    ] + [
        {
            "topic": "/task_generator_node/Ai2_Bot2/odom",
            "frame_id": "Ai2_Bot2/odom",
            "child_frame_id": "Ai2_Bot2/base_footprint",
            "stamp_ns": 2000 + index,
        }
        for index in range(71)
    ]
    result = validate_odom_evidence(
        samples=samples,
        publisher_gids=["gid-manual", "gid-graph"],
        tf_edges=[
            ["Ai2_Bot2/world", "Ai2_Bot2/odom"],
            ["Ai2_Bot2/odom", "Ai2_Bot2/base_link"],
            ["Ai2_Bot2/odom", "Ai2_Bot2/base_footprint"],
        ],
        expected_topic="/task_generator_node/Ai2_Bot2/odom",
        expected_world_frame="Ai2_Bot2/world",
        expected_odom_frame="Ai2_Bot2/odom",
        expected_base_frame="Ai2_Bot2/base_footprint",
    )
    assert result["pass"] is False
    assert result["frame_pair_counts"] == {
        "Ai2_Bot2/odom|Ai2_Bot2/base_footprint": 71,
        "Ai2_Bot2/odom|Ai2_Bot2/base_link": 1015,
    }
    assert result["predicates"]["single_frame_pair"]["pass"] is False
    assert result["predicates"]["unique_publisher_gid"]["pass"] is False


def test_single_authoritative_pair_monotonic_gid_and_tf_pass():
    samples = [
        {
            "topic": "/task_generator_node/Ai2_Bot2/odom",
            "frame_id": "Ai2_Bot2/odom",
            "child_frame_id": "Ai2_Bot2/base_footprint",
            "stamp_ns": (index + 1) * 10,
        }
        for index in range(10)
    ]
    result = validate_odom_evidence(
        samples=samples,
        publisher_gids=["one-gid", "one-gid"],
        tf_edges=[
            ["Ai2_Bot2/world", "Ai2_Bot2/odom"],
            ["Ai2_Bot2/odom", "Ai2_Bot2/base_footprint"],
        ],
        expected_topic="/task_generator_node/Ai2_Bot2/odom",
        expected_world_frame="Ai2_Bot2/world",
        expected_odom_frame="Ai2_Bot2/odom",
        expected_base_frame="Ai2_Bot2/base_footprint",
    )
    assert result["pass"] is True
    assert all(item["pass"] for item in result["predicates"].values())


def test_duplicate_or_regressing_stamp_fails():
    samples = [
        {
            "topic": "/odom",
            "frame_id": "odom",
            "child_frame_id": "base_footprint",
            "stamp_ns": value,
        }
        for value in (10, 20, 20, 19)
    ]
    result = validate_odom_evidence(
        samples=samples,
        publisher_gids=["gid"],
        tf_edges=[["world", "odom"], ["odom", "base_footprint"]],
        expected_topic="/odom",
        expected_world_frame="world",
        expected_odom_frame="odom",
        expected_base_frame="base_footprint",
    )
    assert result["pass"] is False
    assert result["predicates"]["strictly_monotonic_stamp"]["pass"] is False
