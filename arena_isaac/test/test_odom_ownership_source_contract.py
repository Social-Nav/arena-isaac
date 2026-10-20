from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "arena_isaac/run_isaacsim.py"
SPAWN = ROOT / "arena_isaac/services/SpawnUsdRobot.py"
DELETE = ROOT / "arena_isaac/services/DeletePrims.py"


def test_runner_publishes_only_registry_selected_manual_records():
    source = RUNNER.read_text(encoding="utf-8")
    assert "get_odom_ownership_specs(backend='manual_step')" in source
    assert "freeze_odom_backend()" in source
    assert "f'{robot_name}/odom', f'{robot_name}/base_link'" not in source


def test_spawn_uses_one_exclusive_backend_and_registers_actual_frames():
    source = SPAWN.read_text(encoding="utf-8")
    assert "if isaac_graph_enabled(odom_backend):" in source
    assert "register_odom_owner(" in source
    assert "base_frame=fq_base_frame" in source
    assert "source_prim_path=odom_prim_path" in source
    assert "enable_isaac_odom_graph =" not in source
    assert "assess_graph_disable_nodes(node_results)" in source
    assert "runtime_topic_after" in source
    assert "live_node.get_attribute('inputs:topicName')" in source
    assert "unregister_odom_robot_after_graph_disable" in source


def test_respawn_and_delete_cleanup_odom_before_prim_reuse_or_destroy():
    spawn_source = SPAWN.read_text(encoding="utf-8")
    delete_source = DELETE.read_text(encoding="utf-8")
    spawn_body = spawn_source.split("def spawn_usd_robot(", 1)[1]
    assert spawn_body.index("unregister_odom_robot(") < spawn_body.index("create_prim_safe(")
    assert delete_source.index("unregister_odom_robot(") < delete_source.index("IsaacSimDestroyPrim")
