from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "arena_isaac/run_isaacsim.py"
SPAWN = ROOT / "arena_isaac/services/SpawnUsdRobot.py"
DELETE = ROOT / "arena_isaac/services/DeletePrims.py"


def test_dynamic_tf_producer_is_unconditional_and_image_backend_independent():
    source = RUNNER.read_text(encoding="utf-8")
    assert "class DynamicCameraTfPublisher" in source
    assert "dynamic_camera_tf_pub = DynamicCameraTfPublisher(controller)" in source
    assert "dynamic_camera_tf_pub.publish(sim_time)" in source
    assert "if should_render:" in source
    assert "if should_render and manual_camera_pub" not in source


def test_spawn_registers_tf_for_native_and_manual_before_image_branch():
    source = SPAWN.read_text(encoding="utf-8")
    register = source.index("_register_vln_camera_tf_spec(", source.index("def configure_camera"))
    native = source.index("if spawnusdrobot_publishers_enabled(backend):", register)
    image_spec = source.index("_register_vln_camera_publish_spec(", native)
    assert register < native < image_spec
    assert "Dynamic camera TF cannot resolve base prim" in source


def test_image_spec_registration_is_backend_independent_metadata():
    source = SPAWN.read_text(encoding="utf-8")
    start = source.index("_register_vln_camera_publish_spec(", source.index("def configure_camera"))
    # Registration must not be gated on the deprecated manual backend: the
    # native camera TF image echo reads the same topic metadata.
    preceding = source[:start]
    last_gate = preceding.rfind("if manual_publishers_enabled(backend):")
    last_native = preceding.rfind("if spawnusdrobot_publishers_enabled(backend):")
    assert last_gate < last_native
    assert "native camera TF image echo" in source


def test_manual_writers_remain_deprecated_and_native_is_not_dual():
    source = SPAWN.read_text(encoding="utf-8")
    assert "DEPRECATED manual image writer registration" in source
    assert "never created beside the native backend" in source


def test_same_robot_replacement_cleans_camera_runtime_before_prim_reuse():
    source = SPAWN.read_text(encoding="utf-8")
    spawn = source[source.index("def spawn_usd_robot(") :]
    cleanup = spawn.index("unregister_vln_camera_robot(prim_path")
    create = spawn.index("create_prim_safe(")
    assert cleanup < create


def test_delete_cleans_exact_camera_owner_before_destroy_even_if_prim_is_absent():
    source = DELETE.read_text(encoding="utf-8")
    function = source[source.index("def delete_prim(") : source.index("def delete_prims_callback(")]
    cleanup = function.index("unregister_vln_camera_robot(target")
    path_resolution = function.index("Prim.resolve_paths([target])")
    destroy = function.index('"IsaacSimDestroyPrim"')
    assert cleanup < path_resolution < destroy


def test_flat_camera_registries_are_removed_in_favor_of_owned_registry():
    source = SPAWN.read_text(encoding="utf-8")
    assert "_VLN_CAMERA_RUNTIME = CameraRuntimeRegistry()" in source
    assert "_VLN_CAMERA_REPLICATOR_HANDLES" not in source
    assert "_VLN_CAMERA_PUBLISH_SPECS" not in source
    assert "_VLN_CAMERA_TF_SPECS" not in source
