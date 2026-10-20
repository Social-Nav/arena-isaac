import json
from pathlib import Path

import pytest

import isaac_utils.camera_publish_backend as backend_module
from isaac_utils.camera_publish_backend import (
    ENV_NAME,
    LEGACY_ENV_NAME,
    MANIFEST_ENV_NAME,
    REQUIRE_MANIFEST_ENV_NAME,
    _reset_camera_backend_for_tests,
    backend_contract,
    camera_publish_backend,
    freeze_camera_backend,
    manual_publishers_enabled,
    spawnusdrobot_publishers_enabled,
)


@pytest.fixture(autouse=True)
def reset_backend_lock():
    _reset_camera_backend_for_tests()
    yield
    _reset_camera_backend_for_tests()


def test_project_default_is_native_and_mutually_exclusive():
    assert camera_publish_backend({}) == "native"
    assert manual_publishers_enabled("native") is False
    assert spawnusdrobot_publishers_enabled("native") is True


def test_explicit_manual_is_deprecated_diagnostic_legacy_only():
    assert camera_publish_backend({ENV_NAME: "manual"}) == "manual"
    assert manual_publishers_enabled("manual") is True
    assert spawnusdrobot_publishers_enabled("manual") is False
    assert backend_contract("manual")["model_input_policy"] == (
        "deprecated_diagnostic_legacy_only_pending_removal"
    )


def test_legacy_spawnusdrobot_alias_maps_to_native():
    assert camera_publish_backend({LEGACY_ENV_NAME: "spawnusdrobot"}) == "native"


def test_canonical_environment_has_precedence_over_legacy():
    assert camera_publish_backend(
        {ENV_NAME: "native", LEGACY_ENV_NAME: "manual"}
    ) == "native"


@pytest.mark.parametrize("value", ["dual", "auto", "both-ish", ""])
def test_dual_auto_and_unknown_backends_fail_closed(value):
    with pytest.raises(ValueError, match="dual/auto-switch are forbidden"):
        camera_publish_backend({ENV_NAME: value})


def test_backend_is_frozen_before_episode_and_manifest_is_updated(tmp_path):
    path = tmp_path / "camera_backend.json"
    session = freeze_camera_backend(
        {ENV_NAME: "native", MANIFEST_ENV_NAME: str(path)}
    )
    before = json.loads(path.read_text(encoding="utf-8"))
    assert before["backend"] == "native"
    assert before["state"] == "FROZEN_BEFORE_EPISODE"
    assert before["depth_semantics"] == "DistanceToImagePlane"
    assert before["tf_producer"] == "DynamicCameraTfPublisher"
    after = session.start_episode("episode-7")
    assert after["state"] == "EPISODE_ACTIVE_BACKEND_LOCKED"
    assert after["episodes"] == ["episode-7"]
    with pytest.raises(RuntimeError, match="auto-switch is forbidden"):
        session.request_switch("manual")


def test_manifest_can_be_required_for_qualified_episode():
    with pytest.raises(RuntimeError, match=MANIFEST_ENV_NAME):
        freeze_camera_backend(
            {ENV_NAME: "native", REQUIRE_MANIFEST_ENV_NAME: "1"}
        )


def test_qualified_manifest_must_be_absolute_and_new(tmp_path):
    with pytest.raises(RuntimeError, match="absolute path"):
        freeze_camera_backend(
            {
                ENV_NAME: "native",
                REQUIRE_MANIFEST_ENV_NAME: "1",
                MANIFEST_ENV_NAME: "relative/camera_backend.json",
            }
        )

    existing = tmp_path / "existing-camera-backend.json"
    existing.write_text("historical evidence\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be new"):
        freeze_camera_backend(
            {
                ENV_NAME: "native",
                REQUIRE_MANIFEST_ENV_NAME: "1",
                MANIFEST_ENV_NAME: str(existing),
            }
        )
    assert existing.read_text(encoding="utf-8") == "historical evidence\n"


def test_required_manifest_is_atomic_and_has_no_fixed_temp_file(tmp_path):
    path = tmp_path / "camera-backend.json"
    freeze_camera_backend(
        {
            ENV_NAME: "native",
            REQUIRE_MANIFEST_ENV_NAME: "1",
            MANIFEST_ENV_NAME: str(path),
        }
    )
    assert path.is_absolute()
    assert json.loads(path.read_text(encoding="utf-8"))["backend"] == "native"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_manifest_write_failure_prevents_case_and_does_not_lock_backend(
    tmp_path, monkeypatch
):
    path = tmp_path / "unwritable-by-fixture.json"

    def fail_write(_path: Path, _payload: dict, **_kwargs):
        raise OSError("fixture manifest write failure")

    monkeypatch.setattr(backend_module, "_atomic_json", fail_write)
    with pytest.raises(OSError, match="fixture manifest write failure"):
        freeze_camera_backend(
            {
                ENV_NAME: "native",
                REQUIRE_MANIFEST_ENV_NAME: "1",
                MANIFEST_ENV_NAME: str(path),
            }
        )
    # The failed qualified startup did not poison the process lock.
    session = freeze_camera_backend({ENV_NAME: "manual"})
    assert session.backend == "manual"


def test_process_lock_rejects_backend_drift():
    freeze_camera_backend({ENV_NAME: "native"})
    with pytest.raises(RuntimeError, match="already frozen"):
        freeze_camera_backend({ENV_NAME: "manual"})
