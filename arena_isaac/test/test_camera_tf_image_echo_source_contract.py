"""Source contracts for exact-stamp camera TF echo.

The native Replicator ROS2 image writer stamps frames from its own render clock,
which is not bit-identical to the main-loop ``SimulationContext.current_time``
used by DynamicCameraTfPublisher.  Exact-time tf2 lookups therefore need a TF
sample that carries the image's own header stamp.  These tests pin that
contract without importing Kit (run_isaacsim.py needs a live SimulationApp).
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "arena_isaac/run_isaacsim.py"


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_dynamic_tf_publisher_accepts_external_exact_stamp():
    source = _source()
    assert "def publish_at_stamp(self, stamp, key: str | None = None) -> int:" in source
    assert "transform.header.stamp = stamp" in source


def test_image_echo_subscribes_to_native_rgb_topic():
    source = _source()
    echo = source[source.index("class CameraTfImageEcho"):]
    assert "get_vln_camera_publish_specs" in echo
    assert "topic_base" in echo
    assert "rgb_topic_name" in echo
    assert "sensor_msgs.msg.Image" in echo
    assert "ReliabilityPolicy.BEST_EFFORT" in echo
    assert "DurabilityPolicy.VOLATILE" in echo


def test_image_echo_uses_image_header_stamp_not_sim_time():
    source = _source()
    echo = source[source.index("class CameraTfImageEcho"):source.index("class ManualReplicatorCameraPublisher")]
    assert "self._tf_publisher.publish_at_stamp(msg.header.stamp, key=key)" in echo
    assert "publish(sim_time" not in echo
    assert "tolerance" not in echo.lower()


def test_image_echo_is_instantiated_next_to_the_dynamic_tf_publisher():
    source = _source()
    dynamic = source.index("dynamic_camera_tf_pub = DynamicCameraTfPublisher(controller)")
    echo = source.index("CameraTfImageEcho(controller, dynamic_camera_tf_pub)", dynamic)
    assert dynamic < echo


def test_image_echo_subscribes_lazily_after_robot_spawn():
    source = _source()
    assert "def sync(self) -> None:" in source
    assert "key in self._subscribed_keys" in source
    assert "camera_tf_image_echo.sync()" in source
    # The registry is populated by the SpawnUsdRobot service, after construction.
    assert "camera_tf_image_echo.sync()" in source[source.index("while simulation_app.is_running():"):]


def test_main_loop_keeps_the_render_cadence_tf_sample():
    source = _source()
    assert "dynamic_camera_tf_pub.publish(sim_time)" in source
