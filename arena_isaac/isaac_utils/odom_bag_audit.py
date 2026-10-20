"""Read-only odometry qualification audit for a sealed rosbag2 directory."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from isaac_utils.odom_ownership import canonical_topic_name, validate_odom_evidence


def audit_odom_bag(
    *,
    bag_path: Path,
    topic: str,
    world_frame: str,
    odom_frame: str,
    base_frame: str,
) -> dict[str, Any]:
    # ROS imports are deliberately lazy: the ownership registry and its unit
    # tests remain runnable without ROS or Isaac.
    import rosbag2_py
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage

    bag_path = Path(bag_path).resolve()
    if not (bag_path / "metadata.yaml").is_file():
        raise FileNotFoundError(f"rosbag metadata missing: {bag_path}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    recorded_topics = {
        canonical_topic_name(name): name for name in topic_types
    }
    recorded_topic = recorded_topics.get(canonical_topic_name(topic))
    if recorded_topic is None or topic_types.get(recorded_topic) != "nav_msgs/msg/Odometry":
        raise ValueError(
            f"expected {topic} to be nav_msgs/msg/Odometry; "
            f"got {topic_types.get(recorded_topic) if recorded_topic else None!r}"
        )

    samples: list[dict[str, Any]] = []
    tf_edges: set[tuple[str, str]] = set()
    while reader.has_next():
        current_topic, payload, bag_timestamp = reader.read_next()
        if current_topic == recorded_topic:
            message = deserialize_message(payload, Odometry)
            stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
                message.header.stamp.nanosec
            )
            samples.append(
                {
                    "topic": canonical_topic_name(current_topic),
                    "recorded_topic": current_topic,
                    "bag_timestamp_ns": int(bag_timestamp),
                    "stamp_ns": stamp_ns,
                    "frame_id": str(message.header.frame_id),
                    "child_frame_id": str(message.child_frame_id),
                }
            )
        elif current_topic in {"/tf", "/tf_static"}:
            message = deserialize_message(payload, TFMessage)
            for transform in message.transforms:
                tf_edges.add(
                    (str(transform.header.frame_id), str(transform.child_frame_id))
                )

    result = validate_odom_evidence(
        samples=samples,
        # rosbag2's sqlite schema does not preserve live DDS endpoint GIDs.
        # Future qualification must supply a separate graph snapshot.
        publisher_gids=[],
        tf_edges=sorted(tf_edges),
        expected_topic=topic,
        expected_world_frame=world_frame,
        expected_odom_frame=odom_frame,
        expected_base_frame=base_frame,
    )
    result["bag_path"] = str(bag_path)
    result["publisher_gid_evidence"] = "NOT_AVAILABLE_IN_ROSBAG2_SQLITE"
    return result


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("odom bag audit output must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_value = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--world-frame", required=True)
    parser.add_argument("--odom-frame", required=True)
    parser.add_argument("--base-frame", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_odom_bag(
        bag_path=args.bag,
        topic=args.topic,
        world_frame=args.world_frame,
        odom_frame=args.odom_frame,
        base_frame=args.base_frame,
    )
    _write_new_json(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
