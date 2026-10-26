"""Gate the final robot velocity and HuNav motion during initial yielding replan."""

import os

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Int16
from std_srvs.srv import Trigger


ROBOT = os.environ.get("ARENA_ROBOT", "").strip() or "Ai2_Bot2"
NS = f"/task_generator_node/{ROBOT}"
INPUT_TOPIC = f"{NS}/cmd_vel_pre_hold"
OUTPUT_TOPIC = f"{NS}/cmd_vel"
HOLD_TOPIC = "/social_yielding/motion_hold"
RELEASE_SERVICE = "/social_yielding/release_motion_hold"


class MotionHoldGate(Node):
    """Replace every final velocity command with zero while the initial hold is active."""

    def __init__(self):
        super().__init__("social_yielding_motion_hold_gate")
        state_qos = QoSProfile(depth=1)
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._enabled = False
        self._hold = False
        self._episode = None
        self._pub = self.create_publisher(Twist, OUTPUT_TOPIC, 10)
        self._hold_pub = self.create_publisher(Bool, HOLD_TOPIC, state_qos)
        self.create_subscription(Twist, INPUT_TOPIC, self._cb_cmd, 10)
        self.create_subscription(
            Bool, "/social_yielding/enabled", self._cb_enabled, state_qos)
        self.create_subscription(
            Int16, "/task_generator_node/task_reset", self._cb_reset, state_qos)
        self.create_service(Trigger, RELEASE_SERVICE, self._release)
        self._publish_hold()
        self.get_logger().info(
            f"MotionHoldGate ready | {INPUT_TOPIC} -> {OUTPUT_TOPIC}")

    def _publish_hold(self):
        self._hold_pub.publish(Bool(data=self._hold))

    def _cb_cmd(self, msg: Twist):
        self._pub.publish(Twist() if self._hold else msg)

    def _cb_enabled(self, msg: Bool):
        was_enabled = self._enabled
        self._enabled = bool(msg.data)
        # task_reset is latched and can arrive before the scenario task publishes its
        # enabled flag. Lock the already-current episode when yielding becomes enabled.
        if self._enabled and not was_enabled and self._episode is not None:
            self._hold = True
            self._publish_hold()
            self._pub.publish(Twist())
            self.get_logger().info(
                f"[Hold] episode #{self._episode}: enabled after reset; motion frozen")
            return
        if not self._enabled and self._hold:
            self._hold = False
            self._publish_hold()
            self.get_logger().info("social_yielding disabled; motion hold released")

    def _cb_reset(self, msg: Int16):
        self._episode = int(msg.data)
        self._hold = self._enabled
        self._publish_hold()
        if self._hold:
            self._pub.publish(Twist())
            self.get_logger().info(
                f"[Hold] episode #{self._episode}: robot and pedestrians frozen while planning")

    def _release(self, _request, response):
        self._hold = False
        self._publish_hold()
        response.success = True
        response.message = f"motion hold released for episode #{self._episode}"
        self.get_logger().info(f"[Hold] {response.message}")
        return response


def main():
    rclpy.init()
    node = MotionHoldGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
