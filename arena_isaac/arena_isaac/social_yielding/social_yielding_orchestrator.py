"""
Social Replan Orchestrator (arena container, ROS node)

Drives the yielding replan once a snapshot has been captured:

    isaac/snapshot_ready (String=snapshot_dir)          <- from run_isaacsim
        -> select_yielding_goal(dir)  (GPT-5.4 vision)  -> camera + pixel (u,v)
        -> pixel_to_map(u, v, <cam>_depth.npy, <cam>_camera.json) -> map (x,y,z)
        -> publish PoseStamped(frame=map) to goal_pose  -> new yielding goal
        -> call isaac/UnpauseSimulation                 -> robot drives to yield

Scope (this stage): pause -> snapshot -> replan -> yield. Restoring the ORIGINAL
goal after yielding is a later stage.

Runs in the arena container: ROS goal topic + isaac service clients live here,
and the snapshot dir is on the shared /opt/arena_ws volume (readable by both
containers). Reprojection (pixel_to_map) is pure numpy; the selector is HTTP.
"""

import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

# Reprojection lives next to this file (same social_yielding subpackage).
from arena_isaac.social_yielding.pixel_to_map import pixel_to_map
from arena_isaac.social_yielding.social_yielding_selector import select_yielding_goal


ROBOT = "Ai2_Bot2"
GOAL_TOPIC = f"/task_generator_node/{ROBOT}/goal_pose"
SNAPSHOT_READY_TOPIC = "isaac/snapshot_ready"
UNPAUSE_SRV = "isaac/UnpauseSimulation"


class SocialReplanOrchestrator(Node):
    def __init__(self):
        super().__init__("social_yielding_orchestrator")

        self._busy = False  # guard against overlapping replans

        self.create_subscription(
            String, SNAPSHOT_READY_TOPIC, self._cb_snapshot_ready, 10)
        self._goal_pub = self.create_publisher(PoseStamped, GOAL_TOPIC, 10)
        self._unpause = self.create_client(Trigger, UNPAUSE_SRV)

        self.get_logger().info(
            f"SocialReplanOrchestrator ready | listening {SNAPSHOT_READY_TOPIC} "
            f"| goal -> {GOAL_TOPIC}")

    def _cb_snapshot_ready(self, msg: String):
        snapshot_dir = msg.data.strip()
        if self._busy:
            self.get_logger().warn(f"[Replan] busy, ignoring: {snapshot_dir}")
            return
        self._busy = True
        # Do the (blocking: HTTP + numpy) work off the executor thread.
        import threading
        threading.Thread(target=self._run_replan, args=(snapshot_dir,),
                         daemon=True).start()

    def _run_replan(self, snapshot_dir: str):
        try:
            self.get_logger().info(f"[Replan] step 1/4: snapshot={snapshot_dir}")
            d = Path(snapshot_dir)
            if not d.is_dir():
                self.get_logger().error(f"[Replan] snapshot dir not found: {d}")
                return

            # 1. selector -> camera + pixel goal
            self.get_logger().info("[Replan] step 2/4: calling yielding selector (GPT)...")
            sel = select_yielding_goal(str(d))
            if not sel or "camera" not in sel or "pixel_goal" not in sel:
                self.get_logger().error("[Replan] selector returned no valid goal")
                return
            cam = sel["camera"]
            u, v = sel["pixel_goal"]
            self.get_logger().info(
                f"[Replan]   selector -> camera={cam}, pixel=({u},{v}), "
                f"reason={sel.get('reason', '')}")

            # 2. reproject pixel -> map
            depth_npy = d / f"{cam}_depth.npy"
            camera_json = d / f"{cam}_camera.json"
            if not depth_npy.is_file() or not camera_json.is_file():
                self.get_logger().error(
                    f"[Replan] missing reprojection inputs: {depth_npy.name} / "
                    f"{camera_json.name} (need updated snapshot_capturer output)")
                return
            self.get_logger().info("[Replan] step 3/4: reprojecting pixel -> map...")
            rp = pixel_to_map(u, v, str(depth_npy), str(camera_json))
            if not rp.get("ok"):
                self.get_logger().error(f"[Replan] reprojection failed: {rp.get('reason')}")
                return
            mx, my, mz = rp["map_point"]
            self.get_logger().warn(
                f"[Replan]   map goal = ({mx:.2f}, {my:.2f}, {mz:.2f}) "
                f"[depth={rp.get('depth'):.2f}m]")

            # 3. publish new yielding goal (map frame)
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = float(mx)
            goal.pose.position.y = float(my)
            goal.pose.position.z = 0.0
            goal.pose.orientation.w = 1.0  # yaw left to the planner
            self._goal_pub.publish(goal)
            self.get_logger().info("[Replan] step 4/4: yielding goal published")

            # 4. unpause the sim so the robot starts moving
            if self._call_unpause(timeout=2.0):
                self.get_logger().warn("[Replan] simulation UNPAUSED -> robot yielding")
            else:
                self.get_logger().error("[Replan] unpause failed; goal sent but sim still paused")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"[Replan] crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._busy = False

    def _call_unpause(self, timeout: float = 2.0) -> bool:
        if not self._unpause.service_is_ready():
            if not self._unpause.wait_for_service(timeout_sec=timeout):
                self.get_logger().warn("unpause service unavailable")
                return False
        future = self._unpause.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().warn("unpause request timed out")
                return False
            time.sleep(0.02)
        resp = future.result()
        return bool(resp is not None and resp.success)


def main():
    rclpy.init()
    node = SocialReplanOrchestrator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
