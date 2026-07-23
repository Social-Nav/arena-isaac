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

import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from nav2_msgs.msg import Costmap, BehaviorTreeLog
from std_srvs.srv import Trigger

# Reprojection lives next to this file (same social_yielding subpackage).
from arena_isaac.social_yielding.pixel_to_map import pixel_to_map
from arena_isaac.social_yielding.social_yielding_selector import select_yielding_goal


ROBOT = "Ai2_Bot2"
NS = f"/task_generator_node/{ROBOT}"
GOAL_TOPIC = f"{NS}/goal_pose"
COSTMAP_TOPIC = f"{NS}/global_costmap/costmap_raw"   # nav2_msgs/Costmap, raw 0-255
SNAPSHOT_READY_TOPIC = "isaac/snapshot_ready"
UNPAUSE_SRV = "isaac/UnpauseSimulation"

# Post-unpause motion probe cadence (seconds between printed lines). Kept coarse so the
# continuous stream doesn't flood the console.
PROBE_PERIOD = 2.0

# Goal validation: reject a candidate goal whose global-costmap cell cost is >= this.
# nav2 raw costs: 254=LETHAL, 253=INSCRIBED (inside robot radius of an obstacle),
# 1..252=inflation gradient, 0=free, 255=unknown. 253 rejects goals the planner
# can't reach anyway (on/inside an obstacle, e.g. selected "on a wall").
GOAL_COST_THRESHOLD = 253
MAX_GOAL_ATTEMPTS = 6   # selector regenerations before giving up


class SocialReplanOrchestrator(Node):
    def __init__(self):
        super().__init__("social_yielding_orchestrator")

        self._busy = False  # guard against overlapping replans

        self.create_subscription(
            String, SNAPSHOT_READY_TOPIC, self._cb_snapshot_ready, 10)
        self._goal_pub = self.create_publisher(PoseStamped, GOAL_TOPIC, 10)
        self._unpause = self.create_client(Trigger, UNPAUSE_SRV)

        # --- post-unpause motion probe caches (diagnose "robot only spins, ignores new goal") ---
        self._cmd_nav = None   # controller_server output (cmd_vel_nav): the DECISION
        self._cmd_out = None   # final cmd_vel (after smoother/collision): what the base gets
        self._odom = None      # measured body twist
        self._plan = None      # received_global_plan (does it point at the new goal?)
        self.create_subscription(Twist, f"{NS}/cmd_vel_nav", lambda m: setattr(self, "_cmd_nav", m), 10)
        self.create_subscription(Twist, f"{NS}/cmd_vel", lambda m: setattr(self, "_cmd_out", m), 10)
        self.create_subscription(Odometry, f"{NS}/odom", lambda m: setattr(self, "_odom", m), 10)
        self.create_subscription(NavPath, f"{NS}/received_global_plan", lambda m: setattr(self, "_plan", m), 10)
        # global costmap (raw 0-255) to validate a candidate goal isn't on an obstacle/wall
        self._costmap = None
        self.create_subscription(Costmap, COSTMAP_TOPIC, lambda m: setattr(self, "_costmap", m), 10)
        # behavior_tree_log: names the currently-RUNNING BT node -> tells us if the robot is
        # in FollowPath (normal) or a recovery (backup/spin/wait). This is the definitive
        # "who is controlling the robot" signal.
        self._bt_running = "?"     # last node that entered RUNNING
        self._bt_recent = ""       # short recent trail of node status changes
        self._probe_epoch = 0      # bumped on each replan so an old continuous probe stops
        self.create_subscription(
            BehaviorTreeLog, f"{NS}/behavior_tree_log", self._cb_bt_log, 10)
        # sim_running (latched Bool from Isaac): the probe must STOP while the sim is
        # paused/frozen (else it keeps printing stale cached values on wall-clock sleep).
        self._sim_running = True
        _state_qos = QoSProfile(depth=1)
        _state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Bool, "isaac/sim_running", lambda m: setattr(self, "_sim_running", bool(m.data)), _state_qos)

        self.get_logger().info(
            f"SocialReplanOrchestrator ready | listening {SNAPSHOT_READY_TOPIC} "
            f"| goal -> {GOAL_TOPIC}")

    def _cb_snapshot_ready(self, msg: String):
        snapshot_dir = msg.data.strip()
        if self._busy:
            self.get_logger().warn(f"[Replan] busy, ignoring: {snapshot_dir}")
            return
        self._busy = True
        self._probe_epoch += 1   # supersede any still-running continuous probe from a prior replan
        # Do the (blocking: HTTP + numpy) work off the executor thread.
        import threading
        threading.Thread(target=self._run_replan, args=(snapshot_dir,),
                         daemon=True).start()

    # BT nodes that mean "a recovery is driving" vs "normal path following".
    _RECOVERY_NODES = ("BackUp", "Spin", "Wait", "DriveOnHeading", "AssistedTeleop",
                       "ClearingActions", "RecoveryFallback", "RecoveryActions")
    _FOLLOW_NODES = ("FollowPath",)

    def _cb_bt_log(self, msg: BehaviorTreeLog):
        # Track the most recent node that transitioned to RUNNING, and keep a short trail.
        for ev in msg.event_log:
            status = ev.current_status.upper()
            if status == "RUNNING":
                self._bt_running = ev.node_name
            # keep a compact trail of the interesting (non-IDLE) transitions
            if ev.node_name in self._RECOVERY_NODES or ev.node_name in self._FOLLOW_NODES:
                self._bt_recent = f"{ev.node_name}:{status}"

    def _control_source(self) -> str:
        """Human-readable label of what's currently driving the robot, from the BT log."""
        n = self._bt_running
        if n in self._FOLLOW_NODES:
            return f"FollowPath (normal path-following)"
        if n in self._RECOVERY_NODES:
            return f"RECOVERY:{n} (NOT following path!)"
        return f"BT:{n}"

    def _goal_cost(self, mx: float, my: float):
        """Global-costmap raw cost (0-255) at map point (mx,my), or None if the
        costmap isn't available / the point is outside it. Used to reject goals
        the selector placed on a wall/obstacle."""
        cm = self._costmap
        if cm is None:
            return None
        res = cm.metadata.resolution
        ox = cm.metadata.origin.position.x
        oy = cm.metadata.origin.position.y
        sx = cm.metadata.size_x
        sy = cm.metadata.size_y
        col = int((mx - ox) / res)
        row = int((my - oy) / res)
        if not (0 <= col < sx and 0 <= row < sy):
            return None  # outside costmap bounds
        return int(cm.data[row * sx + col])

    def _run_replan(self, snapshot_dir: str):
        try:
            self.get_logger().info(f"[Replan] step 1/4: snapshot={snapshot_dir}")
            d = Path(snapshot_dir)
            if not d.is_dir():
                self.get_logger().error(f"[Replan] snapshot dir not found: {d}")
                return

            # 1-2. selector -> pixel -> map, with retry: regenerate the goal if it
            #      reprojects onto an obstacle/wall (high global-costmap cost).
            #      Each rejected pick is fed back to the selector so it avoids it.
            mx = my = None
            rejected: list = []
            for attempt in range(1, MAX_GOAL_ATTEMPTS + 1):
                self.get_logger().info(
                    f"[Replan] step 2/4: yielding selector (GPT) attempt {attempt}/{MAX_GOAL_ATTEMPTS}"
                    f"{f' (avoiding {len(rejected)} rejected)' if rejected else ''}...")
                sel = select_yielding_goal(str(d), rejected=rejected)
                if not sel or "camera" not in sel or "pixel_goal" not in sel:
                    self.get_logger().warn(f"[Replan]   attempt {attempt}: selector returned no valid goal")
                    continue
                cam = sel["camera"]
                u, v = sel["pixel_goal"]
                self.get_logger().info(
                    f"[Replan]   selector -> camera={cam}, pixel=({u},{v}), "
                    f"reason={sel.get('reason', '')}")

                depth_npy = d / f"{cam}_depth.npy"
                camera_json = d / f"{cam}_camera.json"
                if not depth_npy.is_file() or not camera_json.is_file():
                    self.get_logger().error(
                        f"[Replan] missing reprojection inputs: {depth_npy.name} / "
                        f"{camera_json.name} (need updated snapshot_capturer output)")
                    return  # structural problem, retrying won't help
                self.get_logger().info("[Replan] step 3/4: reprojecting pixel -> map...")
                rp = pixel_to_map(u, v, str(depth_npy), str(camera_json))
                if not rp.get("ok"):
                    self.get_logger().warn(f"[Replan]   attempt {attempt}: reprojection failed: {rp.get('reason')}")
                    rejected.append({"camera": cam, "pixel": [u, v],
                                     "reason": f"bad depth/reprojection ({rp.get('reason')})"})
                    continue
                cx, cy, cz = rp["map_point"]

                # validate against the global costmap: reject goals on a wall/obstacle
                cost = self._goal_cost(cx, cy)
                if cost is None:
                    self.get_logger().warn(
                        f"[Replan]   attempt {attempt}: goal ({cx:.2f},{cy:.2f}) has no costmap "
                        f"reading (costmap missing or out of bounds); accepting cautiously")
                elif cost >= GOAL_COST_THRESHOLD:
                    self.get_logger().warn(
                        f"[Replan]   attempt {attempt}: REJECTED goal ({cx:.2f},{cy:.2f}) "
                        f"cost={cost} >= {GOAL_COST_THRESHOLD} (on obstacle/wall) -> regenerating")
                    rejected.append({"camera": cam, "pixel": [u, v],
                                     "reason": "reprojected onto a wall/obstacle"})
                    continue
                else:
                    self.get_logger().info(f"[Replan]   goal cost={cost} (free) -> accepted")

                mx, my = cx, cy
                self.get_logger().warn(
                    f"[Replan]   map goal = ({mx:.2f}, {my:.2f}) [depth={rp.get('depth'):.2f}m, cost={cost}]")
                break

            if mx is None:
                self.get_logger().error(
                    f"[Replan] no valid (obstacle-free) goal after {MAX_GOAL_ATTEMPTS} attempts; "
                    f"leaving sim paused")
                return

            # 3. build the goal (map frame) but DON'T publish yet.
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.pose.position.x = float(mx)
            goal.pose.position.y = float(my)
            goal.pose.position.z = 0.0
            goal.pose.orientation.w = 1.0  # yaw left to the planner

            # 4. UNPAUSE, then IMMEDIATELY publish the new goal — do NOT wait.
            #    Timing subtlety learned from logs:
            #    - If we publish while frozen, bt_navigator plans on a stale costmap -> fails.
            #    - If we unpause and WAIT before publishing, nav2 resumes chasing the OLD goal
            #      (the corridor-end goal), which is currently unreachable -> BT enters BackUp,
            #      an uninterruptible recovery that then blocks our new goal for ~35s.
            #    So: unpause, then publish the new goal AS FAST AS POSSIBLE so it preempts the
            #    old NavigateToPose action before a recovery can latch. bt_navigator's
            #    onGoalPoseReceived preempts whatever is running with the new goal.
            if not self._call_unpause(timeout=2.0):
                self.get_logger().error("[Replan] unpause failed; goal NOT sent, sim still paused")
                return
            goal.header.stamp = self.get_clock().now().to_msg()
            self._goal_pub.publish(goal)
            self.get_logger().warn(
                "[Replan] step 4/4: UNPAUSED + yielding goal published immediately (preempts old goal)")

            # 5. probe the control chain so we can see WHETHER it enters FollowPath.
            self._probe_motion(goal)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"[Replan] crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._busy = False

    def _probe_motion(self, goal: PoseStamped):
        """After unpause, stream the control chain for a few seconds so it's obvious
        WHO is driving the robot (path-follower vs a BT recovery) and why.

        Read the lines like this:
          CONTROL = which BT node is RUNNING (from behavior_tree_log):
            "FollowPath"        -> normal path-following (controller_server drives).
            "RECOVERY:BackUp/Spin/Wait/..." -> a recovery is driving; the robot is NOT
              following the path (this is why cmd_nav is empty and you see slow backup).
          cmd_nav = controller_server output (the DECISION).
            wz!=0, vx~0 (sustained)  -> controller is commanding IN-PLACE ROTATION
              (RotationShim aligning to a goal that's behind the robot). If wz keeps
              flipping sign -> dithering (never converges) = the spin you see.
            vx>0                     -> controller IS driving forward; problem is downstream.
            0,0                      -> controller produced nothing (MPC opt fail / stale TF).
          cmd_out = final cmd_vel. If cmd_nav has vx>0 but cmd_out is 0 -> smoother/
            collision_monitor is zeroing it.
          odom    = what actually executed.
          goalΔ   = heading error (deg) from robot to the new goal; |Δ|>90 explains an
            initial in-place rotate. dist = range to goal.
          plan→goal = does the global plan's endpoint match the new goal? (yes = nav
            accepted it; you said you can see the new path, so expect ~0).
        """
        gx, gy = goal.pose.position.x, goal.pose.position.y
        self.get_logger().warn(
            f"[Probe] streaming control chain CONTINUOUSLY "
            f"(new goal=({gx:+.2f},{gy:+.2f})) — until next replan. Ctrl-C to stop.")
        # Continuous stream: run until (a) the sim pauses/freezes again (next yield cycle
        # begins), or (b) a new snapshot/replan supersedes this probe. Stopping on pause is
        # essential: otherwise this wall-clock loop keeps printing stale cached values while
        # the sim is frozen, AND it blocks _run_replan from returning (so _busy never clears
        # and the next replan is dropped -> infinite stale print).
        self._probe_epoch += 1
        my_epoch = self._probe_epoch
        while my_epoch == self._probe_epoch and self._sim_running:
            time.sleep(PROBE_PERIOD)

            def vw(t):
                return (f"vx={t.linear.x:+.2f} wz={t.angular.z:+.2f}"
                        if t is not None else "  --  ")

            # robot pose from odom (position + yaw) to compute heading error to goal
            goal_str = "goalΔ=  n/a"
            if self._odom is not None:
                p = self._odom.pose.pose.position
                q = self._odom.pose.pose.orientation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                brg = math.atan2(gy - p.y, gx - p.x)
                d = math.degrees((brg - yaw + math.pi) % (2 * math.pi) - math.pi)
                rng = math.hypot(gx - p.x, gy - p.y)
                goal_str = f"goalΔ={d:+6.1f}deg dist={rng:4.2f}m"

            plan_str = "plan→goal=n/a"
            if self._plan is not None and self._plan.poses:
                e = self._plan.poses[-1].pose.position
                plan_str = f"plan_end→goal={math.hypot(gx - e.x, gy - e.y):4.2f}m"

            self.get_logger().info(
                f"[Probe] CONTROL={self._control_source()} | "
                f"cmd_nav[{vw(self._cmd_nav)}] | cmd_out[{vw(self._cmd_out)}] | "
                f"odom[{vw(self._odom.twist.twist if self._odom else None)}] | "
                f"{goal_str} | {plan_str}")
        reason = "sim paused/frozen" if not self._sim_running else "superseded by new replan"
        self.get_logger().warn(f"[Probe] stopped ({reason})")

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
