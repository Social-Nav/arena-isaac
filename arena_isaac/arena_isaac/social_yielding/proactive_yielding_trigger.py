"""
Proactive Yielding Trigger

Monitors for pedestrian-caused path blockage using Nav2's global planner:

  1. Check if any pedestrian is within FOV (±60°) and 5m range.
  2. If yes, call ComputePathToPose to test whether a valid global path
     to the current goal exists.
  3. If path planning fails → corridor is blocked by nearby pedestrian(s)
     → fire _on_blocked().

Pedestrians appear as obstacles in both local and global costmaps via the
social_layer (nav2_social_costmap_plugin::SocialLayer), which subscribes to
/people directly, so nearby people affect the global planner.

Usage:
    cd /opt/arena_ws && source src/Arena/_meta/tools/source
    ros2 run arena_isaac proactive_yielding_trigger
"""

import math
import os
import threading
import time
from pathlib import Path

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient

import tf2_ros
from tf2_ros import TransformException

from people_msgs.msg import People
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int16
from nav2_msgs.action import ComputePathToPose
from std_srvs.srv import Trigger

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FOV_ANGLE        = math.pi / 3.0   # ±60°
MAX_PERSON_DIST  = 5.0             # m
CHECK_PERIOD_SEC = 0.5             # s
COOLDOWN_SEC     = 30.0            # s — min interval between LLM calls
BLOCKED_HOLD_SEC = 3.0             # s — blocked state must persist before triggering


def _angular_diff(a: float, b: float) -> float:
    d = b - a
    while d >  math.pi: d -= 2.0 * math.pi
    while d < -math.pi: d += 2.0 * math.pi
    return d


class ProactiveYieldingTrigger(Node):
    def __init__(self):
        super().__init__("proactive_yielding_trigger")

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._people_msg: People | None = None
        self._goal_msg: PoseStamped | None = None
        self._lock = threading.Lock()

        self.create_subscription(People, "/people", self._cb_people, sensor_qos)
        self.create_subscription(
            PoseStamped, "/task_generator_node/Ai2_Bot2/goal_pose",
            self._cb_goal, 10,
        )
        # Unlatch on simulation unpause (isaac publishes latched Bool state).
        _state_qos = QoSProfile(depth=1)
        _state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Bool, "isaac/sim_running", self._cb_sim_running, _state_qos)
        # Reset all state on a new task/episode.
        self.create_subscription(
            Int16, "/task_generator_node/task_reset", self._cb_task_reset, 10)

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._action_cbg = ReentrantCallbackGroup()
        self._compute_path_cli = ActionClient(
            self, ComputePathToPose,
            "/task_generator_node/Ai2_Bot2/compute_path_to_pose",
            callback_group=self._action_cbg,
        )
        self._pause_sim = self.create_client(
            Trigger, "isaac/PauseSimulation",
        )
        self._capture_snapshot = self.create_client(
            Trigger, "isaac/CaptureSnapshot",
        )

        self._last_trigger_time = 0.0
        self._triggering        = False
        self._blocked_since     = None  # monotonic time when blocked state started
        self._blocked_latched   = False  # True after a trigger; cleared on unpause
        self._sim_was_running   = True   # tracks running state to detect unpause edge

        self.create_timer(CHECK_PERIOD_SEC, self._check_conditions)

        startup_goal = self._load_goal_from_scenario()
        if startup_goal is not None:
            with self._lock:
                self._goal_msg = startup_goal
            self.get_logger().info(
                f"Goal from scenario: "
                f"({startup_goal.pose.position.x:.2f}, {startup_goal.pose.position.y:.2f})"
            )

        self.get_logger().info(
            f"ProactiveYieldingTrigger ready | "
            f"fov=±{math.degrees(FOV_ANGLE):.0f}°, "
            f"max_dist={MAX_PERSON_DIST}m, "
            f"cooldown={COOLDOWN_SEC}s"
        )

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _cb_people(self, msg: People):
        with self._lock:
            self._people_msg = msg
        self.get_logger().info(
            f"People received: {len(msg.people)} person(s)", once=True,
        )

    def _cb_goal(self, msg: PoseStamped):
        with self._lock:
            self._goal_msg = msg
        self.get_logger().info(
            f"Goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})",
            once=True,
        )

    def _cb_sim_running(self, msg: Bool):
        """Unlatch when the sim goes paused -> running (unpause edge)."""
        running = bool(msg.data)
        if running and not self._sim_was_running and self._blocked_latched:
            self._blocked_latched = False
            self._blocked_since = None
            self.get_logger().warn("[Unlock] simulation unpaused -> trigger re-armed")
        self._sim_was_running = running

    def _cb_task_reset(self, msg: Int16):
        """New episode/task: reset ALL state so a stale latch never blocks the
        fresh task. Fires even if the previous task never unpaused."""
        self._blocked_latched = False
        self._blocked_since = None
        self._triggering = False
        self._last_trigger_time = 0.0
        self._sim_was_running = True
        self.get_logger().warn(f"[Reset] task_reset #{msg.data} -> trigger state cleared")

    # ── robot pose ─────────────────────────────────────────────────────────────

    def _get_robot_pose(self) -> tuple[float, float, float] | None:
        try:
            t = self._tf_buffer.lookup_transform(
                "map", "Ai2_Bot2/base_link", rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            return x, y, yaw
        except TransformException:
            return None

    # ── scenario goal loader ───────────────────────────────────────────────────

    def _load_goal_from_scenario(self) -> PoseStamped | None:
        try:
            import subprocess
            def _get_param(node_name, param_name):
                env = os.environ.copy()
                env.setdefault('ROS_DOMAIN_ID', '1')
                r = subprocess.run(
                    ['ros2', 'param', 'get', node_name, param_name],
                    capture_output=True, text=True, timeout=5, env=env,
                )
                for line in r.stdout.splitlines():
                    if 'value is:' in line.lower() or 'string value' in line.lower():
                        return line.split(':')[-1].strip().strip("'\"")
                return None

            world_name    = _get_param('/task_generator_node', 'world')
            scenario_name = _get_param('/task_generator_node', 'task.scenario.file')

            if not world_name:
                self.get_logger().warn("Could not read 'world' param")
                return None

            try:
                import ament_index_python.packages as ament_index
                ass_dir = Path(ament_index.get_package_share_path('arena_simulation_setup'))
            except Exception:
                ass_dir = Path(os.environ.get('ASS_DIR', 'arena_simulation_setup'))

            if not scenario_name:
                scenarios_dir = ass_dir / 'worlds' / world_name / 'scenarios'
                candidates = sorted(
                    e.name for e in scenarios_dir.iterdir() if e.is_dir()
                ) if scenarios_dir.is_dir() else []
                scenario_name = candidates[0] if candidates else None
                if not scenario_name:
                    return None

            scenario_path = ass_dir / 'worlds' / world_name / 'scenarios' / scenario_name / 'scenario.yaml'
            if not scenario_path.exists():
                return None

            with open(scenario_path) as f:
                data = yaml.safe_load(f)

            robots = data.get('robots', [])
            if not robots:
                return None
            goal = robots[0].get('goal')
            if not goal or len(goal) < 2:
                return None

            gx, gy = float(goal[0]), float(goal[1])
            gyaw = math.radians(float(goal[2])) if len(goal) > 2 else 0.0

            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = gx
            msg.pose.position.y = gy
            msg.pose.orientation.z = math.sin(gyaw / 2.0)
            msg.pose.orientation.w = math.cos(gyaw / 2.0)
            return msg

        except Exception as e:
            self.get_logger().warn(f"_load_goal_from_scenario failed: {e}")
            return None

    # ── people in FOV check ────────────────────────────────────────────────────

    def _people_in_fov(
        self, people: People, rx: float, ry: float, yaw: float,
    ) -> list:
        """Return list of people within FOV and MAX_PERSON_DIST."""
        result = []
        for p in people.people:
            dx = p.position.x - rx
            dy = p.position.y - ry
            dist = math.hypot(dx, dy)
            if dist > MAX_PERSON_DIST:
                continue
            angle_to = math.atan2(dy, dx)
            if abs(_angular_diff(yaw, angle_to)) >= FOV_ANGLE:
                continue
            result.append(p)
        return result

    # ── path feasibility check ─────────────────────────────────────────────────

    def _can_plan_to_goal(self, goal: PoseStamped, timeout: float = 5.0) -> bool:
        """Ask Nav2 global planner if a path to goal exists."""
        if not self._compute_path_cli.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("ComputePathToPose server unavailable")
            return True  # assume OK

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = goal
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()
        goal_msg.planner_id = ""

        future = self._compute_path_cli.send_goal_async(goal_msg)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().warn("ComputePathToPose: goal accept timed out")
                return False
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("ComputePathToPose: goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            if time.monotonic() > deadline:
                self.get_logger().warn("ComputePathToPose: result timed out")
                return False
            time.sleep(0.05)

        result = result_future.result()
        if result.status == 4:  # SUCCEEDED
            n = len(result.result.path.poses)
            self.get_logger().info(
                f"Path OK: {n} poses", throttle_duration_sec=5.0,
            )
            return True
        else:
            # Diagnostics: goal is fixed & free (confirmed), so a failure here is
            # usually a bad START (planner resolves the robot pose from TF; if TF
            # is momentarily stale/missing the start is invalid). Log the goal and
            # whether WE can resolve the robot pose at this same instant.
            gx = goal.pose.position.x
            gy = goal.pose.position.y
            pose = self._get_robot_pose()
            pose_str = (f"({pose[0]:.2f},{pose[1]:.2f})" if pose else "TF-UNRESOLVABLE")
            self.get_logger().warn(
                f"No path to goal — status={result.status} | "
                f"goal=({gx:.2f},{gy:.2f}) | robot_pose(TF)={pose_str} | "
                f"npose={len(result.result.path.poses)}",
                throttle_duration_sec=2.0,
            )
            return False

    # ── main timer ─────────────────────────────────────────────────────────────

    def _check_conditions(self):
        if self._triggering:
            return

        # Once a PATH BLOCKED has fired, the sim is paused and stays paused; the
        # scene is frozen so re-checking would just re-fail. Stay latched and skip
        # the whole check — including the ComputePathToPose poll — so we no longer
        # spam our "No path to goal" nor Nav2's "Failed to create plan" warnings.
        # (Unlatch on unpause / new LLM goal will be added later.)
        if self._blocked_latched:
            self.get_logger().info(
                "[Check] skipped: already latched (a PATH BLOCKED fired earlier)",
                throttle_duration_sec=5.0)
            return

        with self._lock:
            people = self._people_msg
            goal   = self._goal_msg

        if goal is None or people is None:
            self.get_logger().warn(
                f"[Check] waiting for inputs: goal={'ok' if goal else 'MISSING'}, "
                f"people={'ok' if people else 'MISSING'}",
                throttle_duration_sec=5.0)
            return

        pose = self._get_robot_pose()
        if pose is None:
            self.get_logger().warn(
                "[Check] robot pose unavailable (TF map->Ai2_Bot2/base_link failed)",
                throttle_duration_sec=5.0)
            return

        rx, ry, yaw = pose

        # ── Two core conditions — printed EVERY cycle before the hold timer ──
        # Condition 1: any pedestrian in FOV (±FOV_ANGLE, within MAX_PERSON_DIST)?
        nearby = self._people_in_fov(people, rx, ry, yaw)
        cond1 = len(nearby) > 0

        # Condition 2: no feasible global path. The path check is an expensive
        # Nav2 action, so only run it when cond1 holds (avoids spamming the
        # planner when nobody is nearby).
        if cond1:
            path_ok = self._can_plan_to_goal(goal)
            cond2 = not path_ok
        else:
            cond2 = False

        if not (cond1 and cond2):
            self._blocked_since = None
            return

        # Both conditions met — start or continue blocked timer
        now = time.monotonic()

        if self._blocked_since is None:
            self._blocked_since = now
            self.get_logger().info("Blocked state detected, waiting for confirmation...")
            return

        elapsed = now - self._blocked_since
        if elapsed < BLOCKED_HOLD_SEC:
            self.get_logger().info(
                f"Blocked for {elapsed:.1f}s / {BLOCKED_HOLD_SEC:.1f}s",
                throttle_duration_sec=1.0,
            )
            return

        names = ", ".join(p.name for p in nearby)
        self.get_logger().warn(
            f"BLOCKED | nearby=[{names}] | no feasible global path for {elapsed:.1f}s"
        )
        self._blocked_since = None
        self._last_trigger_time = now
        self._blocked_latched = True
        self._triggering = True
        threading.Thread(target=self._trigger_thread, daemon=True).start()

    def _trigger_thread(self):
        try:
            self._on_blocked()
        finally:
            self._triggering = False

    # ── LLM hook (stub) ───────────────────────────────────────────────────────

    def _on_blocked(self):
        self.get_logger().error(
            "========== PATH BLOCKED ==========\n"
            "  No feasible global path to goal.\n"
            "  Nearby pedestrians are blocking the corridor.\n"
            "  Pausing simulation, then capturing snapshot.\n"
            "=================================="
        )

        # Strict order: pause must take effect BEFORE the snapshot is requested,
        # so images are of a frozen scene. Wait for the pause response (this runs
        # in a daemon thread, so blocking here does not stall the executor).
        self.get_logger().info("[Yield] step 1/3: requesting isaac/PauseSimulation ...")
        if not self._call_trigger_sync(self._pause_sim, "pause", timeout=2.0):
            self.get_logger().warn("[Yield] step 1/3 FAILED: pause not confirmed; aborting snapshot")
            return
        self.get_logger().info("[Yield] step 2/3: pause confirmed by Isaac Sim")

        # Now request the snapshot. Method A: the isaac side only queues it and
        # returns immediately; the actual capture happens in its paused main loop.
        self.get_logger().info("[Yield] step 3/3: requesting isaac/CaptureSnapshot ...")
        if self._call_trigger_sync(self._capture_snapshot, "snapshot", timeout=2.0):
            self.get_logger().info(
                "[Yield] step 3/3: snapshot request accepted (queued on Isaac side; "
                "see isaac log for [Snapshot] capture progress)")
        else:
            self.get_logger().warn("[Yield] step 3/3 FAILED: CaptureSnapshot service unavailable")

    def _call_trigger_sync(self, client, label: str, timeout: float = 2.0) -> bool:
        """Call a std_srvs/Trigger client and wait for the response.

        Uses the same poll-until-done pattern as _can_plan_to_goal (the
        MultiThreadedExecutor completes the future on another thread). Returns
        True only if the service replied success.
        """
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=timeout):
                self.get_logger().warn(f"{label} service unavailable")
                return False

        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().warn(f"{label} request timed out")
                return False
            time.sleep(0.02)

        resp = future.result()
        return bool(resp is not None and resp.success)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ProactiveYieldingTrigger()
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
