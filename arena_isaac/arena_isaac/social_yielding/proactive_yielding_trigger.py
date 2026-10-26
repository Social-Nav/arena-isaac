"""
Proactive Yielding Trigger

Fires ONE yield per episode after the scene-ready goal gate opens:

  1. Isaac initializes character animation graphs and publishes the current episode id on
     /isaac/scene_ready. RobotManager validates the local costmap and sensor TFs, then
     publishes the final episode id on /scene_ready and both goals are released.
  2. Wait until Nav2 accepts the original robot goal and publishes a fresh global plan.
  3. Pause Isaac -> request a snapshot -> the orchestrator's LLM picks a yield goal Y
     and drives there.
  4. Done for this episode. The orchestrator owns the rest of the flow (arrival at Y,
     the path-back-to-A test, re-yields, and the final resume of the original goal G).
     A new task_reset re-arms us for the next episode.

WHY READINESS, NOT BLOCKAGE
    The trigger used to watch for a pedestrian-caused blockage (no feasible path /
    forced detour / head-on conflict). Those rules are in _blockage_rules_backup.py
    (gitignored, not imported) if you want them back. The problem: "the global planner
    cannot find a path" is exactly what a lagging TF or a not-yet-current costmap also
    looks like, so the trigger fired on infrastructure hiccups with no pedestrian
    involved -- and each false fire cost a pause + snapshot + LLM call.

WHY task_reset ALONE IS NOT "READY"
    task_reset is published while Isaac is still PAUSED -- node.py publishes it inside
    reset_task(), before after_reset_task() unpauses -- and before the goal is sent:
    robot_manager.reset() only creates the _publish_goal_loop task, which then waits on
    a sim tick, fresh odom TF, and pose sync. So task_reset leads "actually running" by
    seconds. It is used here only as the EPISODE ID (re-arm signal), never as ready.

THE READINESS CONJUNCTION (all must hold simultaneously)
    sim_running     Isaac unpaused, i.e. after_reset_task() completed.
    scene_ready     /scene_ready equals this task_reset episode id. This proves character
                    graphs, the fresh local costmap and local sensor TF origins are ready.
    goal            a goal_pose arrived for THIS episode. Implicitly strong: the
                    publisher already waited for sim tick + fresh odom TF + pose sync.
    nav accepted    navigate_to_pose status is ACCEPTED/EXECUTING -> bt_navigator and
                    both costmaps are up and chewing on our goal.
    global plan     a non-empty planner /plan arrived AFTER the goal. This is the
                    load-bearing one: it is the proof the planner really produced a path.
                    Resets that fail (TF extrapolation, "costmap timed out") never get
                    here, so they simply never trigger instead of burning an LLM call.
    robot TF        map->base_link resolvable.
Usage:
    cd /opt/arena_ws && source src/Arena/_meta/tools/source
    ros2 run arena_isaac proactive_yielding_trigger
"""

import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import tf2_ros
from tf2_ros import TransformException

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from action_msgs.msg import GoalStatusArray, GoalStatus
from std_msgs.msg import Bool, Int16
from std_srvs.srv import Trigger

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# From the `robot` launch arg via ARENA_ROBOT. Must match it: a wrong value subscribes
# everything below to a namespace nobody publishes to, and the trigger just waits forever.
ROBOT = os.environ.get("ARENA_ROBOT", "").strip() or "Ai2_Bot2"
NS = f"/task_generator_node/{ROBOT}"
BASE_FRAME = f"{ROBOT}/base_link"

# Scene settling happens before either goal is released. Once the fresh global plan arrives,
# pause on the next short poll instead of adding another window during which the robot can move.
CHECK_PERIOD_SEC = 0.05
# Match NavFn's configured goal tolerance, with one global-costmap cell of slack.
PLAN_GOAL_TOLERANCE = 1.1

# navigate_to_pose statuses that mean "nav2 has our goal and is working on it".
_NAV_ACTIVE_STATUSES = frozenset({
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
})


class ProactiveYieldingTrigger(Node):
    def __init__(self):
        super().__init__("proactive_yielding_trigger")

        _state_qos = QoSProfile(depth=1)
        _state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._lock = threading.Lock()

        # ── readiness inputs ──────────────────────────────────────────────────
        self._goal_msg: PoseStamped | None = None
        self._plan_ok = False       # fresh full planner path seen after the goal
        self._nav_active = False    # navigate_to_pose ACCEPTED/EXECUTING
        self._sim_running = False   # Isaac unpaused
        self._scene_ready_episode: int | None = None

        self.create_subscription(
            PoseStamped, f"{NS}/goal_pose", self._cb_goal, 10)
        self.create_subscription(
            NavPath, f"{NS}/plan", self._cb_plan, 10)
        self.create_subscription(
            GoalStatusArray, f"{NS}/navigate_to_pose/_action/status",
            self._cb_nav_status, 10)
        self.create_subscription(
            Bool, "isaac/sim_running", self._cb_sim_running, _state_qos)
        self.create_subscription(
            Int16, "/scene_ready", self._cb_scene_ready, _state_qos)

        # ── episode state ─────────────────────────────────────────────────────
        # _episode is the last task_reset counter seen; _fired_episode is the one we
        # already yielded for. One yield per episode, so a scenario that needs several
        # resets to start cleanly only ever triggers on the run that actually got going.
        self._episode: int | None = None
        self._fired_episode: int | None = None
        self._triggering = False

        self.create_subscription(
            Int16, "/task_generator_node/task_reset", self._cb_task_reset, _state_qos)

        # The orchestrator owns the episode once a yield starts (drive to Y, test, re-yield,
        # resume G). Latched so we learn the current state on startup.
        self._yield_in_progress = False
        self.create_subscription(
            Bool, "social_yielding/active",
            lambda m: setattr(self, "_yield_in_progress", bool(m.data)), _state_qos)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._srv_cbg = ReentrantCallbackGroup()
        self._pause_sim = self.create_client(
            Trigger, "isaac/PauseSimulation", callback_group=self._srv_cbg)
        self._capture_snapshot = self.create_client(
            Trigger, "isaac/CaptureSnapshot", callback_group=self._srv_cbg)

        # Scenario-level enable (latched, published by task_generator at reset). The node always
        # launches but stays fully INERT until enabled: the check timer is created only while
        # enabled and cancelled when disabled, so there is zero polling when off.
        self._enabled = False
        self._check_timer = None
        self.create_subscription(
            Bool, "/social_yielding/enabled", self._cb_enabled, _state_qos)

        self.get_logger().info(
            f"ProactiveYieldingTrigger ready | robot={ROBOT} ns={NS} | mode=on-reset "
            f"| poll={CHECK_PERIOD_SEC}s | requires episode-scoped /scene_ready")

    # ── readiness callbacks ───────────────────────────────────────────────────

    def _cb_goal(self, msg: PoseStamped):
        with self._lock:
            self._goal_msg = msg
            # A new goal invalidates the previous plan: the next plan must arrive AFTER
            # this goal, so the last episode's path is never accepted as evidence.
            self._plan_ok = False
            # task_reset is now latched. Never invent an episode id from a goal;
            # the matching episode and scene-ready state may arrive on later callbacks.
        self.get_logger().info(
            f"[Ready] goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def _cb_plan(self, msg: NavPath):
        if not msg.poses:
            return
        with self._lock:
            if self._goal_msg is None:
                return  # a plan for the previous episode's goal; ignore
            goal_stamp = self._goal_msg.header.stamp
            plan_stamp = msg.header.stamp
            goal_ns = goal_stamp.sec * 1_000_000_000 + goal_stamp.nanosec
            plan_ns = plan_stamp.sec * 1_000_000_000 + plan_stamp.nanosec
            # Topic callbacks can be delivered out of order. When both messages are stamped,
            # reject a cached plan that predates this episode's released original goal.
            if plan_ns < goal_ns:
                return
            end = msg.poses[-1].pose.position
            goal = self._goal_msg.pose.position
            # Planner paths and goal_pose are normally in map. Never compare coordinates
            # across frames (the MPC's received_global_plan is clipped and in odom).
            if not msg.header.frame_id or msg.header.frame_id != self._goal_msg.header.frame_id:
                self.get_logger().warn(
                    f"[Ready] ignoring plan in {msg.header.frame_id!r}; "
                    f"goal frame is {self._goal_msg.header.frame_id!r}",
                    throttle_duration_sec=5.0)
                return
            if ((end.x - goal.x) ** 2 + (end.y - goal.y) ** 2) ** 0.5 > PLAN_GOAL_TOLERANCE:
                return
            first = not self._plan_ok
            self._plan_ok = True
        if first:  # log the transition only -- the planner republishes at its own rate
            self.get_logger().info(f"[Ready] global plan: {len(msg.poses)} poses")

    def _cb_nav_status(self, msg: GoalStatusArray):
        last = next(reversed(list(msg.status_list)), None)
        self._nav_active = (
            last is not None and last.status in _NAV_ACTIVE_STATUSES)

    def _cb_sim_running(self, msg: Bool):
        self._sim_running = bool(msg.data)

    def _cb_scene_ready(self, msg: Int16):
        episode = int(msg.data)
        self._scene_ready_episode = episode if episode >= 0 else None

    def _cb_task_reset(self, msg: Int16):
        """Adopt a new episode id and clear stale goal/plan state."""
        with self._lock:
            self._goal_msg = None
            self._plan_ok = False
        self._nav_active = False
        self._episode = int(msg.data)
        self._scene_ready_episode = None
        self._triggering = False
        self.get_logger().debug(
            f"[Reset] task_reset #{msg.data} -> stale goal/plan/scene state cleared")

    # ── robot pose ────────────────────────────────────────────────────────────

    def _robot_xy(self) -> tuple[float, float] | None:
        """Robot (x, y) in the map frame, or None if TF cannot be resolved."""
        try:
            t = self._tf_buffer.lookup_transform("map", BASE_FRAME, rclpy.time.Time())
            return (t.transform.translation.x, t.transform.translation.y)
        except TransformException:
            return None

    # ── readiness evaluation ──────────────────────────────────────────────────

    def _missing_signals(self, xy) -> list[str]:
        """Readiness signals that are NOT yet satisfied. Empty list == environment ready.

        `xy` is the result of _robot_xy() (None => TF unresolvable).
        """
        with self._lock:
            have_goal = self._goal_msg is not None
            have_plan = self._plan_ok
        missing = []
        if not self._sim_running:
            missing.append("sim-paused")
        if self._episode is None or self._scene_ready_episode != self._episode:
            missing.append("scene-not-ready")
        if not have_goal:
            missing.append("no-goal")
        if not self._nav_active:
            missing.append("nav-idle")
        if not have_plan:
            missing.append("no-global-plan")
        if xy is None:
            missing.append("no-robot-TF")
        return missing

    def _cb_enabled(self, msg: Bool):
        """Enable/disable the whole pipeline (latched, from the scenario's social_yielding field)."""
        enabled = bool(msg.data)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if enabled and self._check_timer is None:
            self._check_timer = self.create_timer(CHECK_PERIOD_SEC, self._check_ready)
            self.get_logger().info("social_yielding ENABLED (readiness check running)")
        elif not enabled and self._check_timer is not None:
            self._check_timer.cancel()
            self.destroy_timer(self._check_timer)
            self._check_timer = None
            self.get_logger().info("social_yielding DISABLED (readiness check stopped)")

    def _check_ready(self):
        if self._triggering:
            return

        # The orchestrator is mid-episode (driving to Y / testing / re-yielding). It owns
        # the flow until it resumes G, and it re-yields on its own when needed.
        if self._yield_in_progress:
            return

        # One yield per episode. _episode is None until the first task_reset, which is
        # also the point at which the readiness inputs become meaningful.
        if self._episode is None:
            self.get_logger().info(
                "[Ready] waiting for the first task_reset", throttle_duration_sec=10.0)
            return
        if self._fired_episode == self._episode:
            return

        xy = self._robot_xy()
        missing = self._missing_signals(xy)
        if missing:
            self.get_logger().info(
                f"[Ready] episode #{self._episode} waiting on: {', '.join(missing)}",
                throttle_duration_sec=5.0)
            return

        self.get_logger().warn(
            f"YIELD TRIGGERED | episode #{self._episode} scene ready + fresh global plan "
            "| -> proactive replan")
        self._fired_episode = self._episode
        self._triggering = True
        threading.Thread(target=self._trigger_thread, daemon=True).start()

    def _trigger_thread(self):
        try:
            self._on_ready()
        finally:
            self._triggering = False

    # ── pause + snapshot (hands off to the orchestrator) ──────────────────────

    def _on_ready(self):
        self.get_logger().error(
            "========== YIELD TRIGGERED ==========\n"
            f"  Episode: #{self._episode}\n"
            "  Trigger: environment ready after reset (proactive, pre-departure)\n"
            "  Pausing simulation, then capturing snapshot.\n"
            "===================================="
        )

        # Strict order: the pause must take effect BEFORE the snapshot is requested, so the
        # images are of a frozen scene. Blocking here is fine -- daemon thread, not the executor.
        self.get_logger().info("[Yield] step 1/3: requesting isaac/PauseSimulation ...")
        if not self._call_trigger_sync(self._pause_sim, "pause", timeout=2.0):
            self.get_logger().warn(
                "[Yield] step 1/3 FAILED: pause not confirmed; aborting snapshot")
            # Let this episode try again rather than losing its only yield.
            self._fired_episode = None
            return
        self.get_logger().info("[Yield] step 2/3: pause confirmed by Isaac Sim")

        # The isaac side only QUEUES the capture and returns; the actual capture happens in
        # its paused main loop, then it publishes isaac/snapshot_ready -> orchestrator.
        self.get_logger().info("[Yield] step 3/3: requesting isaac/CaptureSnapshot ...")
        if self._call_trigger_sync(self._capture_snapshot, "snapshot", timeout=2.0):
            self.get_logger().info(
                "[Yield] step 3/3: snapshot request accepted (queued on Isaac side; "
                "see isaac log for [Snapshot] capture progress)")
        else:
            self.get_logger().warn(
                "[Yield] step 3/3 FAILED: CaptureSnapshot service unavailable")
            self._fired_episode = None

    def _call_trigger_sync(self, client, label: str, timeout: float = 2.0) -> bool:
        """Call a std_srvs/Trigger client and wait for the response.

        Poll-until-done because the MultiThreadedExecutor completes the future on another
        thread. Returns True only if the service replied success.
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
