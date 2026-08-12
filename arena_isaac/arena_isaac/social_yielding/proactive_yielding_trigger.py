"""
Proactive Yielding Trigger

Fires ONE yield per episode, as soon as the environment is confirmed ready:

  1. Wait for every readiness signal below to hold simultaneously (briefly confirmed --
     see the constants: 2 polls / 0.5 s, short-circuited if the robot moves 0.25 m first).
  2. Pause Isaac -> request a snapshot -> the orchestrator's LLM picks a yield goal Y
     and drives there.
  3. Done for this episode. The orchestrator owns the rest of the flow (arrival at Y,
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
    goal            a goal_pose arrived for THIS episode. Implicitly strong: the
                    publisher already waited for sim tick + fresh odom TF + pose sync.
    nav accepted    navigate_to_pose status is ACCEPTED/EXECUTING -> bt_navigator and
                    both costmaps are up and chewing on our goal.
    global plan     a non-empty received_global_plan arrived AFTER the goal. This is the
                    load-bearing one: it is the proof the planner really produced a path.
                    Resets that fail (TF extrapolation, "costmap timed out") never get
                    here, so they simply never trigger instead of burning an LLM call.
    robot TF        map->base_link resolvable.
    people          at least one /people message seen (an EMPTY list counts -- some
                    scenarios have no pedestrians and must still be able to yield).

Usage:
    cd /opt/arena_ws && source src/Arena/_meta/tools/source
    ros2 run arena_isaac proactive_yielding_trigger
"""

import math
import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import tf2_ros
from tf2_ros import TransformException

from people_msgs.msg import People
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

# Readiness confirmation. The point of this trigger is to fire BEFORE the robot commits to
# its route, so the confirmation window is deliberately tight and doubly bounded:
#
#   TIME:     READY_CONFIRM_POLLS consecutive polls, i.e. CHECK_PERIOD_SEC * POLLS seconds.
#             Two polls is enough to reject a single-poll flicker (a plan published off a
#             stale costmap right before nav aborts it) without letting the robot drive off.
#   DISTANCE: READY_MAX_TRAVEL, measured from where the robot was when readiness first held.
#             Once exceeded, fire immediately and skip the rest of the window.
#
# At the DEFAULT constants the distance guard never actually fires: the only poll that can
# consult it is the first one, where the measured travel is 0 by construction. It is a
# backstop that becomes load-bearing if READY_CONFIRM_POLLS is ever raised -- at confirm=4
# and 0.8 m/s it cuts the window short at 0.40 m instead of 0.75 m. Kept deliberately so
# raising the window cannot silently let the robot drive most of its route first.
#
# Why the window cannot be zero, and why it is also nearly free: nav2 publishes
# received_global_plan when the BT's ComputePathToPose returns, roughly one BT tick (~0.1 s)
# before FollowPath issues the first cmd_vel. So readiness turns true just as the robot
# starts moving, and every extra 0.1 s of confirmation is ~0.035 m at the passive cruise
# speed (0.35 m/s) or ~0.08 m at aggressive (0.8 m/s, and accel-limited from 0 anyway).
CHECK_PERIOD_SEC    = 0.25  # s -- readiness poll period
READY_CONFIRM_POLLS = 2     # consecutive ready polls before firing (=> 0.5 s)
READY_MAX_TRAVEL    = 0.25  # m -- short-circuit the window once the robot has moved this far

# navigate_to_pose statuses that mean "nav2 has our goal and is working on it".
_NAV_ACTIVE_STATUSES = frozenset({
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
})


class ProactiveYieldingTrigger(Node):
    def __init__(self):
        super().__init__("proactive_yielding_trigger")

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        _state_qos = QoSProfile(depth=1)
        _state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._lock = threading.Lock()

        # ── readiness inputs ──────────────────────────────────────────────────
        self._people_seen = False   # any /people msg (empty list counts -- see module doc)
        self._goal_msg: PoseStamped | None = None
        self._plan_ok = False       # non-empty received_global_plan seen after the goal
        self._nav_active = False    # navigate_to_pose ACCEPTED/EXECUTING
        self._sim_running = False   # Isaac unpaused

        self.create_subscription(People, "/people", self._cb_people, sensor_qos)
        self.create_subscription(
            PoseStamped, f"{NS}/goal_pose", self._cb_goal, 10)
        self.create_subscription(
            NavPath, f"{NS}/received_global_plan", self._cb_plan, 10)
        self.create_subscription(
            GoalStatusArray, f"{NS}/navigate_to_pose/_action/status",
            self._cb_nav_status, 10)
        self.create_subscription(
            Bool, "isaac/sim_running", self._cb_sim_running, _state_qos)

        # ── episode state ─────────────────────────────────────────────────────
        # _episode is the last task_reset counter seen; _fired_episode is the one we
        # already yielded for. One yield per episode, so a scenario that needs several
        # resets to start cleanly only ever triggers on the run that actually got going.
        self._episode: int | None = None
        self._fired_episode: int | None = None
        # Confirmation state: consecutive ready polls, and where the robot was when
        # readiness first held (for the travel-distance short circuit).
        self._ready_polls = 0
        self._ready_xy: tuple[float, float] | None = None
        self._triggering = False

        self.create_subscription(
            Int16, "/task_generator_node/task_reset", self._cb_task_reset, 10)

        # The orchestrator owns the episode once a yield starts (drive to Y, test, re-yield,
        # resume G). Latched so we learn the current state on startup.
        self._yield_in_progress = False
        self.create_subscription(
            Bool, "social_yielding/active",
            lambda m: setattr(self, "_yield_in_progress", bool(m.data)), _state_qos)

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._srv_cbg = ReentrantCallbackGroup()
        self._pause_sim = self.create_client(
            Trigger, "isaac/PauseSimulation", callback_group=self._srv_cbg)
        self._capture_snapshot = self.create_client(
            Trigger, "isaac/CaptureSnapshot", callback_group=self._srv_cbg)

        # Scenario-level enable (latched, published by task_generator at reset). The node always
        # launches but stays fully INERT until enabled: the check timer is created only while
        # enabled and cancelled when disabled, so there is zero polling when off.
        self._enabled     = False
        self._check_timer = None
        self.create_subscription(
            Bool, "/social_yielding/enabled", self._cb_enabled, _state_qos)

        self.get_logger().info(
            f"ProactiveYieldingTrigger ready | robot={ROBOT} ns={NS} | mode=on-reset "
            f"| poll={CHECK_PERIOD_SEC}s, confirm={READY_CONFIRM_POLLS} polls "
            f"({CHECK_PERIOD_SEC * READY_CONFIRM_POLLS:.2f}s) or {READY_MAX_TRAVEL}m travel")

    # ── readiness callbacks ───────────────────────────────────────────────────

    def _cb_people(self, msg: People):
        # An empty people list still counts as "the pipeline is alive" -- scenarios
        # without pedestrians must be able to yield too.
        self._people_seen = True

    def _cb_goal(self, msg: PoseStamped):
        adopted = False
        with self._lock:
            self._goal_msg = msg
            # A new goal invalidates the previous plan: the next plan must arrive AFTER
            # this goal, so the last episode's path is never accepted as evidence.
            self._plan_ok = False
            # STARTUP RACE. The first task_reset is published during task_generator's
            # setup() on a VOLATILE topic, so if our subscription is not matched yet the
            # message is lost -- and with auto_reset defaulting to False nothing resends
            # it, which would strand us waiting for an episode id forever. A goal_pose is
            # only ever published from the reset path (robot_manager._publish_goal_loop),
            # and always AFTER task_reset, so its arrival proves an episode started. Adopt
            # it as episode 0 rather than idling until the user resets by hand.
            if self._episode is None:
                self._episode = 0
                adopted = True
        if adopted:
            self.get_logger().warn(
                "[Ready] goal arrived before any task_reset (lost the startup race); "
                "adopting it as episode #0")
        self.get_logger().info(
            f"[Ready] goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def _cb_plan(self, msg: NavPath):
        if not msg.poses:
            return
        with self._lock:
            if self._goal_msg is None:
                return  # a plan for the previous episode's goal; ignore
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

    def _cb_task_reset(self, msg: Int16):
        """New episode: re-arm. Clears the per-episode readiness inputs so the NEXT
        episode's goal/plan must arrive fresh -- a stale plan from the episode we just
        abandoned must never count as ready."""
        with self._lock:
            self._episode = int(msg.data)
            self._goal_msg = None
            self._plan_ok = False
        self._nav_active = False
        self._ready_polls = 0
        self._ready_xy = None
        self._triggering = False
        self.get_logger().debug(
            f"[Reset] task_reset #{msg.data} -> re-armed, waiting for environment ready")

    # ── robot pose ────────────────────────────────────────────────────────────

    def _robot_xy(self) -> tuple[float, float] | None:
        """Robot (x, y) in the map frame, or None if TF cannot be resolved.

        Doubles as the robot-TF readiness check and as the source for the travel-distance
        short circuit, so one lookup serves both.
        """
        try:
            t = self._tf_buffer.lookup_transform("map", BASE_FRAME, rclpy.time.Time())
            return (t.transform.translation.x, t.transform.translation.y)
        except TransformException:
            return None

    # ── readiness evaluation ──────────────────────────────────────────────────

    def _missing_signals(self, xy) -> list[str]:
        """Readiness signals that are NOT yet satisfied. Empty list == environment ready.

        `xy` is the result of _robot_xy() (None => TF unresolvable), passed in so the caller
        can reuse the same lookup for the travel-distance check.
        """
        with self._lock:
            have_goal = self._goal_msg is not None
            have_plan = self._plan_ok
        missing = []
        if not self._sim_running:
            missing.append("sim-paused")
        if not have_goal:
            missing.append("no-goal")
        if not self._nav_active:
            missing.append("nav-idle")
        if not have_plan:
            missing.append("no-global-plan")
        if xy is None:
            missing.append("no-robot-TF")
        if not self._people_seen:
            missing.append("no-/people")
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
            self._ready_polls = 0
            self._ready_xy = None
            self.get_logger().info("social_yielding DISABLED (readiness check stopped)")

    def _check_ready(self):
        if self._triggering:
            return

        # The orchestrator is mid-episode (driving to Y / testing / re-yielding). It owns
        # the flow until it resumes G, and it re-yields on its own when needed.
        if self._yield_in_progress:
            self._ready_polls = 0
            self._ready_xy = None
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
            # Not ready (or no longer ready): drop the partial confirmation.
            if self._ready_polls:
                self.get_logger().info(
                    f"[Ready] confirmation reset after {self._ready_polls} poll(s): "
                    f"{', '.join(missing)}")
            self._ready_polls = 0
            self._ready_xy = None
            self.get_logger().info(
                f"[Ready] episode #{self._episode} waiting on: {', '.join(missing)}",
                throttle_duration_sec=5.0)
            return

        self._ready_polls += 1
        if self._ready_xy is None:
            self._ready_xy = xy

        # How far the robot has travelled since readiness first held. nav2 starts driving
        # about one BT tick after the plan appears, so this is normally centimetres.
        moved = math.dist(xy, self._ready_xy) if (xy and self._ready_xy) else 0.0

        if self._ready_polls < READY_CONFIRM_POLLS and moved < READY_MAX_TRAVEL:
            self.get_logger().info(
                f"[Ready] all signals up for episode #{self._episode}; confirming "
                f"{self._ready_polls}/{READY_CONFIRM_POLLS} (moved {moved:.2f}m)")
            return

        why = ("confirmed" if self._ready_polls >= READY_CONFIRM_POLLS
               else f"robot already moved {moved:.2f}m >= {READY_MAX_TRAVEL}m, firing early")
        self.get_logger().warn(
            f"YIELD TRIGGERED | episode #{self._episode} environment ready ({why}) | "
            f"robot moved {moved:.2f}m since ready | -> proactive replan")
        self._fired_episode = self._episode
        self._ready_polls = 0
        self._ready_xy = None
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
