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
import os
import time
from pathlib import Path

import yaml

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.action import ActionClient

import tf2_ros
from tf2_ros import TransformException

from std_msgs.msg import String, Bool, Int16
from geometry_msgs.msg import PoseStamped
from nav2_msgs.msg import Costmap, BehaviorTreeLog
from nav2_msgs.action import ComputePathToPose
from std_srvs.srv import Trigger

# Reprojection lives next to this file (same social_yielding subpackage).
from arena_isaac.social_yielding.pixel_to_map import pixel_to_map
from arena_isaac.social_yielding.social_yielding_selector import select_yielding_goal


# From the `robot` launch arg via ARENA_ROBOT; must match the trigger's value.
ROBOT = os.environ.get("ARENA_ROBOT", "").strip() or "Ai2_Bot2"
NS = f"/task_generator_node/{ROBOT}"
GOAL_TOPIC = f"{NS}/goal_pose"
COSTMAP_TOPIC = f"{NS}/global_costmap/costmap_raw"   # nav2_msgs/Costmap, raw 0-255
COMPUTE_PATH_ACTION = f"{NS}/compute_path_to_pose"
SNAPSHOT_READY_TOPIC = "isaac/snapshot_ready"
UNPAUSE_SRV = "isaac/UnpauseSimulation"
PAUSE_SRV = "isaac/PauseSimulation"
CAPTURE_SRV = "isaac/CaptureSnapshot"
BASE_FRAME = f"{ROBOT}/base_link"


# --- Yield-recovery loop (drive to yield goal Y -> can we get back to the pre-yield spot A?
#     yes:行人让开了, resume original goal G. no: re-yield from here.) ---
ARRIVE_TOL = 0.5           # m — robot within this of yield goal Y counts as "arrived" (TF map dist)
ARRIVE_TIMEOUT = 90.0      # s — give up waiting to reach Y (wall clock; probe already bails on pause).
                           # 40->90: robot was moving toward Y (dist shrinking) but slowly, so give
                           # it more time to actually arrive before declaring failure.
MAX_YIELD_ROUNDS = 5       # safety valve: after this many consecutive yields, force-resume G and stop
# Topic (latched) telling the proactive trigger a yield episode is in progress, so it does NOT
# fire another yield mid-episode (incl. while re-yielding). Published by orchestrator.
YIELD_ACTIVE_TOPIC = "social_yielding/active"

# Keep the snapshot dir on disk after the replan has read it, or delete it.
#
# This is deliberately NOT a "don't capture" switch: the replan pipeline reads the
# snapshot back off disk (selector loads the PNGs, pixel_to_map loads
# <cam>_depth.npy + <cam>_camera.json), so a capture that never lands is a yield
# that cannot happen. The only safe thing to make optional is the retention after
# step 4 published the goal, which is when nothing reads the dir any more.
#
# Default false: a long collection run yields many times and each yield leaves three
# PNGs + a depth npy, which is pure growth once the replan has consumed them. Set
# ARENA_SAVE_SNAPSHOTS=true when debugging the yield pipeline -- the snapshot is the
# only record of what the LLM actually saw when it picked a goal.
SAVE_SNAPSHOTS = os.environ.get("ARENA_SAVE_SNAPSHOTS", "false").strip().lower() in (
    "true", "1", "yes", "on")

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

        # NOTE: the cmd_vel_nav / cmd_vel / odom / received_global_plan taps that used to live
        # here were removed. They only ever fed per-sample log lines, and that job now belongs
        # to the controller's [CMD-CHAIN] + [MOTION-DIAG] lines, which see the whole command
        # chain (solved -> cmd_vel_nav -> smoothed -> cmd_vel -> odom) from inside the process
        # that actually solves it. Duplicating it here produced two streams of the same story.
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

        # --- yield-recovery state machine ---
        # A = robot map pose captured at the START of a yield episode (the spot where it got
        #     blocked). Fixed for the whole episode (NOT updated on re-yields). Recovery test
        #     is "can we plan from here back to A?" -> if yes, pedestrians cleared -> resume G.
        self._pos_A = None            # (x, y) in map frame, or None
        self._orig_goal_G = None      # PoseStamped, the scenario's final goal
        self._yield_active = False    # True between first yield and resume/giveup
        self._yield_round = 0         # consecutive yields this episode (safety valve)
        self._pending_reyield = False # set when A still blocked -> capture again after _busy clears

        # TF for robot map pose + resume-path check
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        # re-yield needs to pause + capture a fresh snapshot ourselves
        self._pause = self.create_client(Trigger, PAUSE_SRV)
        self._capture = self.create_client(Trigger, CAPTURE_SRV)
        # global planner query: "can I reach A from here?"
        self._compute_path_cli = ActionClient(self, ComputePathToPose, COMPUTE_PATH_ACTION)
        # latched "yield episode in progress" flag -> proactive trigger suppresses new yields
        _latch_qos = QoSProfile(depth=1)
        _latch_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._active_pub = self.create_publisher(Bool, YIELD_ACTIVE_TOPIC, _latch_qos)
        self._publish_active(False)

        # Scenario-level enable (latched, from task_generator). The trigger already stops firing
        # when disabled, so normally no snapshot arrives; this is a defensive gate so a stray
        # snapshot_ready (e.g. a manual Isaac capture) can't start an LLM yield episode when off.
        self._enabled = False
        self.create_subscription(
            Bool, "/social_yielding/enabled",
            lambda m: setattr(self, "_enabled", bool(m.data)), _state_qos)

        # A task_reset ABANDONS whatever episode we were driving: the robot is teleported
        # back to start and a fresh goal G is published, so spot A, the yield goal Y and the
        # in-flight arrival wait all describe a world that no longer exists. Without this the
        # episode state leaks across resets -- _yield_active stays True, which keeps
        # social_yielding/active latched True, which makes the proactive trigger skip every
        # future episode ("orchestrator owns control") and the pipeline never fires again.
        self.create_subscription(
            Int16, "/task_generator_node/task_reset", self._cb_task_reset, 10)

        self.get_logger().info(
            f"SocialReplanOrchestrator ready | robot={ROBOT} | listening {SNAPSHOT_READY_TOPIC} "
            f"| goal -> {GOAL_TOPIC} | save_snapshots={SAVE_SNAPSHOTS}")

    def _cb_task_reset(self, msg: Int16):
        """Abandon the current yield episode; the new task owns the robot now."""
        was_active = self._yield_active or self._busy
        # Supersede any running probe / arrival wait so _drive_and_wait_arrival returns
        # False promptly instead of chasing the previous episode's Y.
        self._probe_epoch += 1
        self._yield_active = False
        self._busy = False
        self._pos_A = None
        self._orig_goal_G = None
        self._yield_round = 0
        self._pending_reyield = False
        self._publish_active(False)   # re-arm the proactive trigger for the new episode
        if was_active:
            self.get_logger().warn(
                f"[Reset] task_reset #{msg.data} arrived mid-yield -> episode ABANDONED, "
                f"state cleared, trigger re-armed")
        else:
            self.get_logger().debug(f"[Reset] task_reset #{msg.data} -> state cleared")

    def _cb_snapshot_ready(self, msg: String):
        snapshot_dir = msg.data.strip()
        if not self._enabled:
            self.get_logger().info(
                f"[Replan] social_yielding disabled; ignoring snapshot: {snapshot_dir}",
                throttle_duration_sec=5.0)
            return
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

    # ── yield-recovery helpers ─────────────────────────────────────────────────

    def _robot_map_pose(self):
        """Robot (x, y) in the map frame via TF, or None. Used for arrival + A capture."""
        try:
            t = self._tf_buffer.lookup_transform("map", BASE_FRAME, rclpy.time.Time())
            return (t.transform.translation.x, t.transform.translation.y)
        except TransformException:
            return None

    def _load_original_goal_G(self) -> PoseStamped | None:
        """Load the ACTIVE task's final goal G from its scenario.yaml.

        Resolves the scenario the current task is using via live ROS params
        (world + task.scenario.file), then reads the goal. Supports both schemas:
          - robot:  {waypoints: [[x,y,yaw], ...]}  -> G = last waypoint  (current format)
          - robots: [{goal: [x,y,yaw]}]            -> G = robots[0].goal (legacy)
        """
        def _param(node, name):
            import subprocess
            env = os.environ.copy()
            env.setdefault('ROS_DOMAIN_ID', '1')
            try:
                r = subprocess.run(['ros2', 'param', 'get', node, name],
                                   capture_output=True, text=True, timeout=5, env=env)
            except Exception:
                return None
            for line in r.stdout.splitlines():
                if 'value is:' in line.lower() or 'string value' in line.lower():
                    return line.split(':')[-1].strip().strip("'\"")
            return None

        world = _param('/task_generator_node', 'world')
        scenario = _param('/task_generator_node', 'task.scenario.file')
        if not world:
            self.get_logger().warn("[Resume] cannot read 'world' param; no G")
            return None
        try:
            import ament_index_python.packages as ament_index
            ass = Path(ament_index.get_package_share_path('arena_simulation_setup'))
        except Exception:
            ass = Path(os.environ.get('ASS_DIR', 'arena_simulation_setup'))

        if not scenario:
            sdir = ass / 'worlds' / world / 'scenarios'
            cands = sorted(e.name for e in sdir.iterdir() if e.is_dir()) if sdir.is_dir() else []
            scenario = cands[0] if cands else None
            if not scenario:
                return None
        path = ass / 'worlds' / world / 'scenarios' / scenario / 'scenario.yaml'
        if not path.exists():
            self.get_logger().warn(f"[Resume] scenario not found: {path}")
            return None
        try:
            data = yaml.safe_load(open(path))
        except Exception as e:
            self.get_logger().warn(f"[Resume] scenario parse failed: {e}")
            return None

        gxyz = None
        # current schema: robot.waypoints (goal = last waypoint)
        rob = data.get('robot')
        if isinstance(rob, dict):
            wps = rob.get('waypoints') or []
            if wps:
                gxyz = wps[-1]
        # legacy schema: robots[0].goal
        if gxyz is None:
            robs = data.get('robots') or []
            if robs and isinstance(robs[0], dict) and robs[0].get('goal'):
                gxyz = robs[0]['goal']
        if not gxyz or len(gxyz) < 2:
            self.get_logger().warn(f"[Resume] no goal in {path}")
            return None

        gx, gy = float(gxyz[0]), float(gxyz[1])
        gyaw = math.radians(float(gxyz[2])) if len(gxyz) > 2 else 0.0
        g = PoseStamped()
        g.header.frame_id = 'map'
        g.pose.position.x = gx
        g.pose.position.y = gy
        g.pose.orientation.z = math.sin(gyaw / 2.0)
        g.pose.orientation.w = math.cos(gyaw / 2.0)
        self.get_logger().info(f"[Resume] original goal G = ({gx:.2f}, {gy:.2f}) from {scenario}")
        return g

    def _can_plan_to(self, x: float, y: float, timeout: float = 5.0) -> bool:
        """Ask the global planner if a path from the robot's current pose to (x,y) exists.
        Used to test whether pedestrians have cleared enough to get back to spot A."""
        if not self._compute_path_cli.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("[Resume] ComputePathToPose server unavailable; assume blocked")
            return False
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = "map"
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.pose.position.x = float(x)
        goal.goal.pose.position.y = float(y)
        goal.goal.pose.orientation.w = 1.0
        goal.planner_id = ""
        fut = self._compute_path_cli.send_goal_async(goal)
        deadline = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return False
        rfut = gh.get_result_async()
        while not rfut.done():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        res = rfut.result()
        return res is not None and res.status == 4 and len(res.result.path.poses) > 0  # SUCCEEDED

    def _request_pause_and_capture(self) -> bool:
        """Re-trigger a yield: pause the sim, then request a fresh snapshot. Isaac will
        publish snapshot_ready, which re-enters _cb_snapshot_ready for the next round."""
        if not self._call_trigger(self._pause, "pause", 2.0):
            self.get_logger().error("[Re-yield] pause failed; cannot capture")
            return False
        return self._call_trigger(self._capture, "capture", 2.0)

    def _call_trigger(self, client, name: str, timeout: float) -> bool:
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(f"[Re-yield] {name} service unavailable")
            return False
        fut = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > deadline:
                self.get_logger().warn(f"[Re-yield] {name} timed out")
                return False
            time.sleep(0.02)
        r = fut.result()
        return bool(r is not None and r.success)

    def _run_replan(self, snapshot_dir: str):
        try:
            self.get_logger().info(f"[Replan] step 1/4: snapshot={snapshot_dir}")
            d = Path(snapshot_dir)
            if not d.is_dir():
                self.get_logger().error(f"[Replan] snapshot dir not found: {d}")
                return

            # Episode bookkeeping: on the FIRST yield of an episode, capture spot A
            # (where we're blocked now) and the original goal G. A is fixed for the
            # whole episode; re-yields do NOT update it.
            if not self._yield_active:
                a = self._robot_map_pose()
                if a is not None:
                    self._pos_A = a
                self._orig_goal_G = self._load_original_goal_G()
                self._yield_active = True
                self._yield_round = 0
                self._publish_active(True)   # suppress the proactive trigger for the whole episode
                self.get_logger().warn(
                    f"[Yield] episode START | A={('(%.2f,%.2f)' % self._pos_A) if self._pos_A else 'TF-FAIL'} "
                    f"| G={'ok' if self._orig_goal_G else 'MISSING'}")
            self._yield_round += 1
            self.get_logger().info(f"[Yield] round {self._yield_round}/{MAX_YIELD_ROUNDS}")

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

            # 5. drive to Y and wait for arrival (TF map distance). No en-route checking.
            if not self._drive_and_wait_arrival(goal):
                # STATE LEAK if we just return here: _yield_active stays True, so
                # social_yielding/active remains latched True and the proactive trigger
                # skips EVERY later episode ("orchestrator owns control"). The robot also
                # keeps Y as its goal and never gets G back, so it parks at the yield spot
                # for the rest of the run. Ending the episode is the honest recovery: give
                # the robot its real goal back and re-arm the trigger.
                #
                # A pause (superseded) is different from a timeout: on pause another yield
                # cycle is already starting and it owns the flow, so leave state alone.
                if not self._sim_running:
                    self.get_logger().warn(
                        "[Yield] arrival wait superseded (sim paused); leaving episode state "
                        "to the cycle that paused us")
                    return
                self.get_logger().error(
                    f"[Yield] did not reach Y within {ARRIVE_TIMEOUT}s -> ENDING the yield "
                    f"episode and resuming the original goal G, so the trigger is re-armed "
                    f"instead of being latched off for the rest of the run")
                self._resume_original_goal()
                return

            # 6. RECOVERY DECISION: at Y, can we now plan back to spot A? Yes -> pedestrians
            #    cleared -> resume G. No -> re-yield (unless we've hit the safety cap).
            self._decide_resume_or_reyield()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"[Replan] crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._busy = False
            # Snapshot retention. Safe here and only here: every reader of the dir
            # (selector, pixel_to_map) ran inside the try above, so by now it is dead
            # weight. Deleting it earlier would break the re-yield path, which needs a
            # readable dir for each round.
            self._discard_snapshot(snapshot_dir)
            # If A was still blocked, trigger the next yield now that _busy is clear
            # (pause+capture -> Isaac publishes snapshot_ready -> _cb_snapshot_ready re-enters).
            if self._pending_reyield:
                self._pending_reyield = False
                self._request_pause_and_capture()

    def _discard_snapshot(self, snapshot_dir: str):
        """Delete the snapshot dir unless SAVE_SNAPSHOTS. Never raises.

        Failing to clean up must not take down a yield episode, so this logs and moves
        on. The rmtree is scoped to a dir that is named after the timestamp the capturer
        made and that we only ever learn about via snapshot_ready.
        """
        if SAVE_SNAPSHOTS:
            return
        import shutil
        try:
            d = Path(snapshot_dir)
            # Only remove what the capturer made: a dir directly under a "snapshots" parent.
            # Guards against wiping something else if snapshot_ready ever carries a bad path.
            if d.is_dir() and d.parent.name == "snapshots":
                shutil.rmtree(d)
                self.get_logger().info(f"[Snapshot] discarded {d} (ARENA_SAVE_SNAPSHOTS=false)")
            else:
                self.get_logger().warn(
                    f"[Snapshot] refusing to discard unexpected path (not <...>/snapshots/<stamp>): {d}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"[Snapshot] could not discard {snapshot_dir}: {e}")

    def _drive_and_wait_arrival(self, goal: PoseStamped) -> bool:
        """Drive to yield goal Y and wait for arrival, streaming the control chain meanwhile.
        Returns True once within ARRIVE_TOL of Y (TF map distance), False on timeout or if the
        sim pauses (superseded). The path-back-to-A test happens ONLY after arrival
        (in _decide_resume_or_reyield), not en route."""
        gx, gy = goal.pose.position.x, goal.pose.position.y
        self.get_logger().warn(
            f"[Yield] driving to Y=({gx:+.2f},{gy:+.2f}); waiting for arrival (tol={ARRIVE_TOL}m)")
        self._probe_epoch += 1
        my_epoch = self._probe_epoch
        start = time.monotonic()
        # Silent while healthy: motion detail is the controller's job now ([MOTION-DIAG] /
        # [CMD-CHAIN]). We still sample at 0.5s because ARRIVAL DETECTION needs it -- we just
        # do not print each sample. Distance is reported on the outcome lines instead.
        ARRIVE_SAMPLE = 0.5
        dist = float('nan')
        start_dist = None
        tf_failures = 0
        while my_epoch == self._probe_epoch and self._sim_running:
            time.sleep(ARRIVE_SAMPLE)
            if time.monotonic() - start > ARRIVE_TIMEOUT:
                # Timeout is the decisive moment, so make this line carry everything: how far
                # we still are, how much ground we actually covered, and WHO was driving
                # (FollowPath vs a BT recovery) -- none of which [MOTION-DIAG] can tell us.
                closed = (start_dist - dist) if (start_dist is not None and dist == dist) else float('nan')
                self.get_logger().warn(
                    f"[Yield] arrival timeout ({ARRIVE_TIMEOUT}s): dist={dist:.2f}m, "
                    f"closed only {closed:.2f}m since start | CONTROL={self._control_source()}"
                    + (f" | TF unavailable {tf_failures}x" if tf_failures else ""))
                return False
            pose = self._robot_map_pose()
            if pose is None:
                tf_failures += 1
                continue
            dist = math.hypot(gx - pose[0], gy - pose[1])
            if start_dist is None:
                start_dist = dist
            if dist <= ARRIVE_TOL:
                self.get_logger().warn(f"[Yield] REACHED Y (dist={dist:.2f}m)")
                return True
        return False

    def _decide_resume_or_reyield(self):
        """At yield goal Y: can we plan from here back to spot A?
          - yes -> pedestrians cleared, resume original goal G (episode ends).
          - no  -> still blocked; re-yield (pause+capture) unless MAX_YIELD_ROUNDS hit.
        """
        if self._pos_A is None:
            self.get_logger().warn("[Resume] no spot-A recorded; forcing resume of G")
            self._resume_original_goal()
            return

        ax, ay = self._pos_A
        can_back = self._can_plan_to(ax, ay)
        self.get_logger().warn(
            f"[Resume] path back to A=({ax:.2f},{ay:.2f})? -> {'YES' if can_back else 'NO'}")

        if can_back:
            self.get_logger().warn("[Resume] pedestrians cleared -> resuming original goal G")
            self._resume_original_goal()
        elif self._yield_round >= MAX_YIELD_ROUNDS:
            self.get_logger().error(
                f"[Resume] still blocked after {self._yield_round} yields (cap={MAX_YIELD_ROUNDS}); "
                f"GIVING UP re-yield and force-resuming original goal G")
            self._resume_original_goal()
        else:
            self.get_logger().warn(
                f"[Re-yield] A still blocked; capturing again for round {self._yield_round + 1}")
            # pause+capture -> Isaac publishes snapshot_ready -> _cb_snapshot_ready re-enters.
            # _busy is cleared in the finally of this run; request AFTER we return so the next
            # snapshot isn't dropped. Schedule via a tiny timer so this thread unwinds first.
            self._pending_reyield = True

    def _resume_original_goal(self):
        """End the yield episode: publish original goal G (if known) and reset state."""
        if self._orig_goal_G is not None:
            g = self._orig_goal_G
            g.header.stamp = self.get_clock().now().to_msg()
            self._goal_pub.publish(g)
            self.get_logger().warn(
                f"[Resume] published original goal G=({g.pose.position.x:.2f},"
                f"{g.pose.position.y:.2f}); resuming normal navigation")
        else:
            self.get_logger().error("[Resume] original goal G unknown; cannot resume automatically")
        self._yield_active = False
        self._pos_A = None
        self._yield_round = 0
        self._publish_active(False)   # episode over -> proactive trigger may fire again

    def _publish_active(self, active: bool):
        """Latched flag: True for the whole yield episode (incl. re-yields) so the proactive
        trigger won't fire another yield mid-episode."""
        m = Bool()
        m.data = bool(active)
        self._active_pub.publish(m)

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
