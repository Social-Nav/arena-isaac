"""Tests for the pedestrian animation-clock fix and the HuNav -> Isaac target planner.

These exercise the real production modules.  They deliberately avoid importing ``person.py`` /
``NavigatePedestrians.py`` directly, because those pull in ``carb`` and ``omni.*``, which exist only
inside the Isaac container; the logic under test was placed in dependency-free modules for exactly
that reason.

Every test that guards a fix also reproduces the pre-fix behaviour through the same code path (by
passing the pre-fix parameters), so a regression fails on a substantive numeric assertion rather
than on an import error.
"""

import math

import pytest

from pedestrian.simulator.logic.people.animation_clock import (
    AnimationGraphAcquisition,
    AnimationTickHealth,
    MANUAL_TICK_ENV_VAR,
    animation_tick_dt,
    env_flag,
    manual_tick_enabled,
)
from pedestrian.simulator.logic.people.pedestrian_targeting import (
    ARRIVAL_DEAD_BAND_M,
    LEGACY_ARRIVAL_DEAD_BAND_M,
    LOOK_AHEAD_M,
    plan_pedestrian_target,
)
from pedestrian.simulator.logic.people.rendered_state_backend import (
    PUBLISH_RENDERED_STATES_ENV_VAR,
    PedestrianRenderedStateBackend,
    RenderedPedestrianStatePublisher,
    make_backend_if_enabled,
)

WAYPOINT_POP_THRESHOLD_M = 0.3  # Person.update()'s THRESHOLD_DISTANCE


def _dist(a, b):
    return math.dist(a[:2], b[:2])


# --------------------------------------------------------------------------------------------
# F-A: the animation tick must supply the whole dt on every step, rendered ones included
# --------------------------------------------------------------------------------------------

def test_animation_tick_dt_is_the_full_physics_dt_on_a_rendered_step():
    """Gate-0 measured that manual ticking switches automatic evaluation off for that character.

    So a rendered step must be ticked like any other; skipping it to "avoid double counting" would
    silently drop 1/render_every_n_steps of the elapsed time.
    """
    assert animation_tick_dt(1.0 / 60.0, is_render_step=False) == pytest.approx(1.0 / 60.0)
    assert animation_tick_dt(1.0 / 60.0, is_render_step=True) == pytest.approx(1.0 / 60.0)
    assert animation_tick_dt(1.0 / 30.0, is_render_step=True) == pytest.approx(1.0 / 30.0)


def test_animation_tick_dt_rejects_non_positive_dt():
    assert animation_tick_dt(0.0) == 0.0
    assert animation_tick_dt(-0.01) == 0.0


def test_ticked_seconds_equals_summed_physics_dt_over_a_render_cycle():
    """The whole point of the fix: animation time must track simulated time 1:1.

    Six steps of a render_every_n_steps=6 cycle, one of which renders.
    """
    health = AnimationTickHealth()
    total = 0.0
    for index in range(6):
        dt = animation_tick_dt(1.0 / 60.0, is_render_step=(index == 5))
        total += dt
        health.record(dt, 0.0, (0.0, 0.0, 0.0))
    assert total == pytest.approx(6.0 / 60.0)
    assert health.ticked_seconds == pytest.approx(6.0 / 60.0)


# --------------------------------------------------------------------------------------------
# F-A: the health monitor is the only signal there is, because the tick fails silently
# --------------------------------------------------------------------------------------------

def _drive(health, seconds, speed, actual_speed, dt=1.0 / 60.0):
    """Feed the monitor `seconds` of commanded walking at `speed`, moving at `actual_speed`."""
    x = 0.0
    messages = []
    steps = int(round(seconds / dt))
    for _ in range(steps):
        x += actual_speed * dt
        report = health.record(dt, speed, (x, 0.0, 0.0))
        if report is not None:
            messages.append(report)
    return messages


def test_health_monitor_flags_a_frozen_character_loudly():
    health = AnimationTickHealth()
    messages = _drive(health, seconds=8.0, speed=1.0, actual_speed=0.0)
    assert messages, "a character that never moved while commanded to walk must be reported"
    severity, text = messages[0]
    assert severity == "error"
    assert "NOT ADVANCING" in text
    assert health.ratio() == pytest.approx(0.0)


def test_health_monitor_flags_the_pre_fix_nine_percent_starvation():
    """The measured pre-fix state (0.079-0.102 of commanded speed) must trip the monitor."""
    health = AnimationTickHealth()
    messages = _drive(health, seconds=8.0, speed=1.0, actual_speed=0.0909)
    assert messages
    assert messages[0][0] == "error"
    assert 0.079 <= health.ratio() <= 0.102, (
        "must land inside the band measured across the six runs; got %.4f" % health.ratio())
    assert health.ratio() < health.stalled_ratio


def test_health_monitor_accepts_a_working_tick_and_says_so_once():
    health = AnimationTickHealth()
    messages = _drive(health, seconds=8.0, speed=1.0, actual_speed=0.95)
    assert len(messages) == 1, "the healthy marker must be emitted exactly once"
    severity, text = messages[0]
    assert severity == "info"
    assert "advancing the character" in text


def test_health_monitor_never_judges_an_idle_pedestrian():
    health = AnimationTickHealth()
    assert _drive(health, seconds=60.0, speed=0.0, actual_speed=0.0) == []
    assert health.commanded_seconds == 0.0
    assert health.ratio() is None
    assert health.ticked_seconds == pytest.approx(60.0, rel=1e-6)


def test_health_monitor_waits_for_its_minimum_window():
    health = AnimationTickHealth(min_commanded_seconds=5.0)
    assert _drive(health, seconds=4.0, speed=1.0, actual_speed=0.0) == []


def test_health_monitor_repeats_a_persistent_stall_but_not_every_step():
    health = AnimationTickHealth(min_commanded_seconds=5.0, recheck_seconds=10.0)
    messages = _drive(health, seconds=40.0, speed=1.0, actual_speed=0.0)
    assert 2 <= len(messages) <= 5, (
        "a permanent failure must keep being reported, but not once per physics step; got %d"
        % len(messages))
    assert all(m[0] == "error" for m in messages)


# --------------------------------------------------------------------------------------------
# F-A: the kill switch must be explicit, and its default must be ON
# --------------------------------------------------------------------------------------------

def test_manual_tick_defaults_to_enabled():
    assert manual_tick_enabled(environ={}) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " 0 ", ""])
def test_manual_tick_can_be_disabled_explicitly(value):
    assert manual_tick_enabled(environ={MANUAL_TICK_ENV_VAR: value}) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
def test_manual_tick_stays_enabled_for_truthy_values(value):
    assert manual_tick_enabled(environ={MANUAL_TICK_ENV_VAR: value}) is True


def test_env_flag_default_is_used_when_unset():
    assert env_flag("ARENA_ISAAC_NOT_SET_ANYWHERE", True, environ={}) is True
    assert env_flag("ARENA_ISAAC_NOT_SET_ANYWHERE", False, environ={}) is False


# --------------------------------------------------------------------------------------------
# F-C: acquiring the animation graph must not churn USD, and must never fail quietly
# --------------------------------------------------------------------------------------------

def test_acquisition_authors_once_then_only_queries_until_the_interval_elapses():
    acq = AnimationGraphAcquisition(authoring_interval_s=1.0)
    assert acq.next_action(100.0) == "author"
    assert acq.next_action(100.02) == "query"
    assert acq.next_action(100.5) == "query"
    assert acq.next_action(101.0) == "author"
    assert acq.attempts == 4
    assert acq.authoring_attempts == 2


def test_acquisition_stays_quiet_before_its_deadline():
    acq = AnimationGraphAcquisition(error_after_s=10.0)
    acq.next_action(0.0)
    for now in (0.1, 1.0, 5.0, 9.9):
        acq.next_action(now)
        assert acq.failure_message(now, "/World/Pedestrians/hunav_1") is None


def test_acquisition_escalates_loudly_after_the_deadline_and_keeps_reminding():
    acq = AnimationGraphAcquisition(error_after_s=10.0, error_repeat_s=60.0)
    acq.next_action(0.0)
    assert acq.failure_message(9.0, "/p") is None
    first = acq.failure_message(10.0, "/p")
    assert first is not None
    assert "COULD NOT ACQUIRE THE ANIMATION GRAPH" in first
    assert "/p" in first
    assert acq.failure_message(11.0, "/p") is None, "must not shout every physics step"
    assert acq.failure_message(75.0, "/p") is not None, "must not go silent either"


def test_acquisition_success_marker_carries_the_attempt_count():
    acq = AnimationGraphAcquisition()
    acq.next_action(0.0)
    acq.next_action(0.1)
    message = acq.success_message(0.2, "/World/Pedestrians/hunav_1/ManRoot/char")
    assert "animation graph acquired" in message
    assert "2 attempt(s)" in message
    assert "/World/Pedestrians/hunav_1/ManRoot/char" in message


# --------------------------------------------------------------------------------------------
# F-D: the sub-threshold look-ahead amplification
# --------------------------------------------------------------------------------------------

def test_pre_fix_parameters_amplify_a_2cm_residual_into_a_2m_walk():
    """Positive control: the same code path with the pre-fix parameters reproduces the defect."""
    current = (0.0, 0.0, 0.0)
    goal = (0.02, 0.0, 0.0)
    plan = plan_pedestrian_target(current, goal, 0.0,
                                  look_ahead_m=2.0,
                                  dead_band_m=LEGACY_ARRIVAL_DEAD_BAND_M,
                                  clamp_look_ahead=False)
    assert plan is not None
    assert _dist(current, plan.target) == pytest.approx(2.0)
    assert plan.walk_speed == pytest.approx(0.8), "and at the 0.8 m/s fallback speed"


@pytest.mark.parametrize("residual,expected_amplification",
                         [(0.011, 181.8), (0.02, 100.0), (0.1, 20.0), (0.5, 4.0)])
def test_pre_fix_amplification_grows_without_bound_as_the_residual_shrinks(
        residual, expected_amplification):
    """Model-free statement of the defect: commanded walk / residual, pre-fix.

    The commanded target sat a fixed 2.0 m away no matter how small the residual, so the
    amplification factor diverges as the residual approaches the 1 cm dead band.
    """
    current = (0.0, 0.0, 0.0)
    plan = plan_pedestrian_target(current, (residual, 0.0, 0.0), 0.0,
                                  look_ahead_m=2.0,
                                  dead_band_m=LEGACY_ARRIVAL_DEAD_BAND_M,
                                  clamp_look_ahead=False)
    assert plan is not None
    amplification = _dist(current, plan.target) / residual
    assert amplification == pytest.approx(expected_amplification, rel=1e-2)


@pytest.mark.parametrize("residual", [0.25, 0.3, 0.5, 1.0, 1.9])
def test_fixed_amplification_is_exactly_one_above_the_dead_band(residual):
    current = (0.0, 0.0, 0.0)
    plan = plan_pedestrian_target(current, (residual, 0.0, 0.0), 0.0)
    assert plan is not None
    assert _dist(current, plan.target) / residual == pytest.approx(1.0)


@pytest.mark.parametrize("residual", [0.0, 0.005, 0.011, 0.02, 0.1, 0.2, 0.249])
def test_fixed_parameters_command_nothing_below_the_dead_band(residual):
    assert plan_pedestrian_target((0.0, 0.0, 0.0), (residual, 0.0, 0.0), 0.0) is None


def test_a_2cm_residual_now_idles_the_character():
    assert plan_pedestrian_target((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), 0.0) is None


def test_look_ahead_is_clamped_to_the_residual_so_the_target_is_never_past_the_logical_pose():
    current = (0.0, 0.0, 0.0)
    for residual in (0.3, 0.5, 1.0, 1.9):
        plan = plan_pedestrian_target(current, (residual, 0.0, 0.0), 0.5)
        assert plan is not None
        assert _dist(current, plan.target) == pytest.approx(residual)
        assert plan.target[0] <= residual + 1e-9


def test_a_distant_goal_still_gets_the_full_look_ahead_beyond_the_pop_threshold():
    """The look-ahead exists so the waypoint survives Person.update()'s 0.3 m pop threshold."""
    current = (0.0, 0.0, 0.0)
    plan = plan_pedestrian_target(current, (12.63, 0.0, 0.0), 0.9)
    assert plan is not None
    assert _dist(current, plan.target) == pytest.approx(LOOK_AHEAD_M)
    assert _dist(current, plan.target) > WAYPOINT_POP_THRESHOLD_M
    assert plan.walk_speed == pytest.approx(0.9)


def test_dead_band_sits_below_the_waypoint_pop_threshold():
    """Otherwise two different arrival tolerances would fight each other."""
    assert 0.0 < ARRIVAL_DEAD_BAND_M < WAYPOINT_POP_THRESHOLD_M


def test_requested_speed_below_the_floor_falls_back_to_the_default():
    plan = plan_pedestrian_target((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), 0.01)
    assert plan is not None
    assert plan.walk_speed == pytest.approx(0.8)


def test_residual_is_reported_for_diagnostics():
    plan = plan_pedestrian_target((0.0, 0.0, 0.0), (3.0, 4.0, 0.0), 1.0)
    assert plan is not None
    assert plan.residual_m == pytest.approx(5.0)


def _simulate_commanded_walk(dead_band_m, look_ahead_m, clamp_look_ahead, seconds=60.0,
                             dt=1.0 / 30.0, sway_m=0.02, frozen_goal=(0.0, 0.0, 0.0),
                             start=(0.0, 0.0, 0.0)):
    """Closed loop against a frozen logical agent; returns the integrated *commanded* walk distance.

    Mirrors the real loop: ``navigate_pedestrian`` re-plans from the *rendered* pose every step and
    ``Person.update()`` pops any waypoint closer than 0.3 m.

    ``sway_m`` is an explicit, declared **assumption**, not a measurement: the pose read back from
    the animation graph is the character's root, which jitters with the walk/idle cycle and with
    turning, so it is not a noiseless point.  The result asserted is the *contrast*, which holds for
    any jitter between the pre-fix dead band (0.01 m) and the fixed one (0.25 m).  That is the
    mechanism: a dead band below the animation's own positional noise floor can never be satisfied,
    so the bridge keeps
    commanding a fresh 2 m walk forever; a dead band above it absorbs the noise and the character
    idles.
    """
    walked = [float(v) for v in start]  # the part of the pose produced by commanded walking
    commanded_path = 0.0
    steps = int(round(seconds / dt))
    for index in range(steps):
        rendered = (walked[0], walked[1] + sway_m * math.sin(index * 0.7), walked[2])
        plan = plan_pedestrian_target(rendered, frozen_goal, 0.0,
                                      look_ahead_m=look_ahead_m, dead_band_m=dead_band_m,
                                      clamp_look_ahead=clamp_look_ahead)
        if plan is None:
            continue
        target = plan.target
        if _dist(target, rendered) < WAYPOINT_POP_THRESHOLD_M:
            continue  # Person.update() pops it -> Idle
        step = plan.walk_speed * dt
        remaining = _dist(target, rendered)
        scale = min(1.0, step / remaining)
        before = tuple(walked)
        for axis in range(2):
            walked[axis] += (target[axis] - rendered[axis]) * scale
        commanded_path += _dist(before, walked)
    return commanded_path


def test_pre_fix_parameters_make_a_frozen_pedestrian_walk_forever():
    """Positive control for the measured wander: 0.861 m over 75.3 s with the logical pose frozen."""
    path = _simulate_commanded_walk(dead_band_m=LEGACY_ARRIVAL_DEAD_BAND_M, look_ahead_m=2.0,
                                    clamp_look_ahead=False)
    assert path > 0.5, "pre-fix parameters must reproduce the measured wander; got %.3f m" % path


def test_fixed_parameters_command_no_walk_for_a_frozen_pedestrian():
    path = _simulate_commanded_walk(dead_band_m=ARRIVAL_DEAD_BAND_M, look_ahead_m=LOOK_AHEAD_M,
                                    clamp_look_ahead=True)
    assert path < 0.25, "a frozen logical agent must not make the character walk; got %.3f m" % path


def test_fixed_parameters_still_converge_from_a_real_offset():
    """The dead band must not stop the character from closing a genuine gap."""
    pos = (-3.0, 0.0, 0.0)
    goal = (0.0, 0.0, 0.0)
    for _ in range(400):
        plan = plan_pedestrian_target(pos, goal, 0.8)
        if plan is None:
            break
        target = plan.target
        if _dist(target, pos) < WAYPOINT_POP_THRESHOLD_M:
            break
        step = plan.walk_speed * (1.0 / 30.0)
        remaining = _dist(target, pos)
        scale = min(1.0, step / remaining)
        pos = tuple(pos[axis] + (target[axis] - pos[axis]) * scale for axis in range(3))
    assert _dist(pos, goal) < WAYPOINT_POP_THRESHOLD_M, (
        "must converge to within the pop threshold, ended %.3f m away" % _dist(pos, goal))


# --------------------------------------------------------------------------------------------
# F-B: the rendered pose must become observable
# --------------------------------------------------------------------------------------------

class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _FakeNode:
    def __init__(self):
        self.publisher = _FakePublisher()
        self.created = []

    def create_publisher(self, msg_type, topic, depth):
        self.created.append((msg_type, topic, depth))
        return self.publisher


class _FakeState:
    def __init__(self, position, orientation=(0.0, 0.0, 0.0, 1.0)):
        self.position = position
        self.orientation = orientation


class _FakePerson:
    def __init__(self, path):
        self.path = path


def _publisher_with_fake_node(topic="/test/pedestrian_rendered_states"):
    node = _FakeNode()
    pub = RenderedPedestrianStatePublisher(node=node, topic=topic)
    return pub, node


def test_rendered_state_publisher_emits_one_entry_per_pedestrian():
    pub, node = _publisher_with_fake_node()
    backend_a = PedestrianRenderedStateBackend(publisher=pub)
    backend_b = PedestrianRenderedStateBackend(publisher=pub)
    backend_a.initialize(_FakePerson("/World/Pedestrians/hunav_1"))
    backend_b.initialize(_FakePerson("/World/Pedestrians/hunav_2"))

    assert backend_a.update(_FakeState((1.0, 2.0, 0.0)), 1.0 / 30.0) is True
    assert backend_b.update(_FakeState((5.0, 6.0, 0.0)), 1.0 / 30.0) is True

    assert node.created == [(node.created[0][0], "/test/pedestrian_rendered_states", 10)]
    last = node.publisher.published[-1]
    names = [p.name for p in last.pedestrians]
    assert names == ["hunav_1", "hunav_2"], "the snapshot must always be complete"
    positions = {p.name: (p.pose.position.x, p.pose.position.y) for p in last.pedestrians}
    assert positions["hunav_1"] == pytest.approx((1.0, 2.0))
    assert positions["hunav_2"] == pytest.approx((5.0, 6.0))


def test_rendered_state_publisher_finite_differences_the_rendered_velocity():
    pub, node = _publisher_with_fake_node()
    backend = PedestrianRenderedStateBackend(publisher=pub)
    backend.initialize(_FakePerson("/World/Pedestrians/hunav_1"))
    dt = 1.0 / 30.0
    backend.update(_FakeState((0.0, 0.0, 0.0)), dt)
    first = node.publisher.published[-1].pedestrians[0]
    assert (first.twist.linear.x, first.twist.linear.y) == pytest.approx((0.0, 0.0))
    assert first.animation_state == 0  # IDLE: no motion observed yet

    backend.update(_FakeState((0.02, 0.0, 0.0)), dt)
    second = node.publisher.published[-1].pedestrians[0]
    assert second.twist.linear.x == pytest.approx(0.6, rel=1e-6)
    assert second.animation_state == 1  # WALKING


def test_rendered_state_publisher_stamps_with_sim_time():
    pub, _node = _publisher_with_fake_node()
    pub.submit("hunav_1", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), False, 12.25)
    msg = pub.build_message(12.25)
    assert msg.header.stamp.sec == 12
    assert msg.header.stamp.nanosec == 250000000


def test_rendered_state_backend_is_read_only_with_respect_to_pose():
    """It must never write the pose back: that path is forbidden by troubleshooting.md:238."""
    pub, _node = _publisher_with_fake_node()
    backend = PedestrianRenderedStateBackend(publisher=pub)
    backend.initialize(_FakePerson("/World/Pedestrians/hunav_1"))
    state = _FakeState((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    backend.update(state, 1.0 / 30.0)
    assert state.position == (1.0, 2.0, 3.0)
    assert state.orientation == (0.0, 0.0, 0.0, 1.0)


def test_rendered_state_publishing_defaults_on_and_can_be_switched_off_loudly():
    assert make_backend_if_enabled(environ={}) is not None
    assert make_backend_if_enabled(
        environ={PUBLISH_RENDERED_STATES_ENV_VAR: "0"}) is None
