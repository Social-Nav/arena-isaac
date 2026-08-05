"""Render-cadence-independent animation clock policy for Isaac pedestrians.

Why this module exists
----------------------
``omni.anim.graph.core`` evaluates a character's animation graph **only** from Kit's application
update loop.  The Arena eval loop advances Isaac with ``world.step(render=should_render)`` and
renders one step in ``ARENA_ISAAC_RENDER_EVERY_N_STEPS`` (6 in headless), and
``world.step(render=False)`` never calls ``app.update()``.  The graph therefore received a single
~1/60 s animation frame per ~0.18 s of simulated time, and the rendered pedestrians walked at
~9 % of their commanded speed while the evaluation graded HuNav's *logical* positions -- a
divergence measured at up to 12.63 m.

``omni.anim.graph.core.Character.update(dt)`` ticks one character by an explicit dt and is bound on
the object ``Person.character_graph`` already holds.  Driving it from the existing physics callback
fixes the starvation **without changing the render cadence at all**, which is a hard project
requirement: ``render_every_n_steps > 1`` exists so the benchmark can run near real time and so the
render pipeline's GPU cost stays bounded.

Measured behaviour of the installed ``omni.anim.graph.core-107.3.4`` that this module encodes
(evidence: ``tmp/lane_r_pedestrian_animation/GATE0_RESULT.md``):

1. The tick advances the character with **zero** ``app.update()`` and zero ``world.step()`` calls:
   1.000 s of supplied dt produced 1.0259 m against a 1.0884 m/s reference (ratio 0.943).
2. The advanced pose is published to Fabric/USD -- the data Hydra draws from -- on the next
   application update, i.e. on the next *rendered* step.  Mid-burst the Fabric value stays
   bit-stale;
   at the next render it equals the internally ticked value to 6 decimal places.  Since nothing is
   drawn between renders, the frame that is actually drawn carries the full ticked displacement
   (confirmed in rendered pixels: 60.75 px/m for a manual burst vs 63.80 px/m for an automatic one).
3. **Once a character has been ticked manually, automatic evaluation effectively stops for it**
   (60 render steps afterwards advanced it 0.0173 m, 1.6 % of the 1.086 m the same 60 steps produced
   beforehand).  So the tick must supply the *whole* elapsed physics dt on *every* step, **including
   rendered ones**.  Skipping the rendered step would lose 1/6 of the elapsed time; it would not
   avoid double counting, because there is no double counting to avoid.  ``animation_tick_dt()``
   exists to make that decision explicit and regression-tested.
4. ``Character.update()`` returns ``None`` and does **not** raise when the timeline is paused or
   stopped -- it fails *silently*.  Its return value is therefore worthless as a health signal, so
   :class:`AnimationTickHealth` measures actual progress instead.  This is the whole reason the
   original defect survived unnoticed: nothing in the pipeline complained.

Everything here is pure Python (stdlib only) so it can be unit-tested in the evaluator container,
where ``carb`` / ``omni.*`` do not exist.
"""

import os

__all__ = [
    "MANUAL_TICK_ENV_VAR",
    "PUBLISH_RENDERED_STATES_ENV_VAR",
    "RENDERED_STATES_TOPIC_ENV_VAR",
    "DEFAULT_RENDERED_STATES_TOPIC",
    "env_flag",
    "manual_tick_enabled",
    "animation_tick_dt",
    "AnimationTickHealth",
    "AnimationGraphAcquisition",
]

MANUAL_TICK_ENV_VAR = "ARENA_ISAAC_PEDESTRIAN_ANIMATION_TICK"
PUBLISH_RENDERED_STATES_ENV_VAR = "ARENA_ISAAC_PUBLISH_RENDERED_PEDESTRIAN_STATES"
RENDERED_STATES_TOPIC_ENV_VAR = "ARENA_ISAAC_RENDERED_PEDESTRIAN_STATES_TOPIC"
DEFAULT_RENDERED_STATES_TOPIC = "/task_generator_node/pedestrian_rendered_states"

_FALSEY = {"0", "false", "no", "off", ""}


def env_flag(name: str, default: bool = True, environ=None) -> bool:
    """Read a boolean ``ARENA_ISAAC_*`` switch using this repository's existing convention."""
    environ = os.environ if environ is None else environ
    raw = environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in _FALSEY


def manual_tick_enabled(environ=None) -> bool:
    """Is the manual animation tick enabled?  Default **on**.

    Turning it off restores the starved ~1/``render_every_n_steps`` behaviour, so the caller must
    announce the downgrade loudly -- see :func:`manual_tick_disabled_message`.
    """
    return env_flag(MANUAL_TICK_ENV_VAR, True, environ)


def manual_tick_disabled_message(render_every_n_steps=None) -> str:
    extra = ""
    if render_every_n_steps:
        extra = (" With render_every_n_steps=%s the rendered pedestrians will move at roughly "
                 "1/%s of their commanded speed." % (render_every_n_steps, render_every_n_steps))
    return ("[PedestrianAnimation] MANUAL ANIMATION TICK DISABLED by %s -- the animation graph "
            "will again be evaluated only on rendered steps, so rendered pedestrians will lag the "
            "logical "
            "positions the evaluation grades against.%s Unset %s to restore the fix."
            % (MANUAL_TICK_ENV_VAR, extra, MANUAL_TICK_ENV_VAR))


def animation_tick_dt(physics_dt: float, is_render_step: bool = False) -> float:
    """How much animation time to request for a physics step of ``physics_dt``.

    Always the whole ``physics_dt``, **including on rendered steps**.  Measured fact 3 in the module
    docstring: manual ticking switches automatic evaluation off for that character, so a rendered
    step contributes essentially nothing on its own and must be ticked like any other.  The
    ``is_render_step`` argument is accepted, and deliberately ignored, so that the decision is
    visible at the call site and locked down by a test.
    """
    del is_render_step
    dt = float(physics_dt)
    return dt if dt > 0.0 else 0.0


class AnimationTickHealth:
    """Detects the one failure mode ``Character.update()`` cannot report: no progress.

    The tick is only allowed to be judged over intervals where a walk was actually commanded, so
    idle pedestrians never trip it.  What it compares is *distance the character actually covered*
    against *distance the commanded speed implies*, both integrated over the same intervals.

    Scope, stated plainly: this monitors the pose read back from the animation graph.  It would not
    notice a hypothetical failure in which the graph advances internally but the render path never
    receives the pose -- that mode was measured **not** to occur (the pose is published on the next
    application update), and detecting it would require a renderer-side probe.
    """

    def __init__(self, min_commanded_seconds=5.0, stalled_ratio=0.25, healthy_ratio=0.60,
                 recheck_seconds=60.0):
        self.min_commanded_seconds = float(min_commanded_seconds)
        self.stalled_ratio = float(stalled_ratio)
        self.healthy_ratio = float(healthy_ratio)
        self.recheck_seconds = float(recheck_seconds)
        self.ticked_seconds = 0.0
        self.commanded_seconds = 0.0
        self.expected_distance_m = 0.0
        self.travelled_distance_m = 0.0
        self._last_position = None
        self._reported_ok = False
        self._commanded_seconds_at_last_report = None

    def record(self, dt, commanded_speed, position):
        """Fold one ticked physics step in; return a message to log, or ``None``.

        ``position`` is the pose read back from the animation graph (a 2- or 3-element sequence).
        """
        dt = float(dt)
        if dt > 0.0:
            self.ticked_seconds += dt
        commanded_speed = float(commanded_speed or 0.0)
        position = None if position is None else [float(v) for v in position]

        if commanded_speed > 0.0 and dt > 0.0:
            self.commanded_seconds += dt
            self.expected_distance_m += commanded_speed * dt
            if self._last_position is not None and position is not None:
                self.travelled_distance_m += _planar_distance(self._last_position, position)
        self._last_position = position
        return self._verdict_message()

    def ratio(self):
        if self.expected_distance_m <= 0.0:
            return None
        return self.travelled_distance_m / self.expected_distance_m

    def _verdict_message(self):
        if self.commanded_seconds < self.min_commanded_seconds:
            return None
        ratio = self.ratio()
        if ratio is None:
            return None
        if ratio < self.stalled_ratio:
            if self._due_for_report():
                self._commanded_seconds_at_last_report = self.commanded_seconds
                return ("error", self._describe(
                    "PEDESTRIAN ANIMATION IS NOT ADVANCING. Character.update(dt) was called for "
                    "%.2f s of commanded walking but the character covered only %.3f m of the "
                    "%.3f m the commanded speed implies. Character.update() fails silently when "
                    "the timeline is not playing, so this is the only signal there is."
                    % (self.commanded_seconds, self.travelled_distance_m,
                       self.expected_distance_m)))
            return None
        if not self._reported_ok and ratio >= self.healthy_ratio:
            self._reported_ok = True
            self._commanded_seconds_at_last_report = self.commanded_seconds
            return ("info", self._describe(
                "manual animation tick is advancing the character: %.3f m covered of %.3f m "
                "commanded over %.2f s of commanded walking"
                % (self.travelled_distance_m, self.expected_distance_m, self.commanded_seconds)))
        return None

    def _due_for_report(self):
        if self._commanded_seconds_at_last_report is None:
            return True
        return (self.commanded_seconds - self._commanded_seconds_at_last_report
                >= self.recheck_seconds)

    def _describe(self, text):
        return ("[PedestrianAnimation] %s (ratio=%s, ticked=%.2f s)"
                % (text, "n/a" if self.ratio() is None else "%.3f" % self.ratio(),
                   self.ticked_seconds))


class AnimationGraphAcquisition:
    """Policy for acquiring the animation graph without silence and without USD churn.

    The original code re-ran ``add_animation_graph_to_agent()`` -- which executes
    ``RemoveAnimationGraphAPICommand`` followed by ``ApplyAnimationGraphAPICommand`` -- on **every**
    property access while the graph was unresolved, i.e. potentially on every physics step, and if
    it never resolved both physics callbacks simply returned and the pedestrians froze with no error
    at all.  This policy:

    * authors the AnimationGraphAPI on the first attempt and then at most once per
      ``authoring_interval_s``, querying in between (a repeated remove-then-reapply can tear down an
      API that is in the middle of becoming visible);
    * escalates to a single loud error once ``error_after_s`` has elapsed without success, and
      repeats it no more often than ``error_repeat_s`` so a permanent failure is never silent;
    * emits a one-shot success marker, which doubles as the treatment check that the graph really
      was acquired.
    """

    def __init__(self, authoring_interval_s=1.0, error_after_s=10.0, error_repeat_s=60.0):
        self.authoring_interval_s = float(authoring_interval_s)
        self.error_after_s = float(error_after_s)
        self.error_repeat_s = float(error_repeat_s)
        self.attempts = 0
        self.authoring_attempts = 0
        self.first_attempt_at = None
        self._last_authoring_at = None
        self._last_error_at = None

    def next_action(self, now):
        """Return ``"author"`` (author then query) or ``"query"`` (query only)."""
        now = float(now)
        self.attempts += 1
        if self.first_attempt_at is None:
            self.first_attempt_at = now
        if self._last_authoring_at is None or (now - self._last_authoring_at
                                               >= self.authoring_interval_s):
            self._last_authoring_at = now
            self.authoring_attempts += 1
            return "author"
        return "query"

    def elapsed(self, now):
        if self.first_attempt_at is None:
            return 0.0
        return max(0.0, float(now) - self.first_attempt_at)

    def success_message(self, now, prim_path):
        return ("[PedestrianAnimation] animation graph acquired for %s after %d attempt(s) / "
                "%.2f s (%d AnimationGraphAPI authoring pass(es))"
                % (prim_path, self.attempts, self.elapsed(now), self.authoring_attempts))

    def failure_message(self, now, prim_path):
        """One loud message per ``error_repeat_s`` once the deadline has passed, else ``None``."""
        now = float(now)
        if self.elapsed(now) < self.error_after_s:
            return None
        if self._last_error_at is not None and (now - self._last_error_at) < self.error_repeat_s:
            return None
        self._last_error_at = now
        return ("[PedestrianAnimation] COULD NOT ACQUIRE THE ANIMATION GRAPH for %s after %d "
                "attempt(s) over %.2f s. The animation graph is the only transport that moves a "
                "rendered pedestrian, so this pedestrian is frozen. Previously this failed with no "
                "error at all." % (prim_path, self.attempts, self.elapsed(now)))


def _planar_distance(a, b):
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    return (dx * dx + dy * dy) ** 0.5
