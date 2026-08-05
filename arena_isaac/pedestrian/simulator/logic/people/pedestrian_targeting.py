"""Target planning for the HuNav -> Isaac pedestrian bridge.

Pure, dependency-free logic extracted so it can be unit-tested in the evaluator container (where
``carb`` and ``isaacsim_msgs`` do not exist) and so the two measured defects it fixes are visible
and regression-tested.

The bridge converts a HuNav *logical* pedestrian pose into a walk target for the Isaac animation
graph.  It ran with a 0.01 m dead band and an unconditional 2.0 m look-ahead, which produced two
measured problems (Lane F5, ``LANE_F5_REPORT.md`` §1.5 / §8 R4):

* a **stationary** logical agent still commanded a full 2.0 m walk whenever the residual exceeded
  1 cm, at the 0.8 m/s fallback speed, because the residual direction is pure noise at that scale.
  The rendered character accumulated 0.861 m of wander over 75.3 s of simulated time while the
  logical agent was frozen, moved *away* from its commanded target in 48 of 530 logged intervals,
  and the rendered path exceeded the logical path in 11 of 12 agent-runs;
* the look-ahead was never clamped, so even a 0.02 m residual produced a target 2.0 m past the
  logical pose.  This matters far more once the animation clock is fixed: with the rendered
  character actually keeping up, the residual is small almost all the time, so an unclamped 2.0 m
  look-ahead would become the dominant error rather than a corner case.

The dead band is set to the same 0.25 m the vendor extension uses for arrival
(``omni.anim.people`` ``MinDistanceToIntermediateTarget`` / ``final_target_distance``), and it sits
just below ``Person.update()``'s 0.3 m waypoint-pop threshold, so the effective arrival tolerance is
the 0.3 m the consumer already enforces rather than a second, conflicting number.
"""

import math

__all__ = [
    "LOOK_AHEAD_M",
    "ARRIVAL_DEAD_BAND_M",
    "LEGACY_ARRIVAL_DEAD_BAND_M",
    "DEFAULT_WALK_SPEED_MPS",
    "MIN_WALK_SPEED_MPS",
    "PedestrianTarget",
    "plan_pedestrian_target",
]

#: How far past the current pose a waypoint may be projected.  It must exceed ``Person.update()``'s
#: 0.3 m pop threshold for a genuinely distant goal, or the waypoint is popped before it is walked.
LOOK_AHEAD_M = 2.0

#: Below this residual the character is told to idle instead of chasing direction noise.
ARRIVAL_DEAD_BAND_M = 0.25

#: The value that shipped before this fix; kept so the defect can be reproduced in a test.
LEGACY_ARRIVAL_DEAD_BAND_M = 0.01

DEFAULT_WALK_SPEED_MPS = 0.8
MIN_WALK_SPEED_MPS = 0.05


class PedestrianTarget:
    """A walk command: where to walk to, and how fast."""

    __slots__ = ("target", "walk_speed", "residual_m")

    def __init__(self, target, walk_speed, residual_m):
        self.target = (float(target[0]), float(target[1]), float(target[2]))
        self.walk_speed = float(walk_speed)
        self.residual_m = float(residual_m)

    def __repr__(self):  # pragma: no cover - diagnostics only
        return ("PedestrianTarget(target=%r, walk_speed=%.3f, residual_m=%.4f)"
                % (self.target, self.walk_speed, self.residual_m))

    def __eq__(self, other):  # pragma: no cover - tests compare fields directly
        return (isinstance(other, PedestrianTarget) and self.target == other.target
                and self.walk_speed == other.walk_speed)


def plan_pedestrian_target(current_pos, goal_pos, goal_velocity,
                           look_ahead_m=LOOK_AHEAD_M,
                           dead_band_m=ARRIVAL_DEAD_BAND_M,
                           default_walk_speed=DEFAULT_WALK_SPEED_MPS,
                           min_walk_speed=MIN_WALK_SPEED_MPS,
                           clamp_look_ahead=True):
    """Plan the animation-graph walk target, or ``None`` when the character should idle.

    ``current_pos`` is the *rendered* pose (read back from the animation graph) and ``goal_pos`` is
    HuNav's *logical* pose.  The look-ahead is clamped to the residual, so the target is never
    beyond the logical pose: the character walks *to* where HuNav says it is, never past it.

    ``dead_band_m``, ``look_ahead_m`` and ``clamp_look_ahead`` are parameters rather than constants
    purely so that the pre-fix behaviour (``dead_band_m=LEGACY_ARRIVAL_DEAD_BAND_M`` together with
    ``clamp_look_ahead=False``) can be reproduced in a test as a positive control.  Production
    always uses the defaults.
    """
    cx, cy, cz = (float(current_pos[0]), float(current_pos[1]), float(current_pos[2]))
    gx, gy, gz = (float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
    dx, dy, dz = gx - cx, gy - cy, gz - cz
    residual = math.sqrt(dx * dx + dy * dy + dz * dz)

    if residual < float(dead_band_m):
        return None

    reach = min(float(look_ahead_m), residual) if clamp_look_ahead else float(look_ahead_m)
    scale = reach / residual
    target = (cx + dx * scale, cy + dy * scale, cz + dz * scale)

    requested = float(goal_velocity or 0.0)
    walk_speed = requested if requested >= float(min_walk_speed) else float(default_walk_speed)
    return PedestrianTarget(target, walk_speed, residual)
