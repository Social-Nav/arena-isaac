"""Publish the *rendered* pedestrian pose, at physics-callback rate.

Why this exists
---------------
The evaluation's human-proximity metrics are computed from ``human_states.csv``, which is HuNav's
**logical** track: ``hunav.py`` publishes ``response.updated_agents`` and the recorder is a plain
subscription, so no change to metric sampling rate can make that record describe the rendered world.
Meanwhile the only record of where the rendered character actually was came from a ``carb.log_warn``
throttled to one line per prim per 5 s of wall time.  A divergence between the two was therefore
invisible to the pipeline: the defect that made pedestrians render at ~9 % of their commanded speed
produced no artifact at all, and no error.

``Person.update_state()`` already reads the rendered transform out of the animation graph on every
physics step and stores it in ``Person.state``; it was simply thrown away, because
``SpawnPedestrians`` constructed ``Person`` with no ``backend``.  This backend is that missing
consumer.  It is **read-only with respect to pose** -- it publishes what ``update_state`` already
read and never writes a transform, so it does not touch the path forbidden by
``docs/benchmark/troubleshooting.md:238`` (``RemoveAnimationGraphAPICommand``,
``XformPrim.set_world_poses`` / ``set_local_poses``, ``person.state.position = ...``).

Two consequences worth stating so they are not later mistaken for bugs:

* the published pose is a staircase whenever the animation graph is not being ticked every step --
  that is the truth about where the prim is between application updates, and it is exactly what a
  metric computed "against what the camera saw" should use;
* this publishes a topic.  Turning it into a CSV beside ``human_states.csv`` needs a change in
  ``arena_evaluation``'s recorder, which is owned by the deferred metrics work and deliberately not
  touched here.
"""

import math
import os

from pedestrian.simulator.logic.people.animation_clock import (
    DEFAULT_RENDERED_STATES_TOPIC,
    PUBLISH_RENDERED_STATES_ENV_VAR,
    RENDERED_STATES_TOPIC_ENV_VAR,
    env_flag,
)

__all__ = ["RenderedPedestrianStatePublisher", "PedestrianRenderedStateBackend"]


def _log_warn(message):
    try:
        import carb

        carb.log_warn(message)
    except Exception:  # pragma: no cover - only when carb is unavailable
        print(message, flush=True)


def _log_error(message):
    try:
        import carb

        carb.log_error(message)
    except Exception:  # pragma: no cover - only when carb is unavailable
        print(message, flush=True)


def _sim_time():
    try:
        from isaacsim.core.api.simulation_context import SimulationContext
    except Exception:  # pragma: no cover - older Isaac layout
        try:
            from omni.isaac.core import SimulationContext
        except Exception:
            return None
    try:
        return float(SimulationContext.instance().current_time)
    except Exception:
        return None


class RenderedPedestrianStatePublisher:
    """Process-wide publisher shared by every pedestrian backend.

    Each pedestrian writes its latest rendered pose into the snapshot and the whole snapshot is
    published, so a consumer always receives a complete set regardless of callback ordering.  With
    the two HuNav pedestrians of a benchmark episode this is a handful of very small messages per
    physics step.
    """

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Test hook: drop the singleton (also used when a run tears the ROS node down)."""
        cls._instance = None

    def __init__(self, node=None, topic=None):
        self._topic = topic or os.environ.get(RENDERED_STATES_TOPIC_ENV_VAR) \
            or DEFAULT_RENDERED_STATES_TOPIC
        self._node = node
        self._publisher = None
        self._entries = {}
        self._disabled = False
        self._announced = False

    @property
    def topic(self):
        return self._topic

    def _ensure_publisher(self):
        if self._publisher is not None or self._disabled:
            return self._publisher
        try:
            import rclpy

            from arena_people_msgs.msg import Pedestrians

            if self._node is None:
                if not rclpy.ok():
                    # rclpy is initialised in run_isaacsim.main() before any pedestrian can spawn;
                    # if it is not up yet, try again on the next physics step rather than give up.
                    return None
                self._node = rclpy.create_node('pedestrian_rendered_state_publisher')
            self._publisher = self._node.create_publisher(Pedestrians, self._topic, 10)
        except Exception as exc:
            self._disabled = True
            _log_error(
                "[PedestrianRenderedState] could not create the rendered-pedestrian-pose publisher "
                "on %r: %r. The rendered pose will not be observable, so a pedestrian render "
                "regression would again be invisible to the pipeline. Set %s=0 to silence this "
                "deliberately." % (self._topic, exc, PUBLISH_RENDERED_STATES_ENV_VAR))
            return None
        if not self._announced:
            self._announced = True
            _log_warn("[PedestrianRenderedState] publishing rendered pedestrian poses on %r"
                      % self._topic)
        return self._publisher

    def submit(self, name, position, orientation, velocity, walking, sim_time):
        """Record one pedestrian's rendered pose and publish the current snapshot."""
        self._entries[name] = {
            'position': tuple(float(v) for v in position),
            'orientation': tuple(float(v) for v in orientation),
            'velocity': tuple(float(v) for v in velocity),
            'walking': bool(walking),
        }
        publisher = self._ensure_publisher()
        if publisher is None:
            return False
        try:
            publisher.publish(self.build_message(sim_time))
        except Exception as exc:
            self._disabled = True
            _log_error("[PedestrianRenderedState] publish on %r failed: %r; rendered pedestrian "
                       "poses are no longer being published" % (self._topic, exc))
            return False
        return True

    def build_message(self, sim_time=None):
        from arena_people_msgs.msg import Pedestrian, Pedestrians

        msg = Pedestrians()
        msg.header.frame_id = 'map'
        if sim_time is not None:
            msg.header.stamp.sec = int(sim_time)
            msg.header.stamp.nanosec = int(round((float(sim_time) - int(sim_time)) * 1e9))
        for index, name in enumerate(sorted(self._entries)):
            entry = self._entries[name]
            ped = Pedestrian()
            ped.name = str(name)
            ped.id = index
            px, py, pz = entry['position']
            ped.pose.position.x = px
            ped.pose.position.y = py
            ped.pose.position.z = pz
            qx, qy, qz, qw = entry['orientation']
            ped.pose.orientation.x = qx
            ped.pose.orientation.y = qy
            ped.pose.orientation.z = qz
            ped.pose.orientation.w = qw
            vx, vy, vz = entry['velocity']
            ped.twist.linear.x = vx
            ped.twist.linear.y = vy
            ped.twist.linear.z = vz
            ped.animation_state = Pedestrian.WALKING if entry['walking'] else Pedestrian.IDLE
            msg.pedestrians.append(ped)
        return msg


class PedestrianRenderedStateBackend:
    """``Person`` backend that forwards the rendered pose to the shared publisher.

    ``Person`` calls ``initialize(person)`` once and ``update(state, dt)`` from its physics
    callback, which is the same rate ``odom`` is published at.
    """

    def __init__(self, publisher=None):
        self._publisher = publisher
        self._name = None
        self._last_position = None

    def initialize(self, person):
        path = str(getattr(person, 'path', None) or '').rstrip('/')
        self._name = path.split('/')[-1] or 'unknown'
        if self._publisher is None:
            self._publisher = RenderedPedestrianStatePublisher.instance()

    def update(self, state, dt):
        if self._publisher is None or self._name is None:
            return False
        position = tuple(float(v) for v in state.position)
        orientation = tuple(float(v) for v in state.orientation)
        dt = float(dt or 0.0)
        if self._last_position is not None and dt > 0.0:
            velocity = tuple((position[i] - self._last_position[i]) / dt for i in range(3))
        else:
            velocity = (0.0, 0.0, 0.0)
        self._last_position = position
        walking = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2) > 1e-4
        return self._publisher.submit(self._name, position, orientation, velocity, walking,
                                      _sim_time())


def make_backend_if_enabled(environ=None):
    """Return a backend, or ``None`` when publishing is switched off."""
    if not env_flag(PUBLISH_RENDERED_STATES_ENV_VAR, True, environ):
        _log_warn("[PedestrianRenderedState] rendered pedestrian pose publishing DISABLED by %s; "
                  "no artifact will record where the rendered pedestrians actually were"
                  % PUBLISH_RENDERED_STATES_ENV_VAR)
        return None
    return PedestrianRenderedStateBackend()
