"""Episode-scoped readiness gate for Isaac scene initialization.

The task reset finishes while Isaac is paused.  Once simulation resumes, newly
spawned people still need real physics/animation ticks before their animation
graphs leave the bind pose.  This coordinator publishes one ready event only
after every current character graph exists and remains live for a short amount
of simulation time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class SceneReadinessCoordinator:
    def __init__(
        self,
        *,
        people: Callable[[], Iterable[Any]],
        publish: Callable[[int], None],
        settle_sim_seconds: float = 0.75,
    ):
        self._people = people
        self._publish = publish
        self._settle_sim_seconds = float(settle_sim_seconds)
        self._episode: int | None = None
        self._graphs_ready_since: float | None = None
        self._published_episode: int | None = None

    def reset_completed(self, episode: int) -> None:
        self._episode = int(episode)
        self._graphs_ready_since = None
        self._published_episode = None
        self._publish(-1)

    def update(self, sim_time: float) -> bool:
        """Publish the current episode once its scene has settled.

        This must be called after a real ``world.step``.  Render-only updates or
        Replicator steps with ``delta_time=0`` intentionally do not advance this
        gate.
        """
        if self._episode is None or self._published_episode == self._episode:
            return False

        current_people = tuple(self._people())
        graphs_ready = all(
            bool(getattr(person, "animation_ready", False))
            for person in current_people
        )
        if not graphs_ready:
            self._graphs_ready_since = None
            return False

        now = float(sim_time)
        if self._graphs_ready_since is None:
            self._graphs_ready_since = now
            return False
        if now - self._graphs_ready_since < self._settle_sim_seconds:
            return False

        self._published_episode = self._episode
        self._publish(self._episode)
        return True
