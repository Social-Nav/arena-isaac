"""Integration test for the real ``navigate_pedestrian`` service handler.

``arena_isaac/services/NavigatePedestrians.py`` cannot be imported in the evaluator container as-is:
its package ``__init__`` pulls in every service, and those need ``carb`` / ``omni.*``, which exist
only inside the Isaac container.  This module therefore

* installs minimal stand-ins for ``carb``, ``omni.kit.commands`` and the ``Person`` module, and
* imports ``NavigatePedestrians.py`` through a synthetic package so its relative ``from .utils
  import ...`` still resolves to the real ``services/utils.py``.

``isaacsim_msgs``, ``rclpy`` and the pedestrian-targeting module are the **real** ones, so what is
under test is the production handler and the production planner, not a re-implementation.

The assertions are behavioural, not import smoke: they check the waypoint the handler hands to
``Person.update_target_positions`` and whether it clears the queue instead.
"""

import importlib.util
import math
import os
import sys
import types

import pytest

_SERVICES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arena_isaac", "services")
_PKG = "_lane_r_services_under_test"


def _install_stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _StubPerson:
    """Stands in for ``pedestrian...person.Person``: only what the handler touches."""

    def __init__(self, position):
        self.state = types.SimpleNamespace(position=list(position))
        self._target_positions = []
        self.commanded = []

    def update_target_positions(self, positions, walk_speed=1.0):
        self._target_positions.extend(positions)
        self.commanded.append((list(positions), walk_speed))


@pytest.fixture(scope="module")
def navigate_module():
    warnings = []

    _install_stub("carb",
                  log_warn=lambda message: warnings.append(("warn", message)),
                  log_error=lambda message: warnings.append(("error", message)))
    omni = _install_stub("omni")
    omni_kit = _install_stub("omni.kit")
    _install_stub("omni.kit.commands", execute=lambda *a, **k: (True, None))
    omni.kit = omni_kit
    omni_kit.commands = sys.modules["omni.kit.commands"]

    # Import the REAL pedestrian packages first, then replace only the one leaf module that needs
    # Isaac at import time.  Stubbing the top-level `pedestrian` name instead would shadow the real
    # package and make every submodule unimportable.
    import pedestrian.simulator.logic.people as people_pkg

    person_module = _install_stub("pedestrian.simulator.logic.people.person", Person=_StubPerson)
    people_pkg.person = person_module

    package = types.ModuleType(_PKG)
    package.__path__ = [_SERVICES_DIR]
    sys.modules[_PKG] = package

    spec = importlib.util.spec_from_file_location(
        _PKG + ".NavigatePedestrians", os.path.join(_SERVICES_DIR, "NavigatePedestrians.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.Person is _StubPerson, "the stub must be what the handler resolved"
    module._lane_r_warnings = warnings
    module._lane_r_person_module = person_module
    return module


def _goal(navigate_module, name, position, velocity):
    goal = navigate_module.PedestrianGoal()
    goal.name = name
    goal.position.x, goal.position.y, goal.position.z = position
    goal.velocity = float(velocity)
    return goal


def _patch_resolver(navigate_module, monkeypatch, person):
    monkeypatch.setattr(navigate_module, "_resolve_person",
                        lambda goal_name: ("/World/Pedestrians/hunav_1", person))


def test_handler_idles_the_character_inside_the_dead_band(navigate_module, monkeypatch):
    person = _StubPerson((0.0, 0.0, 0.0))
    person._target_positions.append([9.0, 9.0, 0.0])  # a stale waypoint
    _patch_resolver(navigate_module, monkeypatch, person)

    assert navigate_module.navigate_pedestrian(
        _goal(navigate_module, "Pedestrians/hunav_1", (0.02, 0.0, 0.0), 0.0)) is True
    assert person._target_positions == [], "the stale waypoint must be cleared"
    assert person.commanded == [], "no walk may be commanded from a 2 cm residual"


def test_handler_walks_exactly_to_the_logical_pose_for_a_mid_range_residual(navigate_module,
                                                                           monkeypatch):
    person = _StubPerson((0.0, 0.0, 0.0))
    _patch_resolver(navigate_module, monkeypatch, person)

    assert navigate_module.navigate_pedestrian(
        _goal(navigate_module, "Pedestrians/hunav_1", (1.0, 0.0, 0.0), 0.6)) is True
    assert len(person.commanded) == 1
    positions, walk_speed = person.commanded[0]
    assert walk_speed == pytest.approx(0.6)
    target = positions[0]
    assert math.dist(target[:2], (1.0, 0.0)) == pytest.approx(0.0, abs=1e-9), (
        "the target must be the logical pose, not 2 m past it; got %r" % (target,))


def test_handler_keeps_the_full_look_ahead_for_a_distant_logical_pose(navigate_module, monkeypatch):
    person = _StubPerson((0.0, 0.0, 0.0))
    _patch_resolver(navigate_module, monkeypatch, person)

    assert navigate_module.navigate_pedestrian(
        _goal(navigate_module, "Pedestrians/hunav_1", (12.63, 0.0, 0.0), 0.9)) is True
    positions, walk_speed = person.commanded[0]
    assert walk_speed == pytest.approx(0.9)
    assert math.dist(positions[0][:2], (0.0, 0.0)) == pytest.approx(navigate_module._LOOK_AHEAD_M)


def test_handler_uses_the_fixed_dead_band_constant(navigate_module):
    from pedestrian.simulator.logic.people.pedestrian_targeting import ARRIVAL_DEAD_BAND_M

    assert navigate_module._ARRIVAL_DEAD_BAND_M == ARRIVAL_DEAD_BAND_M
    assert navigate_module._ARRIVAL_DEAD_BAND_M == pytest.approx(0.25)


def test_handler_reports_false_when_the_person_is_missing(navigate_module, monkeypatch):
    monkeypatch.setattr(navigate_module, "_resolve_person", lambda name: ("/missing", None))
    assert navigate_module.navigate_pedestrian(
        _goal(navigate_module, "Pedestrians/nope", (1.0, 0.0, 0.0), 0.5)) is False
