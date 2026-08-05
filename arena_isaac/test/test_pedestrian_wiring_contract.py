"""Source-level contract checks for wiring that cannot be executed without Isaac.

``person.py`` imports ``omni.anim.graph.core`` at module scope, so it can only be *executed* inside
a running Kit session.  These tests therefore parse it with ``ast`` -- no import, no stubs -- and
assert the structural facts that a future edit could silently undo:

* ``Person.update`` still drives the animation clock, and still writes the ``PathPoints`` /
  ``Action`` / ``Walk`` variables rather than a pose;
* ``SpawnPedestrians`` still attaches the rendered-state backend, or the rendered pose becomes
  unobservable again with no error;
* nothing in the pedestrian path writes a transform, which
  ``docs/benchmark/troubleshooting.md:238-240`` forbids.
"""

import ast
import os

import pytest

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERSON = os.path.join(_PKG_ROOT, "pedestrian", "simulator", "logic", "people", "person.py")
_SPAWN = os.path.join(_PKG_ROOT, "arena_isaac", "services", "SpawnPedestrians.py")
_NAVIGATE = os.path.join(_PKG_ROOT, "arena_isaac", "services", "NavigatePedestrians.py")

FORBIDDEN_POSE_WRITES = ("set_world_poses", "set_local_poses", "set_world_transform")


def _tree(path):
    with open(path, "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def _class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError("class %s not found" % name)


def _function(scope, name):
    for node in scope.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError("function %s not found" % name)


def _attribute_calls(node):
    """Every ``obj.attr(...)`` call name inside ``node``."""
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


def _string_constants(node):
    return [c.value for c in ast.walk(node) if isinstance(c, ast.Constant)
            and isinstance(c.value, str)]


@pytest.fixture(scope="module")
def person_class():
    return _class(_tree(_PERSON), "Person")


def test_person_update_drives_the_animation_clock(person_class):
    update = _function(person_class, "update")
    assert "_tick_character_animation" in _attribute_calls(update), (
        "Person.update must advance the animation graph by the elapsed physics dt; without it the "
        "graph is only evaluated on rendered steps and pedestrians render at ~1/"
        "render_every_n_steps of their commanded speed")


def test_person_still_drives_the_animation_graph_variables(person_class):
    """The mandated transport (troubleshooting.md:230-233) must stay in place."""
    update = _function(person_class, "update")
    written = set(_string_constants(update))
    for variable in ("PathPoints", "Action", "Walk"):
        assert variable in written, "Person.update must still write %r" % variable


def test_person_tick_helper_exists_and_reports_failure(person_class):
    tick = _function(person_class, "_tick_character_animation")
    calls = _attribute_calls(tick)
    assert "log_error" in calls, (
        "an unavailable or throwing Character.update(dt) must be reported loudly; "
        "Character.update() returns None and does not raise when the timeline is stopped, so "
        "silence here is exactly how the original defect survived")
    assert "record" in calls, "progress must be measured, because the tick fails silently"


def test_person_never_writes_a_pose():
    """troubleshooting.md:238-240 forbids direct-pose workarounds in the pedestrian path."""
    for path in (_PERSON, _NAVIGATE, _SPAWN):
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        for forbidden in FORBIDDEN_POSE_WRITES:
            assert forbidden + "(" not in source, (
                "%s calls %s, which the pedestrian path must not do" % (path, forbidden))


def test_person_state_position_is_only_assigned_by_update_state(person_class):
    """``person.state.position = ...`` from the replay path is forbidden; the readback is not."""
    writers = set()
    for function in person_class.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in ("position",
                                                                         "orientation"):
                    writers.add(function.name)
    assert writers <= {"__init__", "update_state"}, (
        "only the spawn-time initialisation and the animation-graph readback may set the state; "
        "found %s" % sorted(writers))


def test_animation_graph_acquisition_is_throttled_and_loud(person_class):
    acquire = _function(person_class, "_acquire_character_graph")
    calls = _attribute_calls(acquire)
    assert "next_action" in calls, "USD authoring must be throttled, not run on every physics step"
    assert "log_error" in calls, "a permanent acquisition failure must not be silent"
    assert "success_message" in calls, "a successful acquisition must leave a marker in the log"


def test_person_init_prepares_animation_state_before_registering_callbacks(person_class):
    """Both physics callbacks reach for ``character_graph`` on their first invocation."""
    init = _function(person_class, "__init__")
    graph_assigned_at = None
    callback_registered_at = None
    for node in ast.walk(init):
        if (isinstance(node, ast.Assign) and graph_assigned_at is None
                and any(isinstance(t, ast.Attribute) and t.attr == "_character_graph"
                        for t in node.targets)):
            graph_assigned_at = node.lineno
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_physics_callback" and callback_registered_at is None):
            callback_registered_at = node.lineno
    assert graph_assigned_at is not None and callback_registered_at is not None
    assert graph_assigned_at < callback_registered_at, (
        "_character_graph must exist before a physics callback can run (was line %s vs %s)"
        % (graph_assigned_at, callback_registered_at))


def test_spawn_pedestrians_attaches_the_rendered_state_backend():
    tree = _tree(_SPAWN)
    spawn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "spawn_pedestrian":
            spawn = node
    assert spawn is not None
    assert "make_backend_if_enabled" in [
        c.func.id for c in ast.walk(spawn)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)], (
        "without a backend the rendered pedestrian pose is read every physics step and thrown away")

    person_calls = [c for c in ast.walk(spawn)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id == "Person"]
    assert person_calls, "spawn_pedestrian must construct Person"
    for call in person_calls:
        assert "backend" in [kw.arg for kw in call.keywords], (
            "every Person construction must pass backend= (there are two branches)")


def test_navigate_pedestrians_uses_the_shared_planner():
    tree = _tree(_NAVIGATE)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
    assert "plan_pedestrian_target" in imported
    source = open(_NAVIGATE, "r", encoding="utf-8").read()
    assert "dist < 0.01" not in source, "the 1 cm dead band must be gone"
