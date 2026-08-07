"""Regression tests for the duplicate-articulation-writer defect.

The defect: the robot USD ships its own ``IsaacArticulationController`` wired to ``OnPlaybackTick``,
writing a velocity target of ``0.0`` to the drive joints on every graph tick and racing Arena's
controller.  ``SpawnUsdRobot`` contained a mitigation whose remedy was
``targetPrim.ClearTargets(removeSpec=False)``, which is **inert** -- it drops opinions only in the
current edit target while the referenced robot layer keeps its own -- so the mitigation logged
success on every run without ever disabling anything.

Two properties of the real asset are load-bearing, and both were measured on it rather than assumed.
A fixture missing either one passes against defective code and proves nothing:

1. **The embedded controller's opinions arrive through a REFERENCE.**  That composition arc is the
   entire reason ``ClearTargets(removeSpec=False)`` fails.
2. **``inputs:jointNames`` is fed by a CONNECTION, not authored.**  On the real robot all three
   controllers read back an empty joint list.  A first version of the fix classified conflicts by
   joint-name intersection, passed a suite whose fixture authored joint names literally, and then
   neutralised nothing on the real robot while reporting success.  Every fixture here therefore
   connects ``jointNames`` the way ``ConstructArray`` does, and
   ``test_conflict_is_detected_when_joint_names_are_connected`` exists specifically to fail if that
   rule is ever reintroduced.

Power is proven in both directions:

* ``test_old_remedy_is_inert_under_composition`` / ``test_old_remedy_leaves_the_writer_driven`` pin
  the defect's real behaviour and fail if the old mechanism is reinstated as a fix;
* ``test_post_condition_rejects_the_old_remedy`` runs the new enforcement with its neutralisation
  step replaced by the mechanism that shipped, and requires ``False`` -- the functional
  demonstration that the new post-condition catches what production did;
* the three ``test_spawn_usd_robot_*`` source contracts fail against the pre-fix
  ``SpawnUsdRobot.py`` (measured: 3 failed / 11 passed) and pass after.

``SpawnUsdRobot.py`` cannot be imported without Kit, so those parse it with ``ast``, in the style of
``test_pedestrian_wiring_contract.py``.
"""

import ast
import os

import pytest

pytest.importorskip('pxr', reason='these tests need USD, which ships with Isaac/Kit')

# Imported after importorskip on purpose, so a machine without USD skips instead of erroring.
# That inverts the usual grouping, hence the noqa.
from pxr import Sdf, Usd  # noqa: E402,I100

from isaac_utils.utils.articulation_writers import (  # noqa: E402,I100
    ARTICULATION_CONTROLLER_NODE_TYPE,
    CONFLICTING,
    DISJOINT,
    OTHER,
    OWN,
    TWIST_SUBSCRIBER_NODE_TYPE,
    enforce_exclusive_articulation_writer,
    find_articulation_writers,
)

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPAWN_USD_ROBOT = os.path.join(_PKG_ROOT, 'arena_isaac', 'services', 'SpawnUsdRobot.py')

ROBOT = '/World/Robots/Bot'
OWN_GRAPH = ROBOT + '/arena_diff_drive'
OWN_NODE = OWN_GRAPH + '/articulation_controller'
CHASSIS_GRAPH = ROBOT + '/Chassis/Graphs/differential_controller'
EMBEDDED = CHASSIS_GRAPH + '/ArticulationController'
EMBEDDED_TICK = CHASSIS_GRAPH + '/OnPlaybackTick.outputs:tick'
JOINTSTATE_GRAPH = ROBOT + '/Graph/ROS_JointStates'
JOINTSTATE = JOINTSTATE_GRAPH + '/ArticulationController'
DRIVE_JOINTS = ['driving_left_joint', 'driving_right_joint']


def _node(stage, path, node_type):
    prim = stage.DefinePrim(path, 'OmniGraphNode')
    prim.CreateAttribute('node:type', Sdf.ValueTypeNames.Token).Set(node_type)
    return prim


def _controller(stage, path, exec_source=None, target=None,
                joint_names_connected_from=None, joint_names_authored=None):
    """Build an IsaacArticulationController node.

    ``joint_names_connected_from`` reproduces the real asset: ``inputs:jointNames`` is *connected* to
    a ``ConstructArray`` output, so reading the attribute yields an empty list.
    """
    prim = _node(stage, path, ARTICULATION_CONTROLLER_NODE_TYPE)
    jn = prim.CreateAttribute('inputs:jointNames', Sdf.ValueTypeNames.TokenArray)
    if joint_names_connected_from is not None:
        jn.SetConnections([Sdf.Path(joint_names_connected_from)])
    elif joint_names_authored is not None:
        jn.Set([str(j) for j in joint_names_authored])
    exec_attr = prim.CreateAttribute('inputs:execIn', Sdf.ValueTypeNames.Token)
    if exec_source is not None:
        exec_attr.SetConnections([Sdf.Path(exec_source)])
    if target is not None:
        prim.CreateRelationship('inputs:targetPrim').SetTargets([Sdf.Path(target)])
    return prim


@pytest.fixture()
def referenced_robot_stage(tmp_path):
    """A stage reproducing the two measured properties of the real robot.

    The fixture asserts both, so a broken fixture cannot masquerade as a passing test.
    """
    robot_layer = tmp_path / 'bot_description.usda'
    src = Usd.Stage.CreateNew(str(robot_layer))
    src.DefinePrim('/Bot', 'Xform')
    g = '/Bot/Chassis/Graphs/differential_controller'
    _node(src, g + '/OnPlaybackTick', 'omni.graph.action.OnPlaybackTick')
    _node(src, g + '/ArrayNames', 'omni.graph.nodes.ConstructArray')
    _node(src, g + '/ros2_subscribe_twist', TWIST_SUBSCRIBER_NODE_TYPE)
    _controller(src, g + '/ArticulationController',
                exec_source=g + '/OnPlaybackTick.outputs:tick',
                target='/Bot',
                joint_names_connected_from=g + '/ArrayNames.outputs:array')
    # A second embedded graph with no twist subscriber -- the real robot's ROS_JointStates.
    j = '/Bot/Graph/ROS_JointStates'
    _node(src, j + '/OnPlaybackTick', 'omni.graph.action.OnPlaybackTick')
    _node(src, j + '/SubscribeJointState', 'isaacsim.ros2.bridge.ROS2SubscribeJointState')
    _controller(src, j + '/ArticulationController',
                exec_source=j + '/OnPlaybackTick.outputs:tick',
                target='/Bot',
                joint_names_connected_from=j + '/SubscribeJointState.outputs:jointNames')
    src.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim(ROBOT, 'Xform').GetReferences().AddReference(str(robot_layer), '/Bot')

    embedded = stage.GetPrimAtPath(EMBEDDED)
    assert embedded and embedded.IsValid(), 'fixture broken: reference did not resolve'
    rel = embedded.GetRelationship('inputs:targetPrim')
    assert [str(t) for t in rel.GetForwardedTargets()] == [ROBOT], (
        'fixture broken: embedded targetPrim did not compose through the reference'
    )
    assert embedded.GetAttribute('inputs:execIn').GetConnections(), (
        'fixture broken: embedded execIn is not driven, so there is no defect to test'
    )
    jn = embedded.GetAttribute('inputs:jointNames')
    assert jn.GetConnections() and not jn.Get(), (
        'fixture broken: jointNames must be CONNECTED and read back empty, as measured on the real '
        'robot -- otherwise a joint-name-intersection rule would appear to work'
    )

    # Arena's own graph, as differential() creates it: jointNames connected via ConstructArray.
    _node(stage, OWN_GRAPH + '/ros2_subscribe_twist', TWIST_SUBSCRIBER_NODE_TYPE)
    _node(stage, OWN_GRAPH + '/make_array', 'omni.graph.nodes.ConstructArray')
    _controller(stage, OWN_NODE,
                exec_source=OWN_GRAPH + '/ros2_subscribe_twist.outputs:execOut',
                joint_names_connected_from=OWN_GRAPH + '/make_array.outputs:array')
    return stage


def _composed_targets(stage, path):
    rel = stage.GetPrimAtPath(path).GetRelationship('inputs:targetPrim')
    return [str(t) for t in rel.GetForwardedTargets()]


def _exec_connections(stage, path):
    attr = stage.GetPrimAtPath(path).GetAttribute('inputs:execIn')
    return [str(c) for c in attr.GetConnections()]


def _neutralise_with_cleartargets(stage, path):
    """Exactly what shipped in `_disable_embedded_twist_subscribers` before the fix."""
    stage.GetPrimAtPath(path).GetRelationship('inputs:targetPrim').ClearTargets(False)


def _enforce(stage, own=OWN_NODE, joints=DRIVE_JOINTS):
    errors, warnings = [], []
    result = enforce_exclusive_articulation_writer(
        stage=stage, root_path=ROBOT, own_node_path=own, joint_names=joints,
        log_warn=warnings.append, log_error=errors.append,
    )
    return result, errors, warnings


# ---------------------------------------------------------------------------
# The defect itself, locked down.
# ---------------------------------------------------------------------------
def test_old_remedy_is_inert_under_composition(referenced_robot_stage):
    stage = referenced_robot_stage
    assert _composed_targets(stage, EMBEDDED) == [ROBOT]
    _neutralise_with_cleartargets(stage, EMBEDDED)
    assert _composed_targets(stage, EMBEDDED) == [ROBOT], (
        'ClearTargets(removeSpec=False) unexpectedly emptied a referenced relationship. If USD '
        'behaviour changed, the fix is still correct but its rationale needs rewriting.'
    )


def test_setting_an_explicit_empty_list_would_have_composed(referenced_robot_stage):
    """The remedy one token away from the one that shipped does work.

    Recorded so nobody concludes the old code failed because USD made it impossible.
    """
    stage = referenced_robot_stage
    stage.GetPrimAtPath(EMBEDDED).GetRelationship('inputs:targetPrim').SetTargets([])
    assert _composed_targets(stage, EMBEDDED) == []


def test_old_remedy_leaves_the_writer_driven(referenced_robot_stage):
    stage = referenced_robot_stage
    _neutralise_with_cleartargets(stage, EMBEDDED)
    assert _exec_connections(stage, EMBEDDED), (
        'the embedded controller must still be pulsed after the old remedy -- that is the defect'
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_conflict_is_detected_when_joint_names_are_connected(referenced_robot_stage):
    """The regression that the first version of the fix shipped and this suite missed."""
    by_path = {w.path: w for w in
               find_articulation_writers(referenced_robot_stage, ROBOT, OWN_NODE, DRIVE_JOINTS)}
    assert by_path[EMBEDDED].joint_names == (), 'fixture no longer reproduces connected jointNames'
    assert by_path[EMBEDDED].joint_names_connected is True
    assert by_path[EMBEDDED].classification == CONFLICTING, (
        'a writer sharing a graph with an embedded twist subscriber is a conflict even though its '
        'joint names cannot be read; classifying by joint-name intersection makes the fix inert'
    )


def test_own_and_unrelated_are_not_conflicts(referenced_robot_stage):
    by_path = {w.path: w for w in
               find_articulation_writers(referenced_robot_stage, ROBOT, OWN_NODE, DRIVE_JOINTS)}
    assert by_path[OWN_NODE].classification == OWN
    assert by_path[JOINTSTATE].classification == OTHER, (
        'the joint-state graph has no twist subscriber, so it is not part of cmd_vel handling'
    )


def test_authored_disjoint_joints_are_left_alone(referenced_robot_stage):
    """A second wheel pair on the same articulation must survive."""
    stage = referenced_robot_stage
    other = CHASSIS_GRAPH + '/SecondPairArticulationController'
    _controller(stage, other, exec_source=EMBEDDED_TICK,
                joint_names_authored=['rear_left_joint', 'rear_right_joint'])
    by_path = {w.path: w for w in find_articulation_writers(stage, ROBOT, OWN_NODE, DRIVE_JOINTS)}
    assert by_path[other].classification == DISJOINT


# ---------------------------------------------------------------------------
# The fix
# ---------------------------------------------------------------------------
def test_enforce_neutralises_and_verifies(referenced_robot_stage):
    stage = referenced_robot_stage
    result, errors, warnings = _enforce(stage)
    assert result is True
    assert errors == []
    assert _exec_connections(stage, EMBEDDED) == [], 'conflicting writer is still pulsed'
    assert _exec_connections(stage, OWN_NODE), 'the fix disabled our own controller'
    assert any('Neutralised' in w and 'VERIFIED' in w for w in warnings)


def test_enforce_leaves_unrelated_and_disjoint_writers_driven(referenced_robot_stage):
    stage = referenced_robot_stage
    other = CHASSIS_GRAPH + '/SecondPairArticulationController'
    _controller(stage, other, exec_source=EMBEDDED_TICK,
                joint_names_authored=['rear_left_joint', 'rear_right_joint'])
    result, errors, warnings = _enforce(stage)
    assert result is True
    assert errors == []
    assert _exec_connections(stage, other), 'a disjoint wheel pair was wrongly neutralised'
    assert _exec_connections(stage, JOINTSTATE), 'an unrelated writer was wrongly neutralised'
    assert any('not part of an embedded cmd_vel graph' in w for w in warnings)


def test_enforce_always_logs_a_census(referenced_robot_stage):
    """A silent zero-conflict pass is how the first version of this fix hid its own inertness."""
    _, _, warnings = _enforce(referenced_robot_stage)
    census = [w for w in warnings if 'census under' in w]
    assert len(census) == 1, warnings
    assert 'conflicting=1' in census[0], census[0]
    assert 'other=1' in census[0], census[0]


def test_enforce_is_idempotent(referenced_robot_stage):
    stage = referenced_robot_stage
    assert _enforce(stage)[0] is True
    assert _enforce(stage)[0] is True
    assert _exec_connections(stage, EMBEDDED) == []


def test_no_embedded_writer_is_a_clean_pass():
    """The generic URDF-imported path: nothing to neutralise, and it must not complain."""
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim(ROBOT, 'Xform')
    _node(stage, OWN_GRAPH + '/ros2_subscribe_twist', TWIST_SUBSCRIBER_NODE_TYPE)
    _controller(stage, OWN_NODE, exec_source=OWN_GRAPH + '/ros2_subscribe_twist.outputs:execOut')
    result, errors, _ = _enforce(stage)
    assert result is True
    assert errors == []


# ---------------------------------------------------------------------------
# Loudness on failure -- the property the old code lacked.
# ---------------------------------------------------------------------------
def test_post_condition_rejects_the_old_remedy(referenced_robot_stage, monkeypatch):
    """THE functional both-ways proof.

    Run the new enforcement with its neutralisation step replaced by the mechanism that shipped.
    It must return False and log an error, instead of reporting success as production did.
    """
    import isaac_utils.utils.articulation_writers as mod
    monkeypatch.setattr(
        mod, '_detach_exec_input',
        lambda prim: prim.GetRelationship('inputs:targetPrim').ClearTargets(False),
    )
    result, errors, warnings = _enforce(referenced_robot_stage)
    assert result is False
    assert any('VERIFICATION FAILED' in e for e in errors), errors
    assert not any('Neutralised' in w for w in warnings), (
        'a failed disable must not also emit a success message'
    )


def test_missing_own_controller_is_refused(referenced_robot_stage):
    result, errors, _ = _enforce(referenced_robot_stage, own=ROBOT + '/not_a_node')
    assert result is False
    assert any('Expected exactly one own articulation controller' in e for e in errors), errors


def test_own_controller_left_undriven_is_refused(referenced_robot_stage):
    stage = referenced_robot_stage
    stage.GetPrimAtPath(OWN_NODE).GetAttribute('inputs:execIn').SetConnections([])
    result, errors, _ = _enforce(stage)
    assert result is False
    assert any('is not driven after neutralisation' in e for e in errors), errors


# ---------------------------------------------------------------------------
# Source-level contract on the caller.  Fails against the pre-fix file.
# ---------------------------------------------------------------------------
def _spawn_source():
    with open(_SPAWN_USD_ROBOT, 'r', encoding='utf-8') as handle:
        return handle.read()


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError('function %s not found' % name)


def test_spawn_usd_robot_calls_the_enforcement():
    setup = _function(ast.parse(_spawn_source(), filename=_SPAWN_USD_ROBOT),
                      '_setup_ai2_bot2_control_graph')
    called = {n.func.id for n in ast.walk(setup)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert 'enforce_exclusive_articulation_writer' in called, (
        'the Ai2_Bot2 control graph setup must verify exclusive articulation control; without it '
        'the embedded chassis controller zeroes the drive target on every tick'
    )


def test_spawn_usd_robot_has_no_cleartargets_disable_step():
    """`_remap_articulation_target` still uses ClearTargets legitimately, so scope this to the guard."""
    guard = _function(ast.parse(_spawn_source(), filename=_SPAWN_USD_ROBOT),
                      '_disable_embedded_twist_subscribers')
    attrs = {n.attr for n in ast.walk(guard) if isinstance(n, ast.Attribute)}
    assert 'ClearTargets' not in attrs, (
        'ClearTargets(removeSpec=False) cannot empty a relationship whose opinion comes from the '
        'referenced robot layer; it made this guard log success without disabling anything'
    )


def test_spawn_usd_robot_does_not_claim_success_before_verifying():
    """The success log must not be reachable without the verification having passed."""
    setup = _function(ast.parse(_spawn_source(), filename=_SPAWN_USD_ROBOT),
                      '_setup_ai2_bot2_control_graph')
    body = setup.body

    def _index(predicate, what):
        for i, stmt in enumerate(body):
            if any(predicate(n) for n in ast.walk(stmt)):
                return i
        raise AssertionError(
            '%s not found in _setup_ai2_bot2_control_graph; the success message must be gated on '
            'a verified disable' % what
        )

    enforce_idx = _index(
        lambda n: isinstance(n, ast.Name) and n.id == 'enforce_exclusive_articulation_writer',
        'the exclusive-articulation-control verification',
    )
    success_idx = _index(
        lambda n: (isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and 'Created Ai2_Bot2 Arena diff-drive graph' in n.value),
        'the "Created ... diff-drive graph" success message',
    )
    assert enforce_idx < success_idx, (
        'the "Created ... diff-drive graph" message must be emitted only after exclusive '
        'articulation control has been verified'
    )
