"""Enforce that exactly one articulation writer drives the joints a control graph commands.

Why this exists
---------------
A robot USD may ship its own OmniGraph containing an ``IsaacArticulationController``.  Ai2_Bot2 does:
``<robot>/Ai2_Bot2_Chassis/Graphs/differential_controller/ArticulationController`` has its ``execIn``
wired straight to ``OnPlaybackTick.tick``, so it executes on *every* graph tick, and its
``velocityCommand`` comes from an embedded ``DifferentialController`` whose twist subscriber Arena
deliberately disables -- so it commands exactly ``0.0``, forever.

``Articulation.apply_action`` ends in ``set_dof_velocity_targets``, a *persistent* PhysX drive
target.  Arena's own controller writes the real target only on ticks where a ``Twist`` arrived, so
the two writes race inside one ``app.update()`` and whichever lands last governs the following
physics steps.  Measured on the production command path: the embedded controller computed on 84 of
84 graph ticks, the real target survived on 37.8 % of them, and achieved/commanded base speed was
0.291.  With the embedded writer neutralised the target survived 100.000 % of ticks and the ratio
was 0.876 -- a factor of 3.01.  The "3x" in the earlier reports is the reciprocal of that duty
cycle, not a scale factor hiding in a parameter.

Why the check is a *consequence* check and not an intent check
--------------------------------------------------------------
``SpawnUsdRobot._disable_embedded_twist_subscribers`` already tries to fix this, with
``targetPrim.ClearTargets(removeSpec=False)``, and it does not work: measured on the live composed
stage, the composed target list is ``['/World/Robots/Ai2_Bot2']`` before *and after* the call,
because ``ClearTargets`` only drops opinions in the current edit target while the referenced robot
layer keeps its own.  The ``len(targetPrim) == 0`` fail-closed branch inside
``OgnIsaacArticulationController.compute`` is therefore never reached, and the guard prints
"Disabled embedded ArticulationController targets [...]" on every run without having done so.

That is the trap this module is written to avoid: **it verifies against the composed stage and
returns False if it cannot confirm the post-condition.**  It never reports success it has not earned.

Choice of mechanism, by measurement
-----------------------------------
Five candidates were applied to the real robot at the point in the spawn sequence where this runs --
that is, *before* ``timeline.play()`` -- and their *composed* effect read back:

===============================================  ==============================
``targetPrim.ClearTargets(removeSpec=False)``    inert -- composed list unchanged
``targetPrim.SetTargets([])``                    composes
``targetPrim.RemoveTarget(...)``                 composes
``execIn.RemoveConnection(...)``                 composes      <-- used here
``prim.SetActive(False)``                        composes, but see below
===============================================  ==============================

So the guard this replaces did not fail because USD makes the edit hard.  Four of the five candidates
compose, including the one a single token away: ``SetTargets([])``.  It failed because it used the
one that does not.

Removing the ``execIn`` connection is used because it is the only candidate that is both composed
*and* free of side effects:

* Emptying ``targetPrim`` does stop the writes, but ``compute`` then takes its error branch on every
  tick (``db.log_error("No robot prim found for the articulation controller")``) because the node
  retries resolution forever -- roughly 1800 error lines per 300 s episode.
* ``prim.SetActive(False)`` changes stage structure.  Applied after ``timeline.play()`` it destroyed
  the PhysX articulation view outright (``AttributeError: 'Articulation' object has no attribute
  '_physics_view'`` followed by a segfault).  It is not worth the risk for no benefit.
* ``og.Controller.node(...).set_disabled(True)`` **is** the mechanism that stops the writes, and of
  the three that were previously applied together it is the only one that does anything at all:
  measured intact-target fraction ``1.000000`` with it against ``0.0053-0.0263`` without, chassis
  compute count 31-32 per 190 steps falling to ``0``, and the defect returning on revert.  It is
  nevertheless unusable *here*: the USD-embedded graph is not yet instantiated in the OmniGraph
  runtime at this point in the spawn sequence (``og.Controller.node`` raises ``OmniGraphValueError``,
  measured both before and after graph creation), and ``set_disabled`` authors nothing in USD -- so it
  can be neither called now nor pre-set for later.

For completeness, because it is exactly the kind of thing that gets re-suspected: redirecting
``inputs:robotPath`` to a nonexistent prim is inert when applied after the node's first compute,
*even though the attribute demonstrably changes*.  ``compute`` resolves the robot prim exactly once,
behind ``if not state.initialized``, and never consults either input again.  An edit that visibly
applies and changes nothing is the precise failure mode this module exists to detect.

Which writers count as conflicts, and why it is NOT decided by joint names
-------------------------------------------------------------------------
A first version of this module classified a writer as conflicting when its ``inputs:jointNames``
explicitly named one of ours.  **It reported success and neutralised nothing**, because on the real
robot all three controllers -- the embedded chassis one, the ``ROS_JointStates`` one, and Arena's own
-- have ``jointNames`` fed by a *connection* (from a ``ConstructArray`` node), so reading the
attribute yields an empty list for every one of them.  Joint-set intersection is therefore not
computable from USD at this point in the spawn sequence, and any rule resting on it is inert.

The discriminator used instead is the one the original guard already documented and got right:
**a controller that shares its OmniGraph with a non-Arena ``ROS2SubscribeTwist`` is part of an
embedded cmd_vel path and must not write.**  Measured on the real robot, the embedded chassis graph
contains exactly that pairing::

    <robot>/Ai2_Bot2_Chassis/Graphs/differential_controller/
        OnPlaybackTick, DifferentialController, ArticulationController,
        ArrayNames, scale_to_from_stage_units, ros2_context,
        ros2_subscribe_twist, break_3_vector, break_3_vector_01

while ``<robot>/Graph/ROS_JointStates`` contains a joint-state subscriber and no twist subscriber, so
it is correctly left alone -- which matters, because that node commands all DOFs and disabling it was
measured to change nothing (achieved/commanded 0.291 -> 0.290; its ``velocityCommand`` is empty and
``apply_action`` only sets a field when ``np.size(...) > 0``).

Statically authored joint names are still honoured *as an escape*: a co-resident writer whose joint
names are authored and provably disjoint from ours is left alone, so that a robot with two wheel
pairs on one articulation cannot be broken by this.  Every other driven writer is reported and left
untouched, so a new robot's extra writer shows up in the log rather than silently.
"""

from __future__ import annotations

import typing

NODE_TYPE_ATTR = 'node:type'
JOINT_NAMES_ATTR = 'inputs:jointNames'
EXEC_IN_ATTR = 'inputs:execIn'
ARTICULATION_CONTROLLER_NODE_TYPE = 'isaacsim.core.nodes.IsaacArticulationController'
TWIST_SUBSCRIBER_NODE_TYPE = 'isaacsim.ros2.bridge.ROS2SubscribeTwist'

#: Shares a graph with a non-Arena twist subscriber, so it is an embedded cmd_vel writer.
CONFLICTING = 'conflicting'
#: The controller this graph just created, or a sibling in the same graph.  Must stay live.
OWN = 'own'
#: Authored joint names, provably disjoint from ours -- e.g. another wheel pair.  Left alone.
DISJOINT = 'disjoint'
#: Any other articulation writer.  Reported, left alone.
OTHER = 'other'


class ArticulationWriter(typing.NamedTuple):
    path: str
    classification: str
    graph_path: str
    joint_names: tuple[str, ...]
    joint_names_connected: bool
    exec_sources: tuple[str, ...]


def _attr(prim, name):
    attr = prim.GetAttribute(name)
    return attr if (attr and attr.IsValid()) else None


def _node_type(prim) -> str:
    attr = _attr(prim, NODE_TYPE_ATTR)
    if attr is None:
        return ''
    return str(attr.Get() or '')


def _authored_joint_names(prim) -> tuple[tuple[str, ...], bool]:
    """Return ``(joint_names, connected)``.

    Reading a *connected* input returns the attribute's own default rather than the upstream value,
    so an empty tuple is ambiguous between "commands all DOFs" and "resolved upstream".  Conflating
    those two is exactly what made the first version of this module inert, hence the flag.
    """
    attr = _attr(prim, JOINT_NAMES_ATTR)
    if attr is None:
        return (), False
    connected = bool(attr.GetConnections())
    value = attr.Get()
    if value is None:
        return (), connected
    return tuple(str(v) for v in value), connected


def _exec_sources(prim) -> tuple[str, ...]:
    attr = _attr(prim, EXEC_IN_ATTR)
    if attr is None:
        return ()
    return tuple(str(c) for c in attr.GetConnections())


def _graph_path(prim) -> str:
    return str(prim.GetPath().GetParentPath())


def find_articulation_writers(
    stage,
    root_path: str,
    own_node_path: str,
    joint_names: typing.Iterable[str],
) -> list[ArticulationWriter]:
    """Classify every ``IsaacArticulationController`` prim under ``root_path``.

    Pure reads.  Traverses with ``Usd.PrimRange`` including instance proxies, because a robot
    delivered as a reference can compose its graph through one.
    """
    from pxr import Usd

    wanted = {str(j) for j in joint_names}
    root = stage.GetPrimAtPath(root_path)
    if not (root and root.IsValid()):
        return []

    own_graph = str(own_node_path.rsplit('/', 1)[0])
    controllers = []
    twist_graphs: set[str] = set()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        node_type = _node_type(prim)
        if node_type == ARTICULATION_CONTROLLER_NODE_TYPE:
            controllers.append(prim)
        elif node_type == TWIST_SUBSCRIBER_NODE_TYPE:
            twist_graphs.add(_graph_path(prim))

    writers: list[ArticulationWriter] = []
    for prim in controllers:
        path = str(prim.GetPath())
        graph = _graph_path(prim)
        names, connected = _authored_joint_names(prim)
        if path == own_node_path or graph == own_graph:
            classification = OWN
        elif graph not in twist_graphs:
            classification = OTHER
        elif names and not wanted.intersection(names):
            classification = DISJOINT
        else:
            classification = CONFLICTING
        writers.append(ArticulationWriter(
            path=path,
            classification=classification,
            graph_path=graph,
            joint_names=names,
            joint_names_connected=connected,
            exec_sources=_exec_sources(prim),
        ))
    return writers


def _detach_exec_input(prim) -> None:
    """Remove every connection feeding ``inputs:execIn``.

    ``RemoveConnection`` authors a *deletion* list op, which composes over the referenced layer that
    carries the original opinion -- unlike ``ClearTargets(removeSpec=False)``, which does not.
    """
    attr = _attr(prim, EXEC_IN_ATTR)
    if attr is None:
        return
    for source in list(attr.GetConnections()):
        attr.RemoveConnection(source)


def _default_loggers():
    try:  # pragma: no cover - exercised only inside Kit
        import carb
        return carb.log_warn, carb.log_error
    except Exception:  # noqa: BLE001
        import logging
        log = logging.getLogger(__name__)
        return log.warning, log.error


def enforce_exclusive_articulation_writer(
    stage,
    root_path: str,
    own_node_path: str,
    joint_names: typing.Iterable[str],
    log_warn: typing.Callable[[str], typing.Any] | None = None,
    log_error: typing.Callable[[str], typing.Any] | None = None,
) -> bool:
    """Make ``own_node_path`` the only writer of ``joint_names`` under ``root_path``.

    Returns ``True`` only when the post-condition has been **verified on the composed stage**:
    every conflicting writer has no remaining ``execIn`` connection, and our own controller still
    has one.  Returns ``False``, having logged an error, in every other case -- including when there
    was nothing to do but our own controller looks wrong, because that means this function's
    assumptions no longer hold.
    """
    if log_warn is None or log_error is None:
        default_warn, default_error = _default_loggers()
        log_warn = log_warn or default_warn
        log_error = log_error or default_error

    joint_names = tuple(str(j) for j in joint_names)
    try:
        writers = find_articulation_writers(stage, root_path, own_node_path, joint_names)
    except Exception as exc:  # noqa: BLE001
        log_error(
            '[ExclusiveArticulationWriter] Could not enumerate articulation writers under '
            f'{root_path}: {exc!r}'
        )
        return False

    own = [w for w in writers if w.classification == OWN and w.path == own_node_path]
    if len(own) != 1:
        log_error(
            '[ExclusiveArticulationWriter] Expected exactly one own articulation controller at '
            f'{own_node_path}, found {len(own)}. Refusing to claim exclusivity.'
        )
        return False

    for writer in writers:
        if writer.classification == OTHER and writer.exec_sources:
            log_warn(
                '[ExclusiveArticulationWriter] Articulation controller '
                f'{writer.path} is driven but is not part of an embedded cmd_vel graph '
                f'(graph {writer.graph_path} has no ROS2SubscribeTwist). Left untouched '
                'deliberately: the one such node on Ai2_Bot2 commands all DOFs yet was measured '
                'harmless (achieved/commanded 0.291 -> 0.290 when disabled) because its '
                'velocityCommand is empty. Report this if a robot regresses.'
            )
        elif writer.classification == DISJOINT:
            log_warn(
                '[ExclusiveArticulationWriter] Articulation controller '
                f'{writer.path} commands {writer.joint_names}, disjoint from {joint_names}. '
                'Left untouched.'
            )

    conflicts = [w for w in writers if w.classification == CONFLICTING]
    for writer in conflicts:
        prim = stage.GetPrimAtPath(writer.path)
        if not (prim and prim.IsValid()):
            log_error(
                '[ExclusiveArticulationWriter] Conflicting articulation controller '
                f'{writer.path} vanished between enumeration and neutralisation.'
            )
            return False
        try:
            _detach_exec_input(prim)
        except Exception as exc:  # noqa: BLE001
            log_error(
                '[ExclusiveArticulationWriter] Failed to detach execIn on conflicting '
                f'articulation controller {writer.path}: {exc!r}'
            )
            return False

    # ------------------------------------------------------------------
    # Verification, on the COMPOSED stage.  This is the whole point: the guard this replaces
    # failed because it checked its own edit target instead of the composed result.
    # ------------------------------------------------------------------
    try:
        after = find_articulation_writers(stage, root_path, own_node_path, joint_names)
    except Exception as exc:  # noqa: BLE001
        log_error(
            '[ExclusiveArticulationWriter] Could not re-enumerate articulation writers for '
            f'verification under {root_path}: {exc!r}'
        )
        return False

    still_live = [w for w in after if w.classification == CONFLICTING and w.exec_sources]
    if still_live:
        log_error(
            '[ExclusiveArticulationWriter] VERIFICATION FAILED: '
            f'{len(still_live)} conflicting articulation controller(s) still driven after '
            'neutralisation: '
            + '; '.join(f'{w.path} <- {w.exec_sources}' for w in still_live)
            + '. Their velocity-target writes will race this graph on every tick.'
        )
        return False

    own_after = [w for w in after if w.path == own_node_path]
    if len(own_after) != 1 or not own_after[0].exec_sources:
        log_error(
            "[ExclusiveArticulationWriter] VERIFICATION FAILED: this graph's own articulation "
            f'controller {own_node_path} is not driven after neutralisation '
            f'(matches={len(own_after)}). Refusing to report success.'
        )
        return False

    # Always state the census, including when nothing was neutralised.  An earlier version of this
    # function found zero conflicts on the real robot and returned True in silence -- the same shape
    # of failure as the guard it replaces.  A visible "conflicts=0" next to "other driven=N" is what
    # lets a reader notice that the classifier has stopped recognising a writer it should.
    log_warn(
        '[ExclusiveArticulationWriter] census under {root}: own=1 driven={own_driven} '
        'conflicting={n_conf} disjoint={n_disj} other={n_other} (of which driven={n_other_driven}) '
        'joints={joints}'.format(
            root=root_path,
            own_driven=bool(own_after[0].exec_sources),
            n_conf=len(conflicts),
            n_disj=sum(1 for w in after if w.classification == DISJOINT),
            n_other=sum(1 for w in after if w.classification == OTHER),
            n_other_driven=sum(1 for w in after if w.classification == OTHER and w.exec_sources),
            joints=list(joint_names),
        )
    )

    if conflicts:
        log_warn(
            '[ExclusiveArticulationWriter] Neutralised '
            f'{len(conflicts)} duplicate articulation writer(s) on {list(joint_names)} and '
            'VERIFIED on the composed stage that they are no longer driven: '
            + '; '.join(w.path for w in conflicts)
        )
    return True
