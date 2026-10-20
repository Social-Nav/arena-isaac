# Odometry ownership contract

`/<task namespace>/<robot>/odom` has exactly one publisher backend for the
life of an Isaac process.  The selection is frozen before the first robot is
spawned by `ARENA_ODOM_PUBLISH_BACKEND`:

- `manual_step` is the project default. `ManualStepStatePublisher` publishes
  Odometry and `world -> odom -> base` TF from the registered active
  articulation pose on every explicit physics step; pose is the USD/PhysX
  world pose and twist is its same-clock finite difference. SpawnUsdRobot does
  not create its odom OmniGraph.
- `isaac_graph` is an explicit diagnostic alternative. SpawnUsdRobot owns the
  Odometry/TF graph: pose comes from the articulation transform and twist from
  `IsaacComputeOdometry`. The Python step publisher remains clock-only.

The legacy `ARENA_SPAWN_USD_ROBOT_ENABLE_ISAAC_ODOM_GRAPH` cannot select an
owner. It is ignored when inherited from an older launch environment so it
cannot silently resurrect the second graph; explicit graph diagnostics must
set the canonical `ARENA_ODOM_PUBLISH_BACKEND=isaac_graph`. Dual/auto ownership
is forbidden.

For Ai2_Bot2, the authoritative frame pair is
`Ai2_Bot2/odom -> Ai2_Bot2/base_footprint` because:

- `arena_robots/robots/Ai2_Bot2/model_params.yaml` declares
  `robot_base_frame: base_footprint` and `robot_odom_frame: odom`;
- Nav2 substitutes that model value into `robot_base_frame`;
- SpawnUsdRobot selects the live chassis articulation root ending in
  `Ai2_Bot2_Chassis/base_footprint` as the pose/twist source.

The frame label and physics source are intentionally distinct fields in the
registry. Camera TF remains an independent producer and is not part of the
odom backend switch.

Same-path respawn and DeletePrims unregister the exact robot owner before prim
reuse/destruction. Manual DDS publishers are destroyed through exact retained
handles. Every targeted graph Odometry/TF node is assigned a deterministic dead
topic and read back through both its authored USD attribute and live OmniGraph
attribute before registry removal. The ledger distinguishes an absent live node
from a present node, a present/missing live attribute, and the attribute value.
Only an explicitly absent live node may omit runtime readback; a present node
requires a present attribute whose value exactly equals the dead topic. Any
Set/readback error, `None`/empty/old live value, or partial success retains
ownership and blocks replacement; a TF failure cannot be hidden by a successful
Odometry-topic check.

A qualification ledger must independently prove:

1. one canonical topic and frame pair for every sample;
2. strictly increasing header stamps in capture order;
3. exactly one live publisher endpoint GID;
4. `world -> odom` and `odom -> base_footprint` TF connectivity;
5. no stale owner or endpoint after removal/respawn.

Rosbag2 sqlite preserves Odometry payloads but not live DDS endpoint GIDs. The
offline bag audit therefore reproduces frame/stamp evidence and deliberately
fails the GID predicate until a bounded runtime graph snapshot is supplied.
