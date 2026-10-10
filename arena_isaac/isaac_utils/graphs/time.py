import omni.graph.core as og
from isaac_utils.graphs import Graph


def PublishTime(graph_path: str) -> bool:
    graph = Graph(graph_path)

    on_physics_step = graph.node('on_physics_step', 'isaacsim.core.nodes.OnPhysicsStep')
    read_simulation_time = graph.node('read_simulation_time', 'isaacsim.core.nodes.IsaacReadSimulationTime')
    publish_clock = graph.node('publish_clock', 'isaacsim.ros2.bridge.ROS2PublishClock')
    ros2_context = graph.node('ros2_context', 'isaacsim.ros2.bridge.ROS2Context')

    on_physics_step.connect('step', publish_clock, 'execIn')
    ros2_context.connect('context', publish_clock, 'context')
    read_simulation_time.connect('simulationTime', publish_clock, 'timeStamp')
    read_simulation_time.attribute('resetOnStop', False)

    graph.load_extensions()
    return graph.execute(og.Controller())


def simulation_time() -> float:
    """The same continuous native clock used by ROS and sensor graphs."""
    return float(og.Controller.attribute(
        "/World/publish_time/read_simulation_time.outputs:simulationTime"
    ).get())
