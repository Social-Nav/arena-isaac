from pathlib import Path


ODOM_GRAPH = Path(__file__).resolve().parents[1] / "isaac_utils/graphs/odom.py"


def test_odom_graph_creates_each_named_node_once():
    """A graph node cannot be created twice under the same OmniGraph path."""
    source = ODOM_GRAPH.read_text(encoding="utf-8")

    assert source.count("graph.node('compute_odometry'") == 1
    assert source.count("graph.node('get_chassis_prim'") == 1
