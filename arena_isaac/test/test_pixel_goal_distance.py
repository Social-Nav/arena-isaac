import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "arena_isaac"
    / "data_logging"
    / "postprocess"
    / "process_raw_to_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("process_raw_to_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def camera_poses(xs):
    poses = np.repeat(np.eye(4)[None], len(xs), axis=0)
    # Camera optical axes in world coordinates: right=+Y, down=+Z, forward=+X.
    poses[:, :3, :3] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    poses[:, 0, 3] = xs
    return poses


def project_goal(robot_xs, max_goal_distance=4.0):
    depth = np.full((101, 101), 100.0, dtype=np.float32)
    return MODULE.farthest_visible_goal(
        camera_poses(robot_xs),
        frame_idx=0,
        depth=depth,
        fx=10.0,
        fy=10.0,
        cx=50.0,
        cy=50.0,
        lookahead=100.0,
        margin=0.0,
        min_ground_dist=0.0,
        robot_xy=np.column_stack([robot_xs, np.zeros(len(robot_xs))]),
        max_goal_distance=max_goal_distance,
    )


def test_goal_is_limited_to_four_metres():
    goal, relative_frame = project_goal(np.array([0.0, 2.0, 4.0, 4.1]))

    assert goal is not None
    assert relative_frame == 2


def test_path_stops_when_it_first_leaves_radius():
    _, relative_frame = project_goal(np.array([0.0, 2.0, 4.1, 3.0]))

    assert relative_frame == 1
