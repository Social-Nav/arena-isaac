"""Execute wheel scaling with real USD composition and a live graph cache."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('pxr')
from pxr import Sdf, Usd


def scaler(stage, cache):
    source = Path(__file__).resolve().parents[1] / 'arena_isaac/services/EditPrims.py'
    function = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_rescale_differential_controllers')
    ns = dict(Usd=Usd, Sdf=Sdf, math=math,
              DIFFERENTIAL_CONTROLLER_TYPE='isaacsim.robot.wheeled_robots.DifferentialController',
              omni=SimpleNamespace(usd=SimpleNamespace(get_context=lambda: SimpleNamespace(get_stage=lambda: stage))),
              og=SimpleNamespace(get_node_by_path=lambda _: SimpleNamespace(is_valid=lambda: True, get_attribute=cache.__getitem__)),
              carb=SimpleNamespace(log_warn=lambda _: None))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), ns)
    return ns[function.name]


class LiveAttribute:
    def __init__(self, value, reject=False):
        self.value, self.reject = value, reject

    def is_valid(self):
        return True

    def get(self):
        return self.value

    def set(self, value):
        if not self.reject:
            self.value = value


def setup_scaler():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/Robot', 'Xform')
    node = stage.DefinePrim('/Robot/Controller', 'OmniGraphNode')
    node.CreateAttribute('node:type', Sdf.ValueTypeNames.Token).Set('isaacsim.robot.wheeled_robots.DifferentialController')
    cache = {}
    for key, value in [('inputs:wheelRadius', .085), ('inputs:wheelDistance', .416)]:
        node.CreateAttribute(key, Sdf.ValueTypeNames.Double).Set(value)
        cache[key] = LiveAttribute(value)
    return scaler(stage, cache), node, cache


def test_repeated_absolute_scale_and_restore():
    scale, node, cache = setup_scaler()
    for value in [.8, .8, 1.]:
        assert scale('/Robot', value) == 2
        assert cache['inputs:wheelRadius'].get() == pytest.approx(.085 * value)
        assert cache['inputs:wheelDistance'].get() == pytest.approx(.416 * value)
        assert node.GetAttribute('inputs:wheelRadius').Get() == pytest.approx(.085 * value)


def test_live_graph_rejecting_scale_cannot_report_success():
    scale, _, cache = setup_scaler()
    cache['inputs:wheelRadius'].reject = True
    with pytest.raises(RuntimeError, match='live wheel dimension'):
        scale('/Robot', .8)
