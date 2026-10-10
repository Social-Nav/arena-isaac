from pathlib import Path
import importlib.util

import pytest

pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdPhysics

spec = importlib.util.spec_from_file_location(
    'single_scene', Path(__file__).parents[1] / 'isaac_utils/utils/physics_scene.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reference_owners_are_remapped_without_modifying_source_and_reload_is_safe(tmp_path):
    path = str(tmp_path / 'scene.usda')
    source = Usd.Stage.CreateNew(path)
    root = UsdGeom.Xform.Define(source, '/Root').GetPrim()
    source.SetDefaultPrim(root)
    UsdPhysics.Scene.Define(source, '/Root/physicsScene')
    body = UsdGeom.Cube.Define(source, '/Root/body').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateSimulationOwnerRel().SetTargets(['/Root/physicsScene'])
    source.GetRootLayer().Save()
    original = Path(path).read_bytes()
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    for _ in range(2):
        root = UsdGeom.Xform.Define(stage, '/World/Scene').GetPrim()
        root.GetReferences().AddReference(path)
        for _ in range(2):
            module.consolidate_physics_scenes(stage, '/World/Scene', '/physicsScene')
            assert [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)] == ['/physicsScene']
            assert list(stage.GetPrimAtPath('/World/Scene/body').GetRelationship('physics:simulationOwner').GetTargets()) == ['/physicsScene']
            assert Path(path).read_bytes() == original
        stage.RemovePrim('/World/Scene')


def test_unrelated_physics_scene_fails_instead_of_silently_disabling_user_content():
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    UsdPhysics.Scene.Define(stage, '/UserPhysics')
    UsdGeom.Xform.Define(stage, '/World/Scene')
    with pytest.raises(RuntimeError, match='Expected only'):
        module.consolidate_physics_scenes(stage, '/World/Scene', '/physicsScene')
    assert stage.GetPrimAtPath('/UserPhysics').IsActive()


def test_replacement_scene_remaps_new_bodies_even_with_existing_inactive_override():
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    UsdPhysics.Scene.Define(stage, '/World/Scene/physicsScene')
    module.consolidate_physics_scenes(stage, '/World/Scene', '/physicsScene')
    body = UsdGeom.Cube.Define(stage, '/World/Scene/new_body').GetPrim()
    owner = UsdPhysics.RigidBodyAPI.Apply(body).CreateSimulationOwnerRel()
    owner.SetTargets(['/World/Scene/physicsScene'])
    module.consolidate_physics_scenes(stage, '/World/Scene', '/physicsScene')
    assert list(owner.GetTargets()) == ['/physicsScene']


@pytest.mark.parametrize('world', ['MV4AFHQKTKJZ2AABAAAAAEA8', 'MWF4WLIKTIFZIAABAAAAACA8'])
def test_real_grscenes_physics_reference(world):
    path = Path('/data/scenes/commercial_scenes/scenes') / (world + '_usd/start_result_navigation.usd')
    if not path.exists():
        pytest.skip('GRScenes dataset is not mounted')
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    root = UsdGeom.Xform.Define(stage, '/World/Scene').GetPrim()
    root.GetReferences().AddReference(str(path))
    report = module.consolidate_physics_scenes(stage, '/World/Scene', '/physicsScene')
    assert report['disabled'] == ['/World/Scene/physicsScene']
