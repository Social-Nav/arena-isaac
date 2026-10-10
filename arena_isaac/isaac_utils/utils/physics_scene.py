"""Keep referenced environments in the application's single physics scene."""
from pxr import Sdf, Usd, UsdPhysics


def consolidate_physics_scenes(stage, imported_root: str, arena_scene: str) -> dict:
    root = Sdf.Path(imported_root)
    canonical = Sdf.Path(arena_scene)
    prim = stage.GetPrimAtPath(canonical)
    if not prim or not prim.IsActive() or not prim.IsA(UsdPhysics.Scene):
        raise RuntimeError(f'Arena physics scene is unavailable: {canonical}')
    # Include a scene already made inactive by an earlier reference at this
    # path: a replacement environment can introduce new owner relationships.
    imported = [p for p in stage.TraverseAll()
                if p.IsA(UsdPhysics.Scene) and p.GetPath().HasPrefix(root)
                and p.GetPath() != canonical]
    removed = {p.GetPath() for p in imported}
    remapped = 0
    # Session opinions override references without modifying their source layers.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for body in list(stage.Traverse()):
            for rel in body.GetRelationships():
                if rel.GetBaseName() != 'simulationOwner':
                    continue
                targets = rel.GetForwardedTargets()
                if any(t in removed for t in targets):
                    rel.SetTargets(list(dict.fromkeys(canonical if t in removed else t for t in targets)))
                    remapped += 1
        for scene in imported:
            scene.SetActive(False)
    remaining = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if remaining != [str(canonical)]:
        raise RuntimeError(f'Expected only {canonical}, found physics scenes: {remaining}')
    return {'physics_scene': str(canonical), 'disabled': sorted(map(str, removed)),
            'owners_remapped': remapped}
