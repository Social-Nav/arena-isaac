import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "arena_isaac/run_isaacsim.py"


def _manual_class():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ManualReplicatorCameraPublisher"
    )


def test_manual_publisher_exposes_intrinsics_provenance_and_bounded_close():
    methods = {
        node.name: node
        for node in _manual_class().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "camera_info_provenance" in methods
    assert "prepare" in methods
    assert "close" in methods
    source = SOURCE.read_text(encoding="utf-8")
    assert "fallback_fx_fy_500" in source
    assert "annotator.detach" in source
    assert "render_product.destroy()" in source
    assert "warmup frames never receive a" in source


def test_manual_image_path_has_deprecation_tags_at_required_boundaries():
    klass = _manual_class()
    methods = {
        node.name: node for node in klass.body if isinstance(node, ast.FunctionDef)
    }
    assert "DEPRECATED" in (ast.get_docstring(klass) or "")
    assert "DEPRECATED" in ast.get_source_segment(
        SOURCE.read_text(encoding="utf-8"), methods["__init__"]
    )
    assert "DEPRECATED" in (ast.get_docstring(methods["_ensure_camera"]) or "")
    assert "DEPRECATED" in (ast.get_docstring(methods["publish"]) or "")
    source = SOURCE.read_text(encoding="utf-8")
    assert "DEPRECATED image path: default native does not instantiate" in source
    assert source.count("DEPRECATED manual camera image backend initialized") == 1
    publish_source = ast.get_source_segment(source, methods["publish"])
    assert "DEPRECATED manual camera image backend initialized" not in publish_source


def test_manual_publisher_still_uses_distance_to_camera_and_nonfinite_zero():
    source = SOURCE.read_text(encoding="utf-8")
    assert "get_annotator('distance_to_camera')" in source
    assert "arr[~np.isfinite(arr)] = 0.0" in source
    assert "DynamicCameraTfPublisher" in source
    assert "self._tf_pub" not in ast.get_source_segment(
        source, _manual_class()
    )
