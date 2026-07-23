"""
Snapshot Capturer (Isaac side, pure logic — no ROS)

Captures a set of still images when the proactive-yielding trigger fires and the
simulation has been paused:

  - head camera  : RGB + depth
  - back camera  : RGB + depth
  - top-down view : RGB from a temporary orthographic camera created directly
                    above the robot, covering a configurable square extent.

Design notes
------------
* This class contains NO ROS code. It is driven from the main loop in
  ``run_isaacsim.py`` (which owns the render context). The ROS service
  ``isaac/CaptureSnapshot`` only raises a flag; the main loop calls
  :meth:`capture` while the sim is paused.
* Capturing while paused: ``world.step`` is not running, so a freshly created
  render product needs a few ``simulation_app.update()`` pumps before
  ``get_data()`` returns valid pixels. ``update()`` renders without stepping
  physics, so the frozen scene stays frozen.
* The top-down camera is created on demand and destroyed right after capture,
  so there is zero persistent overhead during normal operation.
"""

import os
import sys
import json
import datetime

import numpy as np

import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom, Gf, Sdf

try:
    from isaacsim.core.utils.prims import is_prim_path_valid, delete_prim
except ImportError:  # older Isaac layout
    from omni.isaac.core.utils.prims import is_prim_path_valid, delete_prim


# GfCamera treats aperture in millimetres; orthographic view width in *scene
# units* == horizontalAperture * APERTURE_UNIT (0.1). See pxr/base/gf/camera.h.
_APERTURE_UNIT = 0.1

# A freshly created render product needs a few Replicator orchestrator steps
# before its annotators return valid data. rt_subframes lets ray-traced
# rendering converge within each step.
_WARMUP_STEPS = 8
_RT_SUBFRAMES = 16

_HEAD_BACK_RES = (1280, 720)
_TOPDOWN_RES = (1024, 1024)


class SnapshotCapturer:
    """Grabs head/back/top-down stills into a timestamped folder."""

    def __init__(self, output_root, simulation_app):
        """
        Args:
            output_root: base dir; snapshots go under ``<root>/snapshots/``.
            simulation_app: the SimulationApp instance, used to pump renders
                            while the sim is paused.
        """
        self._app = simulation_app
        self._snapshots_root = os.path.join(output_root, "snapshots")

    # ── public API ────────────────────────────────────────────────────────────

    def capture(self, robot_position, head_cam_path, back_cam_path,
                topdown_half_extent=5.0, topdown_height=8.0, meta=None):
        """Capture all views. Returns (success: bool, out_dir: str | None).

        Args:
            robot_position: (x, y, z) world position of the robot, used to
                centre the top-down camera.
            head_cam_path / back_cam_path: USD Camera prim paths (may be empty
                or invalid — that view is then skipped with a warning).
            topdown_half_extent: half side length (m) the top-down view covers,
                i.e. the view spans ``2 * half_extent`` metres.
            topdown_height: height (m) above the robot for the top-down camera.
            meta: optional dict merged into meta.json (goal, blocking peds, …).
        """
        out_dir = self._make_out_dir()
        sys.stderr.write(f"[Snapshot]   out_dir={out_dir}\n")
        saved = {"head": False, "back": False, "topdown": False}

        # IMPORTANT: create ALL render products first, warm up ONCE, then read.
        # Creating a render product, reading, and destroying it before creating
        # the next one leaves later products' annotators empty — orchestrator.step
        # only populates products that exist at warmup time. So we batch them.
        views = []          # list of dicts: {tag, rp, rgb, depth}
        topdown_cam_path = None
        try:
            # ── setup: head + back (existing camera prims) ──
            for tag, cam_path in (("head", head_cam_path), ("back", back_cam_path)):
                if not cam_path or not is_prim_path_valid(cam_path):
                    sys.stderr.write(
                        f"[Snapshot] {tag} camera path invalid, skipping: {cam_path}\n")
                    continue
                rp = rep.create.render_product(cam_path, _HEAD_BACK_RES)
                # Isaac 5.1: a freshly created render product is lazy — its hydra
                # texture / OmniGraph nodes are not built until the app pumps
                # frames. Attaching an annotator immediately fails inside
                # activate_node_template ("Annotator rgb is not attached to any
                # render products"), and this got worse once the back camera added
                # a second resident SDG pipeline. Pump several updates so the RP is
                # real before attaching (one update was not enough).
                for _ in range(_WARMUP_STEPS):
                    self._app.update()
                rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
                depth_a = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
                # camera_params gives the SAME-FRAME view/projection matrices, so
                # intrinsics/extrinsics stay aligned with this rgb+depth frame.
                camp_a = rep.AnnotatorRegistry.get_annotator("camera_params")
                rgb_a.attach(rp)
                depth_a.attach(rp)
                camp_a.attach(rp)
                views.append({"tag": tag, "rp": rp, "rgb": rgb_a,
                              "depth": depth_a, "camp": camp_a})
                sys.stderr.write(f"[Snapshot] {tag}: render_product+annotators attached\n")

            # ── setup: topdown (temporary orthographic camera) ──
            td = self._setup_topdown(robot_position, topdown_half_extent, topdown_height)
            if td is not None:
                topdown_cam_path = td["cam_path"]
                views.append({"tag": "topdown", "rp": td["rp"], "rgb": td["rgb"],
                              "depth": None, "camp": None})

            # ── warmup ONCE for all render products together ──
            if views:
                sys.stderr.write(f"[Snapshot] warming up {len(views)} view(s) together...\n")
                self._warmup()

            # ── read + save each view ──
            for v in views:
                tag = v["tag"]
                rgb = v["rgb"].get_data()
                sys.stderr.write(
                    f"[Snapshot] {tag}: rgb get_data -> "
                    f"shape={getattr(rgb, 'shape', None)} size={getattr(rgb, 'size', 0)}\n")
                # Extra diagnostics for topdown: check pixel value range
                if tag == "topdown" and rgb is not None and getattr(rgb, "size", 0) > 0:
                    try:
                        import numpy as np
                        arr = np.asarray(rgb)
                        sys.stderr.write(
                            f"[Snapshot] topdown: RGB stats -> "
                            f"dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, "
                            f"mean={arr.mean():.2f}\n")
                    except Exception as e:  # noqa: BLE001
                        sys.stderr.write(f"[Snapshot] topdown: stats failed: {e}\n")
                ok = self._save_rgb(rgb, out_dir, f"{tag}_rgb")
                if v["depth"] is not None:
                    depth = v["depth"].get_data()
                    sys.stderr.write(
                        f"[Snapshot] {tag}: depth get_data -> "
                        f"shape={getattr(depth, 'shape', None)} size={getattr(depth, 'size', 0)}\n")
                    # Raw float32 metric depth (.npy) for reprojection — NOT the
                    # normalized PNG (lossy, unused). Reprojection needs true metres.
                    ok = self._save_depth_npy(depth, out_dir, f"{tag}_depth") or ok
                if v["camp"] is not None:
                    camp = v["camp"].get_data()
                    self._save_camera_json(
                        camp, _HEAD_BACK_RES, out_dir, f"{tag}_camera")
                saved[tag] = ok
        finally:
            # ── teardown: detach annotators BEFORE destroying render products ──
            # Annotators from AnnotatorRegistry are singletons (one per type). If
            # we destroy a render product without detaching, the singleton stays
            # bound to a dead product, and the NEXT capture's get_data() returns
            # empty — the classic "first capture works, all later ones fail" bug.
            for v in views:
                for key in ("rgb", "depth", "camp"):
                    annot = v.get(key)
                    if annot is not None:
                        try:
                            annot.detach(v["rp"])
                        except Exception:  # noqa: BLE001
                            pass
            for v in views:
                try:
                    v["rp"].destroy()
                except Exception:  # noqa: BLE001
                    pass
            if topdown_cam_path is not None:
                try:
                    if is_prim_path_valid(topdown_cam_path):
                        delete_prim(topdown_cam_path)
                except Exception:  # noqa: BLE001
                    pass

        sys.stderr.write(f"[Snapshot]   results: {saved}\n")

        # 4. meta.json
        meta_out = {
            "robot_position": [float(v) for v in robot_position],
            "topdown_half_extent": float(topdown_half_extent),
            "topdown_height": float(topdown_height),
            "saved": saved,
        }
        if meta:
            meta_out.update(meta)
        try:
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta_out, f, indent=2)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[Snapshot] meta.json write failed: {e}\n")

        success = any(saved.values())
        return success, (out_dir if success else None)

    # ── internals ─────────────────────────────────────────────────────────────

    def _make_out_dir(self):
        beijing = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        name = beijing.strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(self._snapshots_root, name)
        old_umask = os.umask(0)
        os.makedirs(out_dir, mode=0o777, exist_ok=True)
        os.umask(old_umask)
        return out_dir

    def _setup_topdown(self, robot_position, half_extent, height):
        """Create temporary ortho camera above robot; return setup dict or None.

        Returns dict {cam_path, rp, rgb} on success, None on failure. Caller must
        destroy the render product and delete the camera prim when done.
        """
        stage = omni.usd.get_context().get_stage()
        cam_path = "/World/_snapshot_topdown_cam"
        try:
            # Remove a stale one if a previous capture left it behind.
            if is_prim_path_valid(cam_path):
                delete_prim(cam_path)

            cam = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
            cam.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
            sys.stderr.write(f"[Snapshot] topdown: camera prim created at {cam_path}\n")

            # Orthographic view width (scene units) = aperture * _APERTURE_UNIT.
            # Convert desired metres → scene units via metersPerUnit.
            mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
            view_scene_units = (2.0 * half_extent) / mpu
            aperture = view_scene_units / _APERTURE_UNIT
            cam.CreateHorizontalApertureAttr(float(aperture))
            cam.CreateVerticalApertureAttr(float(aperture))
            near = 0.01
            far = float((height + 2.0 * half_extent) / mpu + 10.0)
            cam.CreateClippingRangeAttr(Gf.Vec2f(near, far))
            sys.stderr.write(
                f"[Snapshot] topdown: projection=orthographic, "
                f"aperture={aperture:.2f}, clipping=[{near:.2f}, {far:.2f}], mpu={mpu}\n")

            # Place above the robot looking straight DOWN. A USD camera looks
            # down its own local -Z by default; with identity rotation and the
            # camera sitting above the robot, -Z already points at the floor.
            # (A previous 180° flip about X aimed it UP at the empty sky → black.)
            rx, ry, rz = robot_position
            cam_world_pos = (float(rx), float(ry), float(rz + height))
            xform = UsdGeom.Xformable(cam.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*cam_world_pos))
            # No rotation: local -Z = world -Z = straight down.
            sys.stderr.write(
                f"[Snapshot] topdown: world_pos={cam_world_pos}, rotation=identity (look -Z / down)\n")

            rp = rep.create.render_product(cam_path, _TOPDOWN_RES)
            # Isaac 5.1: pump several frames so the lazily-created render product
            # is real before attaching the annotator (see head/back path; one
            # frame was not enough with the back camera's SDG pipeline resident).
            for _ in range(_WARMUP_STEPS):
                self._app.update()
            rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annot.attach(rp)
            sys.stderr.write(
                f"[Snapshot] topdown: render_product created, resolution={_TOPDOWN_RES}\n")
            return {"cam_path": cam_path, "rp": rp, "rgb": rgb_annot}
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[Snapshot] topdown setup failed: {e}\n")
            # Cleanup attempt on failure
            try:
                if is_prim_path_valid(cam_path):
                    delete_prim(cam_path)
            except Exception:  # noqa: BLE001
                pass
            return None

    def _warmup(self):
        """Drive the Replicator orchestrator so annotators produce valid data.

        While the timeline is paused, simulation_app.update() alone does NOT
        tick the Replicator pipeline, so annotator.get_data() stays empty.
        rep.orchestrator.step() synchronously renders one Replicator frame and
        writes the annotator buffers; delta_time=0.0 keeps physics frozen.
        A few steps let a freshly created render product / RTX rendering settle.
        """
        for i in range(_WARMUP_STEPS):
            try:
                rep.orchestrator.step(
                    rt_subframes=_RT_SUBFRAMES,
                    delta_time=0.0,
                    pause_timeline=False,
                )
                if i == 0:
                    sys.stderr.write(
                        f"[Snapshot] warmup: orchestrator.step x{_WARMUP_STEPS} "
                        f"(rt_subframes={_RT_SUBFRAMES})\n")
            except TypeError:
                # Older Replicator signatures may not accept all kwargs.
                try:
                    rep.orchestrator.step(rt_subframes=_RT_SUBFRAMES)
                    if i == 0:
                        sys.stderr.write(
                            f"[Snapshot] warmup: orchestrator.step x{_WARMUP_STEPS} "
                            "(legacy signature)\n")
                except Exception as e:  # noqa: BLE001
                    if i == 0:
                        sys.stderr.write(
                            f"[Snapshot] orchestrator.step failed: {e}; "
                            "falling back to app.update() only\n")
            except Exception as e:  # noqa: BLE001
                if i == 0:
                    sys.stderr.write(
                        f"[Snapshot] orchestrator.step failed: {e}; "
                        "falling back to app.update() only\n")
            self._app.update()

    # ── image saving (PIL if available, else .npy fallback) ────────────────────

    def _save_rgb(self, rgb, out_dir, name):
        if rgb is None or getattr(rgb, "size", 0) == 0:
            sys.stderr.write(f"[Snapshot] {name}: empty RGB, skipped\n")
            return False
        # Diagnose for topdown specifically
        if "topdown" in name:
            sys.stderr.write(
                f"[Snapshot] {name}: before asarray -> type={type(rgb)}, "
                f"shape={getattr(rgb, 'shape', None)}\n")
        rgb = np.asarray(rgb)
        if "topdown" in name:
            sys.stderr.write(
                f"[Snapshot] {name}: after asarray -> dtype={rgb.dtype}, "
                f"shape={rgb.shape}, min={rgb.min()}, max={rgb.max()}\n")
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[..., :3]
            if "topdown" in name:
                sys.stderr.write(
                    f"[Snapshot] {name}: after drop alpha -> "
                    f"shape={rgb.shape}, min={rgb.min()}, max={rgb.max()}\n")
        return self._write_png(rgb.astype(np.uint8), out_dir, name)

    def _save_depth_npy(self, depth, out_dir, name):
        """Save raw float32 metric depth (distance_to_camera) as .npy.

        No normalization — reprojection needs true metres. Inf (sky/no-hit) is
        kept as inf so the reprojector can detect and reject those pixels.
        """
        if depth is None or getattr(depth, "size", 0) == 0:
            sys.stderr.write(f"[Snapshot] {name}: empty depth, skipped\n")
            return False
        try:
            arr = np.asarray(depth).astype(np.float32)
            np.save(os.path.join(out_dir, f"{name}.npy"), arr)
            finite = arr[np.isfinite(arr)]
            rng = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
            sys.stderr.write(
                f"[Snapshot] {name}.npy saved (float32 metres), range={rng}\n")
            return True
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[Snapshot] {name} depth .npy save failed: {e}\n")
            return False

    def _save_camera_json(self, params, res, out_dir, name):
        """Compute K (3x3) + camera<->world matrices from the camera_params
        annotator and save as JSON. Same-frame as rgb/depth (paused, no step).

        Stores BOTH the raw Isaac matrices and derived K/pose so the offline
        reprojector can pin down the exact matrix convention (row/col-major)
        without re-running the sim.
        """
        if not params:
            sys.stderr.write(f"[Snapshot] {name}: empty camera_params, skipped\n")
            return False
        try:
            width, height = int(res[0]), int(res[1])
            view = np.asarray(params["cameraViewTransform"]).reshape(4, 4)
            proj = np.asarray(params["cameraProjection"]).reshape(4, 4)

            # Intrinsics (same derivation as data logger process_camera_data).
            fx = float(proj[0, 0]) * width / 2.0
            fy = float(proj[1, 1]) * height / 2.0
            cx = width / 2.0
            cy = height / 2.0
            K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

            data = {
                "width": width,
                "height": height,
                "K": K,
                # Raw Isaac matrices (row-major as returned). The reprojector
                # decides the exact convention; keep raw to avoid guessing here.
                "cameraViewTransform": view.flatten().tolist(),
                "cameraProjection": proj.flatten().tolist(),
            }
            with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
                json.dump(data, f, indent=2)
            sys.stderr.write(f"[Snapshot] {name}.json saved (K + view/proj)\n")
            return True
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[Snapshot] {name} camera json save failed: {e}\n")
            return False

    def _write_png(self, arr, out_dir, name):
        """Write uint8 array as PNG via PIL; fall back to .npy on failure."""
        try:
            from PIL import Image
            Image.fromarray(arr).save(os.path.join(out_dir, f"{name}.png"))
            return True
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(
                f"[Snapshot] PNG write failed for {name} ({e}); saving .npy\n")
            try:
                np.save(os.path.join(out_dir, f"{name}.npy"), arr)
                return True
            except Exception as e2:  # noqa: BLE001
                sys.stderr.write(f"[Snapshot] .npy fallback failed: {e2}\n")
                return False
