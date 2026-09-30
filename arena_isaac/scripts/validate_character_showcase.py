#!/usr/bin/env python3
"""Spawn Arena's curated pedestrian pool and capture a review image."""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaacsim import SimulationApp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument(
        "--pool-showcase",
        action="store_true",
        help="resolve ten sample requests through the four scenario-facing pools",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="open the Isaac Sim GUI and keep it running for visual inspection",
    )
    args = parser.parse_args()

    app = SimulationApp(
        {"headless": not args.keep_open, "width": 1280, "height": 720}
    )
    try:
        from isaacsim.core.utils.extensions import enable_extension

        for extension in (
            "omni.anim.people",
            "omni.anim.navigation.bundle",
            "omni.anim.timeline",
            "omni.anim.graph.bundle",
            "omni.anim.retarget.bundle",
            "omni.kit.scripting",
        ):
            enable_extension(extension)
        app.update()

        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import GroundPlane
        from isaacsim.core.utils.prims import create_prim, define_prim
        from isaacsim.core.utils.viewports import set_camera_view

        from pedestrian.simulator.logic.people.character_assets import CHARACTER_ASSETS
        from pedestrian.simulator.logic.people.character_selector import (
            CharacterPoolSelector,
        )
        from pedestrian.simulator.logic.people.person import Person

        omni.usd.get_context().new_stage()
        app.update()
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
        world.scene.add(GroundPlane(prim_path="/World/GroundPlane", size=30.0))
        define_prim("/World/Pedestrians", "Xform")
        create_prim(
            "/World/ShowcaseLight",
            "DomeLight",
            attributes={"inputs:intensity": 1100.0},
        )

        people = []
        results = []
        if args.pool_showcase:
            requested_models = (
                ["pedestrian"] * 4
                + ["doctor"] * 2
                + ["police"] * 2
                + ["construction_worker"] * 2
            )
            selector = CharacterPoolSelector(seed=args.seed)
        else:
            requested_models = list(CHARACTER_ASSETS)
            selector = None

        for index, requested_model in enumerate(requested_models):
            row, column = divmod(index, 5)
            position = [float((column - 2) * 2.7), float(1.8 - row * 3.6), 0.0]
            model_name = (
                selector.resolve(f"showcase_{index:02d}", requested_model)
                if selector is not None
                else requested_model
            )
            stage_path = f"/World/Pedestrians/showcase_{index:02d}"
            try:
                person = Person(world, stage_path, model_name, position, 0.0)
                left = np.array([position[0] - 1.5, position[1], position[2]])
                right = np.array([position[0] + 1.5, position[1], position[2]])
                # Queue a long shuttle route so GUI inspection shows continuous
                # walking and turning. Alternate the initial direction between
                # neighboring people to make gait comparison easier.
                endpoints = (right, left) if index % 2 == 0 else (left, right)
                person.update_target_positions(
                    list(endpoints) * 500,
                    walk_speed=0.65,
                )
                people.append(person)
                results.append(
                    {
                        "requested_model": requested_model,
                        "resolved_model": model_name,
                        "spawned": True,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "requested_model": requested_model,
                        "resolved_model": model_name,
                        "spawned": False,
                        "error": repr(exc),
                    }
                )

        world.reset()
        world.play()
        for _ in range(args.steps):
            world.step(render=True)

        for person, result in zip(people, (r for r in results if r["spawned"])):
            try:
                result["animation_graph"] = bool(person.character_graph)
                result["skel_root"] = person.character_skel_root_stage_path
            except Exception as exc:
                result["animation_graph"] = False
                result["animation_error"] = repr(exc)

        os.makedirs(args.output_dir, exist_ok=True)
        camera = rep.create.camera(
            position=(0.0, -18.0, 8.0),
            look_at=(0.0, 0.0, 1.0),
            focal_length=30.0,
        )
        render_product = rep.create.render_product(camera, (1280, 720))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=args.output_dir, rgb=True)
        writer.attach([render_product])
        rep.orchestrator.step(rt_subframes=8)
        writer.detach()

        result_path = os.path.join(args.output_dir, "validation.json")
        with open(result_path, "w", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2)
        print("SHOWCASE_RESULT=" + json.dumps(results), flush=True)
        print("SHOWCASE_OUTPUT=" + args.output_dir, flush=True)
        success = all(
            result["spawned"] and result.get("animation_graph")
            for result in results
        )

        if args.keep_open:
            set_camera_view(
                eye=[0.0, -18.0, 8.0],
                target=[0.0, 0.0, 1.0],
                camera_prim_path="/OmniverseKit_Persp",
            )
            print(
                "SHOWCASE_GUI_READY=Close the Isaac Sim window or press Ctrl+C to exit",
                flush=True,
            )
            try:
                while app.is_running():
                    world.step(render=True)
            except KeyboardInterrupt:
                print("SHOWCASE_GUI_STOPPED", flush=True)

        return 0 if success else 1
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
