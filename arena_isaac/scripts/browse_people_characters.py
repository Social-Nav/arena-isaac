#!/usr/bin/env python3
"""Interactively browse the classic and base-DH Isaac People characters."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

from isaacsim import SimulationApp


# Isaac 4.5 and 5.1 use the same classic People character layout. The USD
# basename is not always identical to its folder, so keep the mapping explicit.
CLASSIC_CHARACTERS = (
    ("F_Business_02", "F_Business_02"),
    ("F_Medical_01", "F_Medical_01"),
    ("M_Medical_01", "M_Medical_01"),
    ("female_adult_police_01_new", "female_adult_police_01_new"),
    ("female_adult_police_02", "female_adult_police_02"),
    ("female_adult_police_03_new", "female_adult_police_03_new"),
    ("male_adult_construction_01_new", "male_adult_construction_01_new"),
    ("male_adult_construction_03", "male_adult_construction_03"),
    ("male_adult_construction_05_new", "male_adult_construction_05_new"),
    ("male_adult_police_04", "male_adult_police_04"),
    ("original_female_adult_business_02", "female_adult_business_02"),
    ("original_female_adult_medical_01", "female_adult_medical_01"),
    ("original_female_adult_police_01", "female_adult_police_01"),
    ("original_female_adult_police_02", "female_adult_police_02"),
    ("original_female_adult_police_03", "female_adult_police_03"),
    ("original_male_adult_construction_01", "male_adult_construction_01"),
    ("original_male_adult_construction_02", "male_adult_construction_02"),
    ("original_male_adult_construction_03", "male_adult_construction_03"),
    ("original_male_adult_construction_05", "male_adult_construction_05"),
    ("original_male_adult_medical_01", "male_adult_medical_01"),
    ("original_male_adult_police_04", "male_adult_police_04"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection",
        choices=("identities", "all", "characters", "dh"),
        default="all",
        help=(
            "identities shows the reviewed 32-person deduplicated catalog; "
            "the other modes browse raw asset collections"
        ),
    )
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--random-variants",
        action="store_true",
        help="use a random DH Extended color variant for each reviewed DH identity",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used by --random-variants (default: 42)",
    )
    args = parser.parse_args()

    app = SimulationApp({"headless": False, "width": 1600, "height": 900})
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

        import isaacsim.storage.native
        import numpy as np
        import omni.client
        import omni.usd
        import yaml
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import GroundPlane
        from isaacsim.core.utils.prims import create_prim, define_prim
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Usd, UsdGeom

        from pedestrian.simulator.logic.people.character_assets import (
            CHARACTER_ASSETS,
            CharacterAsset,
        )
        from pedestrian.simulator.logic.people.person import Person

        classic_entries = []
        dh_character_entries = []
        identity_catalog = None
        if args.collection == "identities":
            catalog_path = (
                Path(__file__).parents[1]
                / "pedestrian"
                / "config"
                / "character_identities.yaml"
            )
            with catalog_path.open(encoding="utf-8") as stream:
                identity_catalog = yaml.safe_load(stream)

            identity_to_pool = {
                identity_id: pool_name
                for pool_name, pool in identity_catalog["pools"].items()
                for identity_id in pool["identities"]
            }
            variant_rng = random.Random(args.seed)

            for identity in identity_catalog["classic_identities"]:
                classic_entries.append(
                    {
                        "source": "characters",
                        "label": identity["id"],
                        "identity_id": identity["id"],
                        "pool": identity_to_pool[identity["id"]],
                        "relative_usd": identity["asset_variants"][0],
                    }
                )
            for identity in identity_catalog["dh_identities"]:
                character_id = identity["uuid"]
                variant = variant_rng.randint(1, 10) if args.random_variants else 0
                if variant:
                    variant_id = f"{character_id}_{variant}"
                    relative_usd = (
                        "Isaac/People/DH_Characters_Extended/"
                        f"{variant_id}/{variant_id}.usd"
                    )
                else:
                    relative_usd = (
                        "Isaac/People/DH_Characters/"
                        f"{character_id}/{character_id}.usd"
                    )
                dh_character_entries.append(
                    {
                        "source": "dh",
                        "label": identity["id"],
                        "identity_id": identity["id"],
                        "uuid": character_id,
                        "pool": identity_to_pool[identity["id"]],
                        "variant": variant,
                        "relative_usd": relative_usd,
                    }
                )

        if args.collection in {"all", "characters"}:
            for folder, usd_name in CLASSIC_CHARACTERS:
                classic_entries.append(
                    {
                        "source": "characters",
                        "label": folder,
                        "relative_usd": (
                            f"Isaac/People/Characters/{folder}/{usd_name}.usd"
                        ),
                    }
                )

        if args.collection in {"all", "dh"}:
            assets_root = isaacsim.storage.native.get_assets_root_path()
            dh_root = f"{assets_root}/Isaac/People/DH_Characters"
            result, dh_entries = omni.client.list(dh_root)
            if result != omni.client.Result.OK:
                raise RuntimeError(f"Cannot list {dh_root}: {result}")
            for entry in sorted(dh_entries, key=lambda item: item.relative_path):
                character_id = entry.relative_path
                if character_id in {"Common", "filter.json"} or character_id.startswith(
                    "."
                ):
                    continue
                dh_character_entries.append(
                    {
                        "source": "dh",
                        "label": character_id,
                        "relative_usd": (
                            "Isaac/People/DH_Characters/"
                            f"{character_id}/{character_id}.usd"
                        ),
                    }
                )

        # The reviewed identity view follows catalog order: classic first, then
        # DH. Raw collection browsing keeps DH first for historical manifests.
        if args.collection == "identities":
            entries = classic_entries + dh_character_entries
        else:
            entries = dh_character_entries + classic_entries
        if not entries:
            raise RuntimeError("No character assets were found")

        omni.usd.get_context().new_stage()
        app.update()
        columns = max(1, args.columns)
        rows = math.ceil(len(entries) / columns)
        spacing_x = 4.0
        spacing_y = 4.5
        ground_size = max(40.0, columns * spacing_x + 8.0, rows * spacing_y + 8.0)
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
        world.scene.add(
            GroundPlane(prim_path="/World/GroundPlane", size=ground_size)
        )
        define_prim("/World/Pedestrians", "Xform")
        create_prim(
            "/World/BrowserLight",
            "DomeLight",
            attributes={"inputs:intensity": 1100.0},
        )

        people = []
        results = []
        for index, entry in enumerate(entries):
            row, column = divmod(index, columns)
            x = (column - (columns - 1) / 2.0) * spacing_x
            y = ((rows - 1) / 2.0 - row) * spacing_y
            position = [float(x), float(y), 0.0]
            model_key = f"browser_{index + 1:02d}"
            CHARACTER_ASSETS[model_key] = CharacterAsset(
                relative_usd=entry["relative_usd"],
                source=entry["source"],
                description=entry["label"],
            )
            # DH labels are UUIDs containing hyphens, which are not valid in a
            # USD prim identifier. Keep the full label in the manifest and use
            # a sanitized, stable stage name for spawning.
            stage_name = f"{model_key}_{entry['source']}"
            result_record = {
                **entry,
                "number": index + 1,
                "row": row + 1,
                "column": column + 1,
                "stage_path": f"/World/Pedestrians/{stage_name}",
            }
            print(
                "BROWSER_ITEM=" + json.dumps(result_record, sort_keys=True),
                flush=True,
            )
            try:
                person = Person(
                    world,
                    result_record["stage_path"],
                    model_key,
                    position,
                    0.0,
                )
                if (
                    args.collection == "identities"
                    and entry.get("pool") == "pedestrian"
                    and entry["source"] == "dh"
                ):
                    hidden_tokens = identity_catalog["appearance_profiles"][
                        "general_pedestrian"
                    ]["hide_prim_name_contains"]
                    hidden_prims = []
                    for prim in Usd.PrimRange(person.prim):
                        prim_name = prim.GetName().lower()
                        if any(token in prim_name for token in hidden_tokens):
                            imageable = UsdGeom.Imageable(prim)
                            if imageable:
                                imageable.MakeInvisible()
                                hidden_prims.append(str(prim.GetPath()))
                    result_record["hidden_prims"] = hidden_prims
                left = np.array([x - 1.25, y, 0.0])
                right = np.array([x + 1.25, y, 0.0])
                endpoints = (right, left) if index % 2 == 0 else (left, right)
                person.update_target_positions(list(endpoints) * 500, walk_speed=0.65)
                people.append(person)
                result_record["spawned"] = True
            except Exception as exc:
                result_record["spawned"] = False
                result_record["error"] = repr(exc)
                # A malformed/non-character asset must not remain in the stage:
                # unresolved prims can poison later USD reference composition.
                world.stage.RemovePrim(result_record["stage_path"])
                world.stage.RemovePrim(result_record["stage_path"] + "_lidar_proxy")
            results.append(result_record)

        world.reset()
        world.play()
        for _ in range(180):
            world.step(render=True)

        for person, result_record in zip(
            people, (record for record in results if record["spawned"])
        ):
            try:
                result_record["animation_graph"] = bool(person.character_graph)
                result_record["skel_root"] = person.character_skel_root_stage_path
            except Exception as exc:
                result_record["animation_graph"] = False
                result_record["animation_error"] = repr(exc)

        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "browser_manifest.json")
        with open(result_path, "w", encoding="utf-8") as stream:
            json.dump(results, stream, ensure_ascii=False, indent=2)

        camera_y = -max(24.0, rows * spacing_y * 1.5)
        camera_z = max(13.0, rows * spacing_y * 1.25)
        set_camera_view(
            eye=[0.0, camera_y, camera_z],
            target=[0.0, 0.0, 1.0],
            camera_prim_path="/OmniverseKit_Persp",
        )
        succeeded = sum(bool(record.get("animation_graph")) for record in results)
        print(
            f"BROWSER_GUI_READY={succeeded}/{len(results)} characters ready; "
            "close Isaac Sim or press Ctrl+C to exit",
            flush=True,
        )
        print(f"BROWSER_MANIFEST={result_path}", flush=True)
        try:
            while app.is_running():
                world.step(render=True)
        except KeyboardInterrupt:
            print("BROWSER_GUI_STOPPED", flush=True)
        return 0 if succeeded == len(results) else 1
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
