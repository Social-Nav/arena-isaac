#!/usr/bin/env python3
"""Inspect Isaac Sim Digital Human assets and print their Persona metadata.

Run this script with Isaac Sim's Python interpreter.  It is intentionally
read-only: no assets or stages are saved.
"""

from __future__ import annotations

import json
import argparse
import sys

from isaacsim import SimulationApp


def _value(value):
    """Convert common USD values into JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "pathString"):
        return value.pathString
    try:
        return list(value)
    except TypeError:
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print one compact JSON object per character instead of full USD metadata.",
    )
    args = parser.parse_args()
    app = SimulationApp({"headless": True})
    try:
        import isaacsim.storage.native
        import omni.client
        from pxr import Usd

        assets_root = isaacsim.storage.native.get_assets_root_path()
        dh_root = f"{assets_root}/Isaac/People/DH_Characters"
        result, entries = omni.client.list(dh_root)
        if result != omni.client.Result.OK:
            print(json.dumps({"error": f"cannot list {dh_root}: {result}"}))
            return 1

        records = []
        for entry in entries:
            character_id = entry.relative_path
            if character_id in {"Common", "filter.json"}:
                continue

            usd_path = f"{dh_root}/{character_id}/{character_id}.usd"
            record = {"id": character_id, "usd_path": usd_path}
            stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadNone)
            if stage is None:
                record["error"] = "failed to open stage"
                records.append(record)
                continue

            root = stage.GetDefaultPrim()
            record["default_prim"] = str(root.GetPath()) if root else None
            persona = root.GetChild("Persona") if root else None
            if not persona or not persona.IsValid():
                record["error"] = "missing Persona prim"
                records.append(record)
                continue

            record["persona"] = {
                attr.GetName(): _value(attr.Get())
                for attr in persona.GetAttributes()
                if attr.IsValid() and attr.IsDefined()
            }
            record["customization"] = {
                rel.GetName(): [str(path) for path in rel.GetTargets()]
                for rel in persona.GetRelationships()
                if rel.IsValid() and rel.HasAuthoredTargets()
            }
            records.append(record)

        if args.summary:
            for record in records:
                persona = record.get("persona", {})
                customization = record.get("customization", {})
                layers = {
                    name.rsplit(":", 1)[-1]: targets
                    for name, targets in customization.items()
                    if name.startswith("_customization:outfit:layer_")
                }
                compact = {
                    "id": record["id"],
                    "gender": persona.get("gender"),
                    "age": persona.get("age"),
                    "age_category": persona.get("age_category"),
                    "body_shape": persona.get("body_shape"),
                    "ethnicity": persona.get("ethnicity"),
                    "stature": persona.get("stature"),
                    "outfit": persona.get("outfit"),
                    "layers": layers,
                    "error": record.get("error"),
                }
                print("DH_SUMMARY=" + json.dumps(compact, sort_keys=True), flush=True)
        else:
            print("DH_INSPECTION_JSON=" + json.dumps(records, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
