"""Curated pedestrian character assets used by Arena's Isaac integration."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import yaml


@dataclass(frozen=True)
class CharacterAsset:
    """A stable scenario model name mapped to an Isaac asset-relative USD."""

    relative_usd: str
    source: str
    description: str
    identity_id: str | None = None
    variant: int = 0
    hide_prim_name_contains: tuple[str, ...] = ()


def _classic(folder: str, usd_name: str, description: str) -> CharacterAsset:
    return CharacterAsset(
        relative_usd=f"Isaac/People/Characters/{folder}/{usd_name}.usd",
        source="classic",
        description=description,
    )


def _dh(character_id: str, description: str) -> CharacterAsset:
    return CharacterAsset(
        relative_usd=(
            f"Isaac/People/DH_Characters/{character_id}/{character_id}.usd"
        ),
        source="dh",
        description=description,
    )


# This ordered allowlist is the first review pool.  Names are deliberately
# conservative: the Isaac 5.1 Persona metadata contains no business/student
# classification for the selected DH assets, so we do not invent one.
CHARACTER_ASSETS: dict[str, CharacterAsset] = {
    "female_medical_01": _classic(
        "original_female_adult_medical_01",
        "female_adult_medical_01",
        "adult female medical worker",
    ),
    "male_medical_01": _classic(
        "original_male_adult_medical_01",
        "male_adult_medical_01",
        "adult male medical worker",
    ),
    "female_business_01": _classic(
        "original_female_adult_business_02",
        "female_adult_business_02",
        "adult female business worker",
    ),
    "female_casual_01": _dh(
        "3809dbd8-06f1-11ef-b76b-f4b30194174e",
        "41-year-old female, casual cardigan and jeans",
    ),
    "female_senior_01": _dh(
        "a8777216-069a-11ef-ab41-b04f13dd02fd",
        "62-year-old female, casual cardigan and jeans",
    ),
    "male_casual_01": _dh(
        "d63572fc-0690-11ef-ad97-b04f13dd02fd",
        "41-year-old male, casual T-shirt and trousers",
    ),
    "male_casual_02": _dh(
        "c638e5d6-0697-11ef-b025-b04f13dd02fd",
        "50-year-old male, casual shirt and trousers",
    ),
    "male_senior_01": _dh(
        "f50e69fb-067b-11ef-871f-b04f13dd02fd",
        "64-year-old male, casual T-shirt and jeans",
    ),
    "female_young_adult_01": _dh(
        "ce9ae2ef-06ed-11ef-8c2e-f4b30194174e",
        "24-year-old female, workwear coverall",
    ),
    "male_young_adult_01": _dh(
        "02c80685-06e3-11ef-ae8a-f4b30194174e",
        "20-year-old male, workwear T-shirt, trousers, and high-vis vest",
    ),
}


# Preserve existing scenario files while moving new scenarios to the stable
# names above.  These aliases do not add assets to the ten-character review
# pool.
CHARACTER_ALIASES: dict[str, str] = {
    # Stable spellings used by older Arena scenario/model files.
    "actor1": "female_business_01",
    "biped_demo": "female_business_01",
    "biped_demo_meters": "female_business_01",
    "F_Business_02": "female_business_01",
    "F_Medical_01": "female_medical_01",
    "M_Medical_01": "male_medical_01",
    "female_adult_medical_01": "female_medical_01",
    "male_adult_medical_01": "male_medical_01",
    "female_adult_business_02": "female_business_01",
    # Asset-folder spellings seen in direct Isaac integrations.
    "original_female_adult_medical_01": "female_medical_01",
    "original_male_adult_medical_01": "male_medical_01",
    "original_female_adult_business_02": "female_business_01",
}


# Older scenarios in this repository still name police and construction
# characters.  Keep those explicit models loadable without including them in
# the new ten-person review pool.
LEGACY_CHARACTER_ASSETS: dict[str, CharacterAsset] = {
    "female_adult_police_01": _classic(
        "original_female_adult_police_01",
        "female_adult_police_01",
        "legacy female police officer",
    ),
    "female_adult_police_02": _classic(
        "original_female_adult_police_02",
        "female_adult_police_02",
        "legacy female police officer",
    ),
    "female_adult_police_03": _classic(
        "original_female_adult_police_03",
        "female_adult_police_03",
        "legacy female police officer",
    ),
    "male_adult_construction_01": _classic(
        "original_male_adult_construction_01",
        "male_adult_construction_01",
        "legacy male construction worker",
    ),
    "male_adult_construction_02": _classic(
        "original_male_adult_construction_02",
        "male_adult_construction_02",
        "legacy male construction worker",
    ),
    "male_adult_construction_03": _classic(
        "original_male_adult_construction_03",
        "male_adult_construction_03",
        "legacy male construction worker",
    ),
    "male_adult_construction_05": _classic(
        "original_male_adult_construction_05",
        "male_adult_construction_05",
        "legacy male construction worker",
    ),
    "male_adult_police_04": _classic(
        "original_male_adult_police_04",
        "male_adult_police_04",
        "legacy male police officer",
    ),
}


def _load_identity_catalog() -> dict:
    catalog_resource = files("pedestrian.config").joinpath(
        "character_identities.yaml"
    )
    with catalog_resource.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


IDENTITY_CATALOG = _load_identity_catalog()
CHARACTER_POOLS: dict[str, tuple[str, ...]] = {
    name: tuple(pool["identities"])
    for name, pool in IDENTITY_CATALOG["pools"].items()
}

_IDENTITY_TO_POOL = {
    identity_id: pool_name
    for pool_name, identity_ids in CHARACTER_POOLS.items()
    for identity_id in identity_ids
}
_APPEARANCE_PROFILES = IDENTITY_CATALOG.get("appearance_profiles", {})


def _identity_hidden_tokens(identity_id: str) -> tuple[str, ...]:
    pool = IDENTITY_CATALOG["pools"][_IDENTITY_TO_POOL[identity_id]]
    profile_name = pool.get("appearance_profile")
    if not profile_name:
        return ()
    profile = _APPEARANCE_PROFILES[profile_name]
    return tuple(token.lower() for token in profile["hide_prim_name_contains"])


CATALOG_CHARACTER_ASSETS: dict[str, CharacterAsset] = {}

for identity in IDENTITY_CATALOG["classic_identities"]:
    identity_id = identity["id"]
    CATALOG_CHARACTER_ASSETS[identity_id] = CharacterAsset(
        relative_usd=identity["asset_variants"][0],
        source="classic",
        description=f"{identity['role']} {identity['gender']} character",
        identity_id=identity_id,
        hide_prim_name_contains=_identity_hidden_tokens(identity_id),
    )

for identity in IDENTITY_CATALOG["dh_identities"]:
    identity_id = identity["id"]
    character_uuid = identity["uuid"]
    hidden_tokens = _identity_hidden_tokens(identity_id)
    CATALOG_CHARACTER_ASSETS[identity_id] = CharacterAsset(
        relative_usd=(
            "Isaac/People/DH_Characters/"
            f"{character_uuid}/{character_uuid}.usd"
        ),
        source="dh",
        description=(
            f"{identity['age']}-year-old {identity['gender']} "
            f"{identity['outfit_category']} character"
        ),
        identity_id=identity_id,
        hide_prim_name_contains=hidden_tokens,
    )
    for variant in range(1, identity["appearance_variants"]):
        variant_id = f"{character_uuid}_{variant}"
        model_name = f"{identity_id}_v{variant:02d}"
        CATALOG_CHARACTER_ASSETS[model_name] = CharacterAsset(
            relative_usd=(
                "Isaac/People/DH_Characters_Extended/"
                f"{variant_id}/{variant_id}.usd"
            ),
            source="dh_extended",
            description=(
                f"variant {variant} of {identity_id}"
            ),
            identity_id=identity_id,
            variant=variant,
            hide_prim_name_contains=hidden_tokens,
        )


def resolve_character_asset(model_name: str) -> tuple[str, CharacterAsset]:
    """Return canonical name and asset, raising clearly for unknown models."""
    canonical_name = CHARACTER_ALIASES.get(model_name, model_name)
    try:
        asset = CHARACTER_ASSETS.get(canonical_name)
        if asset is None:
            asset = LEGACY_CHARACTER_ASSETS.get(canonical_name)
        if asset is None:
            asset = CATALOG_CHARACTER_ASSETS[canonical_name]
        return canonical_name, asset
    except KeyError as exc:
        available = ", ".join(
            (*CHARACTER_POOLS, *CHARACTER_ASSETS, *LEGACY_CHARACTER_ASSETS)
        )
        raise ValueError(
            f"Unknown pedestrian model '{model_name}'. Available models: {available}"
        ) from exc
