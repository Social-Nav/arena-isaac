"""Tests for the stable scenario-facing pedestrian model registry."""

import pytest

from pedestrian.simulator.logic.people.character_assets import (
    CHARACTER_ASSETS,
    resolve_character_asset,
)


def test_review_pool_contains_exactly_ten_models():
    assert len(CHARACTER_ASSETS) == 10
    assert len(set(CHARACTER_ASSETS)) == 10


def test_review_pool_supports_classic_and_dh_assets():
    sources = {asset.source for asset in CHARACTER_ASSETS.values()}
    assert sources == {"classic", "dh"}
    assert all(asset.relative_usd.endswith(".usd") for asset in CHARACTER_ASSETS.values())


def test_existing_scenario_name_resolves_to_canonical_model():
    canonical, asset = resolve_character_asset("female_adult_medical_01")
    assert canonical == "female_medical_01"
    assert "original_female_adult_medical_01" in asset.relative_usd


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name"),
    [
        ("actor1", "female_business_01"),
        ("biped_demo_meters", "female_business_01"),
        ("F_Medical_01", "female_medical_01"),
    ],
)
def test_legacy_model_names_are_deterministic(legacy_name, canonical_name):
    resolved_name, _ = resolve_character_asset(legacy_name)
    assert resolved_name == canonical_name


def test_unknown_model_fails_with_available_names():
    with pytest.raises(ValueError, match="Unknown pedestrian model") as exc_info:
        resolve_character_asset("not_a_character")
    assert "female_medical_01" in str(exc_info.value)
