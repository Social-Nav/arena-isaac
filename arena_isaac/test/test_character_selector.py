"""Tests for scenario-facing pedestrian category resolution."""

from pedestrian.simulator.logic.people.character_assets import (
    CHARACTER_POOLS,
    resolve_character_asset,
)
from pedestrian.simulator.logic.people.character_selector import (
    CharacterPoolSelector,
)


def test_reviewed_pool_sizes():
    assert {name: len(identities) for name, identities in CHARACTER_POOLS.items()} == {
        "pedestrian": 22,
        "doctor": 2,
        "police": 4,
        "construction_worker": 4,
    }


def test_pool_selection_is_reproducible_for_a_seed():
    first = CharacterPoolSelector(seed=1234)
    second = CharacterPoolSelector(seed=1234)
    assert [first.resolve(f"person_{i}", "pedestrian") for i in range(30)] == [
        second.resolve(f"person_{i}", "pedestrian") for i in range(30)
    ]


def test_pool_uses_every_identity_before_repeating():
    selector = CharacterPoolSelector(seed=9)
    resolved = [selector.resolve(f"person_{i}", "pedestrian") for i in range(22)]
    identities = [resolve_character_asset(model)[1].identity_id for model in resolved]
    assert len(set(identities)) == 22


def test_pool_assignment_is_cached_by_entity():
    selector = CharacterPoolSelector(seed=5)
    first = selector.resolve("visitor_01", "pedestrian")
    selector.resolve("visitor_02", "pedestrian")
    assert selector.resolve("visitor_01", "pedestrian") == first


def test_exact_model_bypasses_pool_randomization():
    selector = CharacterPoolSelector(seed=5)
    assert selector.resolve("doctor_01", "classic_female_medical_01") == (
        "classic_female_medical_01"
    )
    assert selector.resolve("visitor_01", "dh_3809dbd8_v07") == (
        "dh_3809dbd8_v07"
    )


def test_all_dh_pedestrian_assets_apply_general_profile():
    for identity_id in CHARACTER_POOLS["pedestrian"]:
        _, asset = resolve_character_asset(identity_id)
        if asset.source.startswith("dh"):
            assert asset.hide_prim_name_contains == (
                "helmet_safety",
                "vest_highvis",
            )
