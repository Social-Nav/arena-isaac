"""Structural checks for the reviewed Isaac People identity inventory."""

from pathlib import Path

import yaml


CATALOG_PATH = (
    Path(__file__).parents[1]
    / "pedestrian"
    / "config"
    / "character_identities.yaml"
)


def _catalog():
    with CATALOG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_identity_counts_match_reviewed_inventory():
    catalog = _catalog()
    assert len(catalog["classic_identities"]) == 11
    assert len(catalog["dh_identities"]) == 21
    assert catalog["summary"]["total_identities"] == 32


def test_identity_keys_and_dh_uuids_are_unique():
    catalog = _catalog()
    identities = catalog["classic_identities"] + catalog["dh_identities"]
    assert len({identity["id"] for identity in identities}) == 32
    assert len({identity["uuid"] for identity in catalog["dh_identities"]}) == 21


def test_every_dh_identity_has_eleven_appearances():
    catalog = _catalog()
    assert all(
        identity["appearance_variants"] == 11
        for identity in catalog["dh_identities"]
    )


def test_scenario_pools_partition_all_identities():
    catalog = _catalog()
    identities = catalog["classic_identities"] + catalog["dh_identities"]
    identity_ids = {identity["id"] for identity in identities}
    pooled_ids = [
        identity_id
        for pool in catalog["pools"].values()
        for identity_id in pool["identities"]
    ]
    assert len(pooled_ids) == 32
    assert len(set(pooled_ids)) == 32
    assert set(pooled_ids) == identity_ids


def test_scenario_pool_sizes():
    pools = _catalog()["pools"]
    assert {name: len(pool["identities"]) for name, pool in pools.items()} == {
        "pedestrian": 22,
        "doctor": 2,
        "police": 4,
        "construction_worker": 4,
    }
