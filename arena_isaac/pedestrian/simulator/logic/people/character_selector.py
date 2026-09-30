"""Reproducible pool-to-concrete-character selection for Isaac pedestrians."""

from __future__ import annotations

from collections import deque
import os
import random
import secrets
from threading import Lock

from pedestrian.simulator.logic.people.character_assets import (
    CATALOG_CHARACTER_ASSETS,
    CHARACTER_POOLS,
    resolve_character_asset,
)


MODEL_SEED_ENV = "ARENA_PEDESTRIAN_MODEL_SEED"


class CharacterPoolSelector:
    """Resolve scenario pool names while avoiding repeats within pool cycles."""

    def __init__(self, seed: int | None = None):
        if seed is None:
            configured_seed = os.environ.get(MODEL_SEED_ENV)
            seed = (
                int(configured_seed)
                if configured_seed is not None
                else secrets.randbits(32)
            )
        self.seed = seed
        self._rng = random.Random(seed)
        self._identity_decks: dict[str, deque[str]] = {}
        self._variant_decks: dict[str, deque[int]] = {}
        self._assignments: dict[tuple[str, str], str] = {}
        self._lock = Lock()

    def _refill_identity_deck(self, pool_name: str) -> deque[str]:
        identities = list(CHARACTER_POOLS[pool_name])
        self._rng.shuffle(identities)
        deck = deque(identities)
        self._identity_decks[pool_name] = deck
        return deck

    def _refill_variant_deck(self, identity_id: str) -> deque[int]:
        asset = CATALOG_CHARACTER_ASSETS[identity_id]
        if not asset.source.startswith("dh"):
            variants = [0]
        else:
            variants = list(range(11))
            self._rng.shuffle(variants)
        deck = deque(variants)
        self._variant_decks[identity_id] = deck
        return deck

    def _select_from_pool(self, pool_name: str) -> str:
        identity_deck = self._identity_decks.get(pool_name)
        if not identity_deck:
            identity_deck = self._refill_identity_deck(pool_name)
        identity_id = identity_deck.popleft()

        variant_deck = self._variant_decks.get(identity_id)
        if not variant_deck:
            variant_deck = self._refill_variant_deck(identity_id)
        variant = variant_deck.popleft()
        if variant == 0:
            return identity_id
        return f"{identity_id}_v{variant:02d}"

    def resolve(self, entity_name: str, requested_model: str) -> str:
        """Return one concrete model, caching pool choices by entity name."""
        with self._lock:
            if requested_model not in CHARACTER_POOLS:
                canonical_name, _ = resolve_character_asset(requested_model)
                return canonical_name

            assignment_key = (entity_name, requested_model)
            resolved = self._assignments.get(assignment_key)
            if resolved is None:
                resolved = self._select_from_pool(requested_model)
                self._assignments[assignment_key] = resolved
            return resolved
