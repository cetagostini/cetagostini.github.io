"""Tests for cetagostini.style."""
import numpy as np
import pytest

from cetagostini.style import COLORS, PALETTE, make_rng


def test_colors_has_required_keys():
    required = {"primary", "secondary", "accent", "bg", "ink", "ink_muted", "green_strong", "line"}
    assert required.issubset(COLORS.keys())


def test_palette_length():
    assert len(PALETTE) >= 6


def test_make_rng_deterministic():
    seed1, rng1 = make_rng("test phrase")
    seed2, rng2 = make_rng("test phrase")
    assert seed1 == seed2
    assert rng1.random() == rng2.random()


def test_make_rng_different_phrase():
    seed1, _ = make_rng("phrase one")
    seed2, _ = make_rng("phrase two")
    assert seed1 != seed2


def test_make_rng_seed_is_sum_of_ord():
    phrase = "hello world"
    seed, _ = make_rng(phrase)
    assert seed == sum(map(ord, phrase))
