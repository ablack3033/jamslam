"""Shared fixtures.

The synthetic renders are session-scoped and deliberately short: a full-length
jam takes ~15s to analyze, which is fine for the eval corpus but too slow to sit
inside a unit test suite. Everything here is seeded, so tests are deterministic.
"""

from __future__ import annotations

import dataclasses

import pytest

from fiddle.audio import preprocess
from fiddle.corpus.library import get_tune
from fiddle.corpus.synth import render_tune


def _short(slug: str, performance=("A", "A", "B", "B")):
    tune = get_tune(slug)
    return dataclasses.replace(tune, performance=performance)


@pytest.fixture(scope="session")
def clean_render():
    return render_tune(_short("soldiers_joy"), difficulty="clean", seed=7)


@pytest.fixture(scope="session")
def clean_result(clean_render):
    from fiddle.pipeline import transcribe

    audio, truth = clean_render
    return transcribe(preprocess(audio)), truth
