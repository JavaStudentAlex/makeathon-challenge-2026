from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_seed_program_produces_evaluable_metric() -> None:
    repo_root = Path(__file__).parents[1]
    initial = _load_module("local_shinka_initial", repo_root / "shinka" / "initial.py")
    evaluate = _load_module(
        "local_shinka_evaluate",
        repo_root / "shinka" / "evaluate.py",
    )

    prediction = initial.run_experiment()
    metrics = evaluate.calculate_scoring_metrics(
        prediction,
        repo_root / "shinka" / "smoke_ground_truth.geojson",
    )

    assert prediction["type"] == "FeatureCollection"
    assert prediction["features"]
    assert metrics["combined_score"] == pytest.approx(1.0)
    assert metrics["year_accuracy"] == pytest.approx(1.0)


def test_initial_seed_program_has_30_minute_training_timeout() -> None:
    repo_root = Path(__file__).parents[1]
    initial = _load_module("local_shinka_initial", repo_root / "shinka" / "initial.py")

    assert initial.TRAINING_TIMEOUT_SECONDS == 30 * 60
