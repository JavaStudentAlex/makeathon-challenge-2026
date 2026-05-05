from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_seed_program_accepts_validation_data_dir(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    initial = _load_module("local_shinka_initial", repo_root / "shinka" / "initial.py")
    validation_data_dir = tmp_path / "validation"
    validation_data_dir.mkdir()

    prediction = initial.run_experiment(validation_data_dir)

    assert prediction["type"] == "FeatureCollection"
    assert prediction["features"]


def test_initial_seed_program_has_30_minute_training_timeout() -> None:
    repo_root = Path(__file__).parents[1]
    initial = _load_module("local_shinka_initial", repo_root / "shinka" / "initial.py")

    assert initial.TRAINING_TIMEOUT_SECONDS == 30 * 60
