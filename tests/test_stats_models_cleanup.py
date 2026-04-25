import importlib
from pathlib import Path

import pytest

PROMOTED_MODULES = (
    "stats_models.eligibility_and_patch_votes",
    "stats_models.spatial_consensus_and_time_median",
    "stats_models.spatial_consensus_and_timing",
)


@pytest.mark.parametrize("module_name", PROMOTED_MODULES)
def test_promoted_stats_model_modules_expose_callable_run_experiment(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)

    assert callable(getattr(module, "run_experiment", None))


def test_retired_shinka_generator_is_absent() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert not Path("shinka/generate_test_submission.py").exists()
    assert not (repo_root / "shinka" / "generate_test_submission.py").exists()


def test_stats_models_source_does_not_reference_retired_generator() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_paths = sorted((repo_root / "stats_models").glob("*.py"))

    assert source_paths

    for source_path in source_paths:
        assert "generate_test_submission" not in source_path.read_text(
            encoding="utf-8"
        ), f"{source_path.relative_to(repo_root)} should not reference the retired generator"
