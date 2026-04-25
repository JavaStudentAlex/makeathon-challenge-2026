import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

import simple_ensembles
import simple_ensembles.balanced_fusion as balanced_fusion
import simple_ensembles.high_recall_fusion as high_recall_fusion
import simple_ensembles.top_ranked_fusion as top_ranked_fusion
import stats_models.runner as runner
from stats_models.runner import _default_output_dir

PROMOTED_MODULES = (
    pytest.param(
        "simple_ensembles.top_ranked_fusion",
        top_ranked_fusion,
        Path("submission/top_ranked_fusion"),
        id="top_ranked_fusion",
    ),
    pytest.param(
        "simple_ensembles.balanced_fusion",
        balanced_fusion,
        Path("submission/balanced_fusion"),
        id="balanced_fusion",
    ),
    pytest.param(
        "simple_ensembles.high_recall_fusion",
        high_recall_fusion,
        Path("submission/high_recall_fusion"),
        id="high_recall_fusion",
    ),
)


def test_package_exports_expose_all_promoted_modules() -> None:
    from simple_ensembles import (
        balanced_fusion as package_balanced_fusion,
        high_recall_fusion as package_high_recall_fusion,
        top_ranked_fusion as package_top_ranked_fusion,
    )

    assert set(simple_ensembles.__all__) == {
        "top_ranked_fusion",
        "balanced_fusion",
        "high_recall_fusion",
    }
    assert package_top_ranked_fusion is top_ranked_fusion
    assert package_balanced_fusion is balanced_fusion
    assert package_high_recall_fusion is high_recall_fusion


@pytest.mark.parametrize(
    ("module_name", "imported_module", "expected_output_dir"),
    PROMOTED_MODULES,
)
def test_default_output_dir_uses_promoted_module_name(
    module_name: str,
    imported_module: ModuleType,
    expected_output_dir: Path,
) -> None:
    assert imported_module.__spec__ is not None
    assert imported_module.__spec__.name == module_name
    assert _default_output_dir(imported_module) == expected_output_dir


@pytest.mark.parametrize(
    ("module_name", "imported_module", "expected_output_dir"),
    PROMOTED_MODULES,
)
def test_python_m_shim_delegates_to_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    imported_module: ModuleType,
    expected_output_dir: Path,
) -> None:
    del imported_module

    captured: dict[str, object] = {}

    def fake_generate_submission(
        model_module,
        data_root,
        output_dir,
        *,
        split="test",
        tiles=None,
        threshold=0.52,
        min_area_ha=0.5,
        feature_builder=None,
    ):
        captured["model_module"] = model_module
        captured["data_root"] = data_root
        captured["output_dir"] = output_dir
        captured["split"] = split
        captured["tiles"] = tiles
        captured["threshold"] = threshold
        captured["min_area_ha"] = min_area_ha
        captured["feature_builder"] = feature_builder
        return tmp_path / "submission.geojson", tmp_path / "manifest.json"

    monkeypatch.setattr(runner, "generate_submission", fake_generate_submission)

    data_root = tmp_path / "does-not-exist"
    output_dir = tmp_path / f"{expected_output_dir.name}-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module_name,
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--split",
            "holdout",
            "--tiles",
            "tile_a",
            "tile_b",
            "--threshold",
            "0.61",
            "--min-area-ha",
            "1.25",
        ],
    )
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)

    assert excinfo.value.code == 0
    assert captured["data_root"] == data_root
    assert captured["output_dir"] == output_dir
    assert captured["split"] == "holdout"
    assert captured["tiles"] == ["tile_a", "tile_b"]
    assert captured["threshold"] == pytest.approx(0.61)
    assert captured["min_area_ha"] == pytest.approx(1.25)

    captured_module = captured["model_module"]
    assert isinstance(captured_module, ModuleType)
    assert callable(getattr(captured_module, "run_experiment", None))
    assert captured_module.__spec__ is not None
    assert captured_module.__spec__.name == module_name
    assert _default_output_dir(captured_module) == expected_output_dir
