# flake8: noqa
"""Generated Shinka method snapshot: contextual_ensemble_cleanup.

Initial Shinka seed program trained from training labels only.

``run_experiment`` trains from the hardcoded training split under
``data/makeathon-challenge/training`` and returns the trained model object.
``run_inference`` applies that model to an unlabeled prediction input directory
and returns GeoJSON predictions.
"""

# EVOLVE-BLOCK-START
from __future__ import annotations

import signal
import threading
import warnings
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from time import monotonic
from typing import Any, NamedTuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import reproject, transform_geom
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_OUTPUT_CRS = "EPSG:4326"


def _find_repo_root() -> Path:
    """Find the repo root even after Shinka copies this file into results/."""

    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "shinka"
            ).is_dir():
                return candidate
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _find_repo_root()
TRAINING_DATA_DIR = REPO_ROOT / "data" / "makeathon-challenge" / "training"
RANDOM_STATE = 7
TRAINING_TIMEOUT_SECONDS = 30 * 60
AEF_BAND_INDEXES = tuple(range(1, 17))
FEATURE_NAMES: tuple[str, ...] = (
    "forest_2020",
    "ndvi_delta",
    "nbr_delta",
    "ndmi_delta",
    "evi_delta",
    "ndre_delta",
    "bsi_delta",
    "ndwi_delta",
    "vv_delta",
    "vv_cv_delta",
    "aef_shift",
    "aef_abs_mean",
    "aef_abs_std",
    "aef_pos_frac",
    "aef_neg_frac",
    "local_shift_mean",
    "local_shift_std",
    "local_shift_max",
    "local_shift_contrast",
    "alert_consensus",
    "seasonal_drop",
    "ndvi_zscore",
    "nbr_zscore",
    "ndmi_zscore",
    "bsi_zscore",
    "vv_zscore",
    "water",
    "crop",
    "urban",
    "cloud",
    *(f"aef_delta_{band:02d}" for band in AEF_BAND_INDEXES),
)
PREDICTION_THRESHOLD = 0.62
PREDICT_DOWNSAMPLE_FACTOR = 8
MAX_POSITIVE_SAMPLES_PER_RASTER = 512
MAX_NEGATIVE_SAMPLES_PER_RASTER = 512
MIN_TRAINING_EXAMPLES_PER_CLASS = 1


class EnsembleMember(NamedTuple):
    name: str
    model: Any
    weight: float


@contextmanager
def _wall_time_limit(seconds: int):
    """Bound candidate training so evolved programs cannot hang indefinitely."""

    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_remaining = signal.alarm(0)
    started_at = monotonic()
    effective_seconds = (
        min(seconds, previous_remaining) if previous_remaining > 0 else seconds
    )

    def _raise_timeout(_signum, _frame) -> None:
        if previous_remaining > 0 and previous_remaining <= seconds:
            if callable(previous_handler):
                previous_handler(_signum, _frame)
            raise TimeoutError(
                f"outer execution exceeded {previous_remaining} seconds"
            )
        raise TimeoutError(
            f"training exceeded {seconds} seconds; keep training bounded"
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(effective_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining > 0:
            elapsed = max(0, int(monotonic() - started_at))
            signal.alarm(max(1, previous_remaining - elapsed))


def _row(**overrides: float) -> dict[str, float]:
    """Create one normalized feature row with stable-forest defaults."""

    values = {
        "forest_2020": 1.0,
        "ndvi_delta": 0.0,
        "nbr_delta": 0.0,
        "ndmi_delta": 0.0,
        "evi_delta": 0.0,
        "ndre_delta": 0.0,
        "bsi_delta": 0.0,
        "ndwi_delta": 0.0,
        "vv_delta": 0.0,
        "vv_cv_delta": 0.0,
        "aef_shift": 0.0,
        "aef_abs_mean": 0.0,
        "aef_abs_std": 0.0,
        "aef_pos_frac": 0.0,
        "aef_neg_frac": 0.0,
        "local_shift_mean": 0.0,
        "local_shift_std": 0.0,
        "local_shift_max": 0.0,
        "local_shift_contrast": 0.0,
        "alert_consensus": 0.0,
        "seasonal_drop": 0.0,
        "ndvi_zscore": 0.0,
        "nbr_zscore": 0.0,
        "ndmi_zscore": 0.0,
        "bsi_zscore": 0.0,
        "vv_zscore": 0.0,
        "water": 0.0,
        "crop": 0.0,
        "urban": 0.0,
        "cloud": 0.0,
    }
    values.update({f"aef_delta_{band:02d}": 0.0 for band in AEF_BAND_INDEXES})
    values.update(overrides)
    return values


def _matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float
    )


def _build_training_examples(
    training_data_dir: str | Path = TRAINING_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised examples from training features and training labels."""

    training_data_dir = Path(training_data_dir)
    if not training_data_dir.is_dir():
        raise FileNotFoundError(
            f"Training data directory not found: {training_data_dir}"
        )

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    rng = np.random.default_rng(RANDOM_STATE)
    for tile_id in _training_tile_ids(training_data_dir):
        baseline_path = _aef_path(training_data_dir, tile_id, 2020)
        if baseline_path is None:
            continue
        label_targets: tuple[np.ndarray, np.ndarray] | None = None
        for year, current_path in _aef_year_paths(training_data_dir, tile_id):
            if year <= 2020:
                continue
            feature_grid, profile = _aef_feature_grid(baseline_path, current_path)
            context_maps = _context_maps(feature_grid)
            if label_targets is None:
                label_targets = _training_label_targets(
                    training_data_dir,
                    tile_id,
                    profile,
                )
            label_years, _label_time_steps = label_targets
            positive_mask = label_years == year
            negative_mask = label_years == 0
            _append_sampled_examples(
                rows,
                labels,
                feature_grid,
                context_maps,
                positive_mask,
                1,
                MAX_POSITIVE_SAMPLES_PER_RASTER,
                rng,
            )
            _append_sampled_examples(
                rows,
                labels,
                feature_grid,
                context_maps,
                negative_mask,
                0,
                MAX_NEGATIVE_SAMPLES_PER_RASTER,
                rng,
            )

    y_train = np.asarray(labels, dtype=int)
    if (
        y_train.size == 0
        or np.count_nonzero(y_train == 1) < MIN_TRAINING_EXAMPLES_PER_CLASS
        or np.count_nonzero(y_train == 0) < MIN_TRAINING_EXAMPLES_PER_CLASS
    ):
        raise ValueError(
            "Training labels did not produce both positive and negative examples"
        )
    return _matrix(rows), y_train


def _fit_ensemble(x_train: np.ndarray, y_train: np.ndarray) -> list[EnsembleMember]:
    """Fit compact tree and linear members with safe fallbacks."""

    members: list[EnsembleMember] = []

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=18,
            max_depth=2,
            learning_rate=0.25,
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=1.0,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        xgb.fit(x_train, y_train)
        members.append(EnsembleMember("xgboost", xgb, 0.28))
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            n_estimators=24,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.20,
            objective="binary",
            min_child_samples=1,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            random_state=RANDOM_STATE + 1,
            n_jobs=1,
            verbosity=-1,
            force_col_wise=True,
        )
        lgbm.fit(x_train, y_train)
        members.append(EnsembleMember("lightgbm", lgbm, 0.28))
    except Exception:
        pass

    try:
        extratrees = ExtraTreesClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE + 2,
            n_jobs=1,
        )
        extratrees.fit(x_train, y_train)
        members.append(EnsembleMember("extratrees", extratrees, 0.24))
    except Exception:
        pass

    try:
        logreg = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=RANDOM_STATE + 3,
                solver="lbfgs",
            ),
        )
        logreg.fit(x_train, y_train)
        members.append(EnsembleMember("logreg", logreg, 0.20))
    except Exception:
        pass

    return members


def _positive_signal(value: np.ndarray | float, scale: float) -> np.ndarray | float:
    return np.clip(np.asarray(value, dtype=float) / scale, 0.0, 1.0)


def _drop_signal(value: np.ndarray | float, scale: float) -> np.ndarray | float:
    return np.clip(-np.asarray(value, dtype=float) / scale, 0.0, 1.0)


def _domain_prior_probability(x_values: np.ndarray) -> np.ndarray:
    """Fixed physics-inspired prior blended with learned model probabilities."""

    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    optical_drop = np.mean(
        np.stack(
            [
                _drop_signal(x_values[:, idx["ndvi_delta"]], 0.80),
                _drop_signal(x_values[:, idx["nbr_delta"]], 0.95),
                _drop_signal(x_values[:, idx["ndmi_delta"]], 0.70),
                _drop_signal(x_values[:, idx["evi_delta"]], 1.00),
                _drop_signal(x_values[:, idx["ndre_delta"]], 0.70),
            ],
            axis=0,
        ),
        axis=0,
    )
    exposure = _positive_signal(x_values[:, idx["bsi_delta"]], 0.70)
    sar_drop = np.mean(
        np.stack(
            [
                _drop_signal(x_values[:, idx["vv_delta"]], 0.30),
                _positive_signal(x_values[:, idx["vv_cv_delta"]], 0.35),
            ],
            axis=0,
        ),
        axis=0,
    )
    semantic_shift = _positive_signal(x_values[:, idx["aef_shift"]], 0.85)
    local_support = np.mean(
        np.stack(
            [
                _positive_signal(x_values[:, idx["local_shift_mean"]], 1.00),
                _positive_signal(x_values[:, idx["local_shift_max"]], 1.35),
                _positive_signal(x_values[:, idx["local_shift_contrast"]], 0.45),
            ],
            axis=0,
        ),
        axis=0,
    )
    band_dispersion = _positive_signal(x_values[:, idx["aef_abs_std"]], 0.60)
    alert = np.clip(x_values[:, idx["alert_consensus"]], 0.0, 1.0)
    forest = np.clip(x_values[:, idx["forest_2020"]], 0.0, 1.0)
    negative_context = np.maximum.reduce(
        [
            np.clip(x_values[:, idx["water"]], 0.0, 1.0),
            np.clip(x_values[:, idx["crop"]], 0.0, 1.0),
            np.clip(x_values[:, idx["urban"]], 0.0, 1.0),
            np.clip(x_values[:, idx["cloud"]], 0.0, 1.0),
            _positive_signal(x_values[:, idx["seasonal_drop"]], 0.90),
            _positive_signal(x_values[:, idx["ndwi_delta"]], 0.90),
        ]
    )

    probability = (
        0.25 * optical_drop
        + 0.15 * exposure
        + 0.15 * sar_drop
        + 0.15 * semantic_shift
        + 0.10 * alert
        + 0.08 * local_support
        + 0.06 * band_dispersion
        - 0.40 * negative_context
    )
    return np.clip(probability, 0.0, 1.0) * (forest >= 0.5)


def _predict_ensemble_probability(
    members: list[EnsembleMember],
    x_values: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    weighted = np.zeros(x_values.shape[0], dtype=float)
    total_weight = 0.0
    names: list[str] = []

    for member in members:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            probabilities = member.model.predict_proba(x_values)[:, 1]
        weighted += float(member.weight) * probabilities
        total_weight += float(member.weight)
        names.append(member.name)

    if not members:
        return _domain_prior_probability(x_values), ["domain_prior"]

    model_probability = weighted / max(total_weight, 1e-12)
    prior_probability = _domain_prior_probability(x_values)
    return (
        np.clip(0.72 * model_probability + 0.28 * prior_probability, 0.0, 1.0),
        names,
    )


def _training_tile_ids(training_data_dir: Path) -> list[str]:
    label_dir = training_data_dir / "labels" / "radd"
    return sorted(
        path.name.removeprefix("radd_").removesuffix("_labels.tif")
        for path in label_dir.glob("radd_*_labels.tif")
    )


def _prediction_tile_ids(prediction_data_dir: Path) -> list[str]:
    aef_root = prediction_data_dir / "aef-embeddings"
    return sorted(
        {"_".join(path.stem.split("_")[:-1]) for path in aef_root.glob("*.tiff")}
    )


def _aef_path(data_dir: Path, tile_id: str, year: int) -> Path | None:
    path = data_dir / "aef-embeddings" / f"{tile_id}_{year}.tiff"
    return path if path.exists() else None


def _aef_year_paths(data_dir: Path, tile_id: str) -> list[tuple[int, Path]]:
    paths: list[tuple[int, Path]] = []
    for path in sorted((data_dir / "aef-embeddings").glob(f"{tile_id}_*.tiff")):
        try:
            year = int(path.stem.rsplit("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        paths.append((year, path))
    return paths


def _aef_feature_grid(
    baseline_path: Path,
    current_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    with (
        rasterio.open(baseline_path) as baseline_src,
        rasterio.open(current_path) as current_src,
    ):
        height = max(current_src.height // PREDICT_DOWNSAMPLE_FACTOR, 1)
        width = max(current_src.width // PREDICT_DOWNSAMPLE_FACTOR, 1)
        baseline = baseline_src.read(
            indexes=AEF_BAND_INDEXES,
            out_shape=(len(AEF_BAND_INDEXES), height, width),
            resampling=Resampling.average,
        ).astype(np.float32)
        current = current_src.read(
            indexes=AEF_BAND_INDEXES,
            out_shape=(len(AEF_BAND_INDEXES), height, width),
            resampling=Resampling.average,
        ).astype(np.float32)
        transform = current_src.transform * current_src.transform.scale(
            current_src.width / width,
            current_src.height / height,
        )
        profile = {
            "height": height,
            "width": width,
            "transform": transform,
            "crs": current_src.crs,
        }

    delta = np.nan_to_num(current - baseline, nan=0.0, posinf=0.0, neginf=0.0)
    return delta, profile


def _context_maps(feature_grid: np.ndarray) -> dict[str, np.ndarray]:
    shift_map = np.linalg.norm(feature_grid, axis=0)
    abs_grid = np.abs(feature_grid)
    abs_mean_map = abs_grid.mean(axis=0)
    abs_std_map = abs_grid.std(axis=0)
    pos_frac_map = (feature_grid > 0).mean(axis=0)
    neg_frac_map = (feature_grid < 0).mean(axis=0)

    padded_shift = np.pad(shift_map, 1, mode="edge")
    neighbourhood = (
        padded_shift[:-2, :-2],
        padded_shift[:-2, 1:-1],
        padded_shift[:-2, 2:],
        padded_shift[1:-1, :-2],
        padded_shift[1:-1, 1:-1],
        padded_shift[1:-1, 2:],
        padded_shift[2:, :-2],
        padded_shift[2:, 1:-1],
        padded_shift[2:, 2:],
    )
    local_shift_mean = sum(neighbourhood) / 9.0
    padded_shift_sq = np.pad(np.square(shift_map), 1, mode="edge")
    neighbourhood_sq = (
        padded_shift_sq[:-2, :-2],
        padded_shift_sq[:-2, 1:-1],
        padded_shift_sq[:-2, 2:],
        padded_shift_sq[1:-1, :-2],
        padded_shift_sq[1:-1, 1:-1],
        padded_shift_sq[1:-1, 2:],
        padded_shift_sq[2:, :-2],
        padded_shift_sq[2:, 1:-1],
        padded_shift_sq[2:, 2:],
    )
    local_shift_std = np.sqrt(
        np.maximum(sum(neighbourhood_sq) / 9.0 - np.square(local_shift_mean), 0.0)
    )
    local_shift_max = np.maximum.reduce(neighbourhood)
    local_shift_contrast = shift_map - local_shift_mean

    return {
        "shift_map": shift_map,
        "aef_abs_mean_map": abs_mean_map,
        "aef_abs_std_map": abs_std_map,
        "aef_pos_frac_map": pos_frac_map,
        "aef_neg_frac_map": neg_frac_map,
        "local_shift_mean_map": local_shift_mean,
        "local_shift_std_map": local_shift_std,
        "local_shift_max_map": local_shift_max,
        "local_shift_contrast_map": local_shift_contrast,
    }


def _feature_row_from_grid(
    feature_grid: np.ndarray,
    row: int,
    col: int,
    context_maps: dict[str, np.ndarray],
) -> dict[str, float]:
    deltas = feature_grid[:, row, col]
    values = _row(
        forest_2020=1.0,
        aef_shift=float(context_maps["shift_map"][row, col]),
        aef_abs_mean=float(context_maps["aef_abs_mean_map"][row, col]),
        aef_abs_std=float(context_maps["aef_abs_std_map"][row, col]),
        aef_pos_frac=float(context_maps["aef_pos_frac_map"][row, col]),
        aef_neg_frac=float(context_maps["aef_neg_frac_map"][row, col]),
        local_shift_mean=float(context_maps["local_shift_mean_map"][row, col]),
        local_shift_std=float(context_maps["local_shift_std_map"][row, col]),
        local_shift_max=float(context_maps["local_shift_max_map"][row, col]),
        local_shift_contrast=float(context_maps["local_shift_contrast_map"][row, col]),
        ndvi_delta=float(-deltas[0]) if deltas.size > 0 else 0.0,
        nbr_delta=float(-deltas[1]) if deltas.size > 1 else 0.0,
        ndmi_delta=float(-deltas[2]) if deltas.size > 2 else 0.0,
        bsi_delta=float(deltas[3]) if deltas.size > 3 else 0.0,
    )
    for index, band in enumerate(AEF_BAND_INDEXES):
        values[f"aef_delta_{band:02d}"] = float(deltas[index])
    return values


def _append_sampled_examples(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_grid: np.ndarray,
    context_maps: dict[str, np.ndarray],
    mask: np.ndarray,
    label: int,
    max_samples: int,
    rng: np.random.Generator,
) -> None:
    locations = np.argwhere(mask)
    if locations.size == 0:
        return
    count = min(max_samples, len(locations))
    selected_indexes = rng.choice(len(locations), size=count, replace=False)
    for row, col in locations[selected_indexes]:
        rows.append(
            _feature_row_from_grid(feature_grid, int(row), int(col), context_maps)
        )
        labels.append(label)


def _training_label_targets(
    training_data_dir: Path,
    tile_id: str,
    reference_profile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    target_years = np.zeros(
        (reference_profile["height"], reference_profile["width"]),
        dtype=np.int32,
    )
    target_time_steps = np.zeros_like(target_years)
    label_root = training_data_dir / "labels"
    _merge_radd_targets(
        label_root / "radd" / f"radd_{tile_id}_labels.tif",
        target_years,
        target_time_steps,
        reference_profile,
    )
    _merge_gladl_targets(
        label_root / "gladl",
        tile_id,
        target_years,
        target_time_steps,
        reference_profile,
    )
    _merge_glads2_targets(
        label_root / "glads2" / f"glads2_{tile_id}_alert.tif",
        label_root / "glads2" / f"glads2_{tile_id}_alertDate.tif",
        target_years,
        target_time_steps,
        reference_profile,
    )
    return target_years, target_time_steps


def _merge_radd_targets(
    path: Path,
    target_years: np.ndarray,
    target_time_steps: np.ndarray,
    reference_profile: dict[str, Any],
) -> None:
    if not path.exists():
        return
    with rasterio.open(path) as src:
        raw = src.read(1)
        positive = raw > 0
        time_steps, years = _radd_time_arrays(raw)
        _merge_target_arrays(
            positive,
            years,
            time_steps,
            src,
            target_years,
            target_time_steps,
            reference_profile,
        )


def _merge_gladl_targets(
    label_dir: Path,
    tile_id: str,
    target_years: np.ndarray,
    target_time_steps: np.ndarray,
    reference_profile: dict[str, Any],
) -> None:
    for alert_path in sorted(label_dir.glob(f"gladl_{tile_id}_alert[0-9][0-9].tif")):
        stem = alert_path.name.removeprefix("gladl_").removesuffix(".tif")
        _tile_id, yy_text = stem.rsplit("_alert", maxsplit=1)
        date_path = label_dir / f"gladl_{tile_id}_alertDate{yy_text}.tif"
        if not date_path.exists():
            continue
        year = 2000 + int(yy_text)
        with (
            rasterio.open(alert_path) as alert_src,
            rasterio.open(date_path) as date_src,
        ):
            alert = alert_src.read(1)
            alert_date = date_src.read(1)
            positive = alert > 0
            time_steps, years = _day_of_year_time_arrays(alert_date, year)
            _merge_target_arrays(
                positive,
                years,
                time_steps,
                alert_src,
                target_years,
                target_time_steps,
                reference_profile,
            )


def _merge_glads2_targets(
    alert_path: Path,
    date_path: Path,
    target_years: np.ndarray,
    target_time_steps: np.ndarray,
    reference_profile: dict[str, Any],
) -> None:
    if not alert_path.exists() or not date_path.exists():
        return
    with rasterio.open(alert_path) as alert_src, rasterio.open(date_path) as date_src:
        alert = alert_src.read(1)
        alert_date = date_src.read(1)
        positive = alert >= 2
        time_steps, years = _day_offset_time_arrays(alert_date, date(2019, 1, 1))
        _merge_target_arrays(
            positive,
            years,
            time_steps,
            alert_src,
            target_years,
            target_time_steps,
            reference_profile,
        )


def _merge_target_arrays(
    positive: np.ndarray,
    years: np.ndarray,
    time_steps: np.ndarray,
    source: Any,
    target_years: np.ndarray,
    target_time_steps: np.ndarray,
    reference_profile: dict[str, Any],
) -> None:
    if not np.any(positive):
        return
    source_years = np.where(positive, years, 0).astype(np.int32)
    source_time_steps = np.where(positive, time_steps, 0).astype(np.int32)
    reprojected_years = _reproject_label_array(source_years, source, reference_profile)
    reprojected_time_steps = _reproject_label_array(
        source_time_steps,
        source,
        reference_profile,
    )
    update = (reprojected_years > 0) & (
        (target_years == 0) | (reprojected_years < target_years)
    )
    target_years[update] = reprojected_years[update]
    target_time_steps[update] = reprojected_time_steps[update]


def _reproject_label_array(
    values: np.ndarray,
    source: Any,
    reference_profile: dict[str, Any],
) -> np.ndarray:
    destination = np.zeros(
        (reference_profile["height"], reference_profile["width"]),
        dtype=np.int32,
    )
    reproject(
        source=values,
        destination=destination,
        src_transform=source.transform,
        src_crs=source.crs,
        dst_transform=reference_profile["transform"],
        dst_crs=reference_profile["crs"],
        resampling=Resampling.nearest,
    )
    return destination


def _prediction_candidates(
    prediction_data_dir: Path,
    members: list[EnsembleMember],
    threshold: float,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for tile_id in _prediction_tile_ids(prediction_data_dir):
        baseline_path = _aef_path(prediction_data_dir, tile_id, 2020)
        if baseline_path is None:
            continue
        for year, current_path in _aef_year_paths(prediction_data_dir, tile_id):
            if year <= 2020:
                continue
            feature_grid, profile = _aef_feature_grid(baseline_path, current_path)
            context_maps = _context_maps(feature_grid)
            rows = [
                _feature_row_from_grid(feature_grid, row, col, context_maps)
                for row in range(feature_grid.shape[1])
                for col in range(feature_grid.shape[2])
            ]
            if not rows:
                continue
            x_values = _matrix(rows)
            probabilities, model_names = _predict_ensemble_probability(
                members,
                x_values,
            )
            probability_grid = probabilities.reshape(
                feature_grid.shape[1],
                feature_grid.shape[2],
            )
            time_step = (year % 100) * 100 + 7
            features.extend(
                _polygonize_prediction_mask(
                    probability_grid >= threshold,
                    time_step,
                    profile,
                    tile_id,
                    "+".join(model_names),
                )
            )
    return features


def _polygonize_prediction_mask(
    positive: np.ndarray,
    time_step: int,
    profile: dict[str, Any],
    tile_id: str,
    model_ensemble: str,
) -> list[dict[str, Any]]:
    if not np.any(positive):
        return []

    from scipy.ndimage import label as connected_components

    component_labels, num_components = connected_components(positive.astype(np.uint8))
    if num_components == 0:
        return []

    features: list[dict[str, Any]] = []
    for component_id in range(1, num_components + 1):
        component = component_labels == component_id
        if int(component.sum()) < 3:
            continue
        values = np.where(component, time_step, 0).astype(np.int32)
        for geometry, value in shapes(
            values,
            mask=component,
            transform=profile["transform"],
        ):
            feature_time_step = int(value)
            features.append(
                {
                    "type": "Feature",
                    "geometry": _to_output_crs(geometry, profile["crs"]),
                    "properties": {
                        "tile_id": tile_id,
                        "time_step": feature_time_step,
                        "year": _year_from_time_step(feature_time_step),
                        "model_ensemble": model_ensemble,
                    },
                }
            )
    return features


def _to_output_crs(geometry: dict[str, Any], crs: Any) -> dict[str, Any]:
    if crs is None:
        return geometry
    source_crs = crs.to_string() if hasattr(crs, "to_string") else str(crs)
    if source_crs == DEFAULT_OUTPUT_CRS:
        return geometry
    return transform_geom(source_crs, DEFAULT_OUTPUT_CRS, geometry, precision=7)


def _first_matching_pixel(
    values: np.ndarray,
    positive: np.ndarray,
    value: int,
) -> tuple[int, int]:
    if value > 0:
        locations = np.argwhere(positive & (values == value))
    else:
        locations = np.argwhere(positive)
    if locations.size == 0:
        return 0, 0
    row, col = locations[0]
    return int(row), int(col)


def _radd_time_arrays(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_steps = np.zeros(raw.shape, dtype=np.int32)
    years = np.zeros(raw.shape, dtype=np.int32)
    for raw_value in np.unique(raw[raw > 0]):
        days_since = int(raw_value) % 10000
        observed = date(2014, 12, 31) + timedelta(days=days_since)
        time_step = (observed.year % 100) * 100 + observed.month
        mask = raw == raw_value
        time_steps[mask] = time_step
        years[mask] = observed.year
    return time_steps, years


def _day_of_year_time_arrays(
    day_of_year: np.ndarray,
    year: int,
) -> tuple[np.ndarray, np.ndarray]:
    time_steps = np.zeros(day_of_year.shape, dtype=np.int32)
    years = np.zeros(day_of_year.shape, dtype=np.int32)
    for raw_day in np.unique(day_of_year[day_of_year > 0]):
        observed = date(year, 1, 1) + timedelta(days=int(raw_day) - 1)
        time_step = (observed.year % 100) * 100 + observed.month
        mask = day_of_year == raw_day
        time_steps[mask] = time_step
        years[mask] = observed.year
    return time_steps, years


def _day_offset_time_arrays(
    day_offset: np.ndarray,
    origin: date,
) -> tuple[np.ndarray, np.ndarray]:
    time_steps = np.zeros(day_offset.shape, dtype=np.int32)
    years = np.zeros(day_offset.shape, dtype=np.int32)
    for raw_offset in np.unique(day_offset[day_offset > 0]):
        observed = origin + timedelta(days=int(raw_offset))
        time_step = (observed.year % 100) * 100 + observed.month
        mask = day_offset == raw_offset
        time_steps[mask] = time_step
        years[mask] = observed.year
    return time_steps, years


def _year_from_time_step(value: Any) -> int | None:
    if value is None:
        return None
    integer = int(value)
    if 100 <= integer <= 9999:
        yy = integer // 100
        month = integer % 100
        if 0 <= yy <= 99 and 1 <= month <= 12:
            return 2000 + yy
    return None


def run_experiment() -> list[EnsembleMember]:
    """Train on hardcoded training data and return the trained model."""

    with _wall_time_limit(TRAINING_TIMEOUT_SECONDS):
        x_train, y_train = _build_training_examples(TRAINING_DATA_DIR)
        return _fit_ensemble(x_train, y_train)


def run_inference(
    model: list[EnsembleMember],
    prediction_data_dir: str | Path,
    threshold: float = PREDICTION_THRESHOLD,
) -> dict[str, Any]:
    """Apply a trained model to unlabeled input data and return predictions."""

    prediction_data_dir = Path(prediction_data_dir)
    if not prediction_data_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction data directory not found: {prediction_data_dir}"
        )
    features = _prediction_candidates(prediction_data_dir, model, threshold)
    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    import json

    default_prediction_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "makeathon-challenge"
        / "prediction"
    )
    trained_model = run_experiment()
    print(json.dumps(run_inference(trained_model, default_prediction_dir), indent=2))
# EVOLVE-BLOCK-END
