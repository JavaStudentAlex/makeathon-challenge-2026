# flake8: noqa
"""Generated Shinka method snapshot: temporal_supported_first_hit.

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
DEFAULT_RUN_TIMEOUT_SECONDS = 40 * 60

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
    "yoy_shift",
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
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

PREDICTION_THRESHOLD = 0.60
PREDICT_DOWNSAMPLE_FACTOR = 4
TRAIN_DOWNSAMPLE_FACTOR = 8
MAX_POSITIVE_SAMPLES_PER_RASTER = 672
MAX_NEGATIVE_SAMPLES_PER_RASTER = 832
MIN_TRAINING_EXAMPLES_PER_CLASS = 1

MIN_COMPONENT_PIXELS = 2
LOCAL_SUPPORT_FRACTION = 0.22
STRONG_PIXEL_MARGIN = 0.12
LOW_THRESHOLD_MARGIN = 0.055
COMPONENT_MEAN_MARGIN = 0.015
COMPONENT_MAX_MARGIN = 0.06
LARGE_COMPONENT_PIXELS = 7
TEMPORAL_EARLY_MARGIN = 0.08
TEMPORAL_NEW_SHIFT_MIN = 0.15
TEMPORAL_NEW_SHIFT_RATIO = 0.26
TEMPORAL_STRONG_MARGIN = 0.16


class EnsembleMember(NamedTuple):
    name: str
    model: Any
    weight: float


class PrototypeDistanceClassifier:
    """Robust centroid-distance classifier for sparse/noisy weak labels."""

    def __init__(self) -> None:
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.pos_proto_: np.ndarray | None = None
        self.neg_proto_: np.ndarray | None = None
        self.temperature_: float = 1.0

    def fit(self, x_values: np.ndarray, y_values: np.ndarray) -> "PrototypeDistanceClassifier":
        x_values = np.asarray(x_values, dtype=np.float32)
        y_values = np.asarray(y_values, dtype=np.int32)
        self.center_ = np.nanmedian(x_values, axis=0)
        q25, q75 = np.nanpercentile(x_values, [25, 75], axis=0)
        self.scale_ = np.maximum(q75 - q25, 1e-3)
        z_values = np.clip((x_values - self.center_) / self.scale_, -8.0, 8.0)
        self.pos_proto_ = np.nanmedian(z_values[y_values == 1], axis=0)
        self.neg_proto_ = np.nanmedian(z_values[y_values == 0], axis=0)
        pos_dist = np.mean(np.square(z_values - self.pos_proto_), axis=1)
        neg_dist = np.mean(np.square(z_values - self.neg_proto_), axis=1)
        margin = neg_dist - pos_dist
        self.temperature_ = float(max(np.nanstd(margin), 0.25))
        return self

    def predict_proba(self, x_values: np.ndarray) -> np.ndarray:
        if (
            self.center_ is None
            or self.scale_ is None
            or self.pos_proto_ is None
            or self.neg_proto_ is None
        ):
            raise RuntimeError("PrototypeDistanceClassifier is not fitted")
        z_values = np.clip(
            (np.asarray(x_values, dtype=np.float32) - self.center_) / self.scale_,
            -8.0,
            8.0,
        )
        pos_dist = np.mean(np.square(z_values - self.pos_proto_), axis=1)
        neg_dist = np.mean(np.square(z_values - self.neg_proto_), axis=1)
        margin = np.clip((neg_dist - pos_dist) / self.temperature_, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-margin))
        return np.column_stack([1.0 - probability, probability])


class FeatureSubsetClassifier:
    """Adapter for statistical experts focused on stable, interpretable columns."""

    def __init__(self, estimator: Any, feature_names: tuple[str, ...]) -> None:
        self.estimator = estimator
        self.columns = np.asarray([FEATURE_INDEX[name] for name in feature_names], dtype=np.int32)

    def fit(self, x_values: np.ndarray, y_values: np.ndarray) -> "FeatureSubsetClassifier":
        self.estimator.fit(np.asarray(x_values)[:, self.columns], y_values)
        return self

    def predict_proba(self, x_values: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(np.asarray(x_values)[:, self.columns])


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


def _build_training_examples(
    training_data_dir: str | Path = TRAINING_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised examples from training features and training labels."""

    training_data_dir = Path(training_data_dir)
    if not training_data_dir.is_dir():
        raise FileNotFoundError(f"Training data directory not found: {training_data_dir}")

    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rng = np.random.default_rng(RANDOM_STATE)

    for tile_id in _training_tile_ids(training_data_dir):
        baseline_path = _aef_path(training_data_dir, tile_id, 2020)
        if baseline_path is None:
            continue

        label_targets: tuple[np.ndarray, np.ndarray] | None = None
        prev_path = baseline_path

        for year, current_path in _aef_year_paths(training_data_dir, tile_id):
            if year <= 2020:
                continue

            feature_grid, yoy_grid, profile = _aef_feature_grid(
                baseline_path,
                current_path,
                prev_path,
                downsample_factor=TRAIN_DOWNSAMPLE_FACTOR,
            )
            prev_path = current_path

            if label_targets is None:
                label_targets = _training_label_targets(training_data_dir, tile_id, profile)

            label_years, _ = label_targets
            x_grid = _feature_matrix_from_grid(feature_grid, yoy_grid)

            shift_map = np.linalg.norm(feature_grid, axis=0)
            yoy_shift = np.linalg.norm(yoy_grid, axis=0)
            future_negative = label_years > year
            negative_mask = (label_years == 0) | future_negative

            hardness = (
                shift_map
                + 0.35 * yoy_shift
                + 0.25 * future_negative.astype(np.float32)
                + 0.08 * np.abs(feature_grid[0] if feature_grid.shape[0] else 0.0)
            )

            pos_idx = _sample_flat_indices(
                label_years == year,
                MAX_POSITIVE_SAMPLES_PER_RASTER,
                rng,
                hardness=hardness,
            )
            neg_idx = _sample_flat_indices(
                negative_mask,
                MAX_NEGATIVE_SAMPLES_PER_RASTER,
                rng,
                hardness=hardness,
            )

            if pos_idx.size:
                chunks.append(x_grid[pos_idx])
                labels.append(np.ones(pos_idx.size, dtype=np.int32))
            if neg_idx.size:
                chunks.append(x_grid[neg_idx])
                labels.append(np.zeros(neg_idx.size, dtype=np.int32))

    if not chunks:
        raise ValueError("Training labels did not produce examples")

    x_train = np.vstack(chunks).astype(np.float32)
    y_train = np.concatenate(labels).astype(np.int32)

    if (
        y_train.size == 0
        or np.count_nonzero(y_train == 1) < MIN_TRAINING_EXAMPLES_PER_CLASS
        or np.count_nonzero(y_train == 0) < MIN_TRAINING_EXAMPLES_PER_CLASS
    ):
        raise ValueError("Training labels did not produce both positive and negative examples")

    return x_train, y_train


def _fit_ensemble(x_train: np.ndarray, y_train: np.ndarray) -> list[EnsembleMember]:
    """Fit a compact heterogeneous shallow ensemble."""

    members: list[EnsembleMember] = []
    pos = int(np.count_nonzero(y_train == 1))
    neg = int(np.count_nonzero(y_train == 0))
    pos_weight = max(1.0, neg / max(1, pos))
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=26,
            max_depth=2,
            learning_rate=0.20,
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=0.95,
            colsample_bytree=0.85,
            scale_pos_weight=pos_weight,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        xgb.fit(x_train, y_train)
        members.append(EnsembleMember("xgboost", xgb, 0.30))
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            n_estimators=34,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.16,
            objective="binary",
            class_weight="balanced",
            min_child_samples=1,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            random_state=RANDOM_STATE + 1,
            n_jobs=1,
            verbosity=-1,
            force_col_wise=True,
        )
        lgbm.fit(x_train, y_train)
        members.append(EnsembleMember("lightgbm", lgbm, 0.30))
    except Exception:
        pass

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        hgb = HistGradientBoostingClassifier(
            max_iter=60,
            max_leaf_nodes=9,
            max_depth=4,
            learning_rate=0.11,
            l2_regularization=0.03,
            random_state=RANDOM_STATE + 2,
        )
        hgb.fit(x_train, y_train, sample_weight=sample_weight)
        members.append(EnsembleMember("hgb", hgb, 0.16))
    except Exception:
        pass

    try:
        from sklearn.ensemble import ExtraTreesClassifier

        extra = ExtraTreesClassifier(
            n_estimators=112,
            max_depth=10,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE + 3,
            n_jobs=1,
        )
        extra.fit(x_train, y_train)
        members.append(EnsembleMember("extratrees", extra, 0.15))
    except Exception:
        pass

    try:
        from sklearn.linear_model import LogisticRegression

        linear = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.75,
                class_weight="balanced",
                max_iter=450,
                random_state=RANDOM_STATE + 4,
                solver="lbfgs",
            ),
        )
        linear.fit(x_train, y_train)
        members.append(EnsembleMember("logreg", linear, 0.07))
    except Exception:
        pass

    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda_features = (
            "ndvi_delta",
            "nbr_delta",
            "ndmi_delta",
            "bsi_delta",
            "vv_delta",
            "vv_cv_delta",
            "aef_shift",
            "yoy_shift",
            "aef_abs_std",
            "local_shift_mean",
            "local_shift_contrast",
            "seasonal_drop",
        )
        lda = FeatureSubsetClassifier(
            make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
            lda_features,
        ).fit(x_train, y_train)
        members.append(EnsembleMember("subset_lda", lda, 0.05))
    except Exception:
        pass

    try:
        from sklearn.naive_bayes import GaussianNB

        nb = GaussianNB()
        nb.fit(x_train, y_train)
        members.append(EnsembleMember("gaussian_nb", nb, 0.03))
    except Exception:
        pass

    try:
        proto = PrototypeDistanceClassifier().fit(x_train, y_train)
        members.append(EnsembleMember("prototype_distance", proto, 0.07))
    except Exception:
        pass

    return members


def _positive_signal(value: np.ndarray | float, scale: float) -> np.ndarray | float:
    return np.clip(np.asarray(value, dtype=float) / scale, 0.0, 1.0)


def _drop_signal(value: np.ndarray | float, scale: float) -> np.ndarray | float:
    return np.clip(-np.asarray(value, dtype=float) / scale, 0.0, 1.0)


def _domain_prior_probability(x_values: np.ndarray) -> np.ndarray:
    """Fixed physics-inspired prior blended with learned model probabilities."""

    idx = FEATURE_INDEX

    optical_drop = np.mean(
        np.stack(
            [
                _positive_signal(x_values[:, idx["ndvi_delta"]], 0.80),
                _positive_signal(x_values[:, idx["nbr_delta"]], 0.95),
                _positive_signal(x_values[:, idx["ndmi_delta"]], 0.70),
                _positive_signal(x_values[:, idx["evi_delta"]], 1.00),
                _positive_signal(x_values[:, idx["ndre_delta"]], 0.70),
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
    yoy_signal = _positive_signal(x_values[:, idx["yoy_shift"]], 0.70)
    band_dispersion = _positive_signal(x_values[:, idx["aef_abs_std"]], 0.58)
    sign_structure = np.clip(x_values[:, idx["aef_neg_frac"]], 0.0, 1.0)

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

    forest = np.clip(x_values[:, idx["forest_2020"]], 0.0, 1.0)
    probability = (
        0.26 * optical_drop
        + 0.15 * exposure
        + 0.13 * sar_drop
        + 0.16 * semantic_shift
        + 0.07 * yoy_signal
        + 0.09 * local_support
        + 0.04 * band_dispersion
        + 0.03 * sign_structure
        + 0.07 * np.clip(x_values[:, idx["alert_consensus"]], 0.0, 1.0)
        - 0.41 * negative_context
    )
    return np.clip(probability, 0.0, 1.0) * (forest >= 0.5)


def _predict_ensemble_probability(
    members: list[EnsembleMember],
    x_values: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    if not members:
        return _domain_prior_probability(x_values), ["domain_prior"]

    weighted = np.zeros(x_values.shape[0], dtype=np.float64)
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

    model_probability = weighted / max(total_weight, 1e-12)
    prior_probability = _domain_prior_probability(x_values)
    return np.clip(0.73 * model_probability + 0.27 * prior_probability, 0.0, 1.0), names


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
    previous_path: Path | None = None,
    downsample_factor: int = PREDICT_DOWNSAMPLE_FACTOR,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(baseline_path) as baseline_src, rasterio.open(current_path) as current_src:
        height = max(current_src.height // downsample_factor, 1)
        width = max(current_src.width // downsample_factor, 1)
        read_shape = (len(AEF_BAND_INDEXES), height, width)

        baseline = baseline_src.read(
            indexes=AEF_BAND_INDEXES,
            out_shape=read_shape,
            resampling=Resampling.average,
        ).astype(np.float32)
        current = current_src.read(
            indexes=AEF_BAND_INDEXES,
            out_shape=read_shape,
            resampling=Resampling.average,
        ).astype(np.float32)

        if previous_path and previous_path != baseline_path:
            with rasterio.open(previous_path) as prev_src:
                previous = prev_src.read(
                    indexes=AEF_BAND_INDEXES,
                    out_shape=read_shape,
                    resampling=Resampling.average,
                ).astype(np.float32)
        else:
            previous = baseline

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
    yoy_delta = np.nan_to_num(current - previous, nan=0.0, posinf=0.0, neginf=0.0)
    return delta, yoy_delta, profile


def _local_mean_std_max(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded = np.pad(values, 1, mode="edge")
    neighbours = (
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, :-2],
        padded[1:-1, 1:-1],
        padded[1:-1, 2:],
        padded[2:, :-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    )
    local_mean = sum(neighbours) / 9.0

    padded_sq = np.pad(np.square(values), 1, mode="edge")
    neighbours_sq = (
        padded_sq[:-2, :-2],
        padded_sq[:-2, 1:-1],
        padded_sq[:-2, 2:],
        padded_sq[1:-1, :-2],
        padded_sq[1:-1, 1:-1],
        padded_sq[1:-1, 2:],
        padded_sq[2:, :-2],
        padded_sq[2:, 1:-1],
        padded_sq[2:, 2:],
    )
    local_std = np.sqrt(np.maximum(sum(neighbours_sq) / 9.0 - np.square(local_mean), 0.0))
    local_max = np.maximum.reduce(neighbours)
    return local_mean, local_std, local_max


def _feature_matrix_from_grid(feature_grid: np.ndarray, yoy_grid: np.ndarray) -> np.ndarray:
    bands, height, width = feature_grid.shape
    n_pixels = height * width
    x_values = np.zeros((n_pixels, len(FEATURE_NAMES)), dtype=np.float32)
    idx = FEATURE_INDEX

    x_values[:, idx["forest_2020"]] = 1.0

    flat = feature_grid.reshape(bands, n_pixels)
    for i, band in enumerate(AEF_BAND_INDEXES[: min(bands, len(AEF_BAND_INDEXES))]):
        x_values[:, idx[f"aef_delta_{band:02d}"]] = flat[i]

    shift_map = np.linalg.norm(feature_grid, axis=0)
    yoy_shift = np.linalg.norm(yoy_grid, axis=0)
    abs_grid = np.abs(feature_grid)
    local_mean, local_std, local_max = _local_mean_std_max(shift_map)

    x_values[:, idx["aef_shift"]] = shift_map.ravel()
    x_values[:, idx["yoy_shift"]] = yoy_shift.ravel()
    x_values[:, idx["aef_abs_mean"]] = abs_grid.mean(axis=0).ravel()
    x_values[:, idx["aef_abs_std"]] = abs_grid.std(axis=0).ravel()
    x_values[:, idx["aef_pos_frac"]] = (feature_grid > 0).mean(axis=0).ravel()
    x_values[:, idx["aef_neg_frac"]] = (feature_grid < 0).mean(axis=0).ravel()
    x_values[:, idx["local_shift_mean"]] = local_mean.ravel()
    x_values[:, idx["local_shift_std"]] = local_std.ravel()
    x_values[:, idx["local_shift_max"]] = local_max.ravel()
    x_values[:, idx["local_shift_contrast"]] = (shift_map - local_mean).ravel()

    if bands > 0:
        std = float(np.nanstd(flat[0])) + 1e-6
        x_values[:, idx["ndvi_delta"]] = -flat[0]
        x_values[:, idx["ndvi_zscore"]] = -flat[0] / std
    if bands > 1:
        std = float(np.nanstd(flat[1])) + 1e-6
        x_values[:, idx["nbr_delta"]] = -flat[1]
        x_values[:, idx["nbr_zscore"]] = -flat[1] / std
    if bands > 2:
        std = float(np.nanstd(flat[2])) + 1e-6
        x_values[:, idx["ndmi_delta"]] = -flat[2]
        x_values[:, idx["ndmi_zscore"]] = -flat[2] / std
    if bands > 3:
        std = float(np.nanstd(flat[3])) + 1e-6
        x_values[:, idx["bsi_delta"]] = flat[3]
        x_values[:, idx["bsi_zscore"]] = flat[3] / std
    if bands > 4:
        x_values[:, idx["evi_delta"]] = -flat[4]
    if bands > 5:
        x_values[:, idx["ndre_delta"]] = -flat[5]
    if bands > 6:
        x_values[:, idx["ndwi_delta"]] = flat[6]
        x_values[:, idx["water"]] = np.clip(flat[6] / 0.9, 0.0, 1.0)
    if bands > 7:
        std = float(np.nanstd(flat[7])) + 1e-6
        x_values[:, idx["vv_delta"]] = flat[7]
        x_values[:, idx["vv_zscore"]] = flat[7] / std
    if bands > 8:
        x_values[:, idx["vv_cv_delta"]] = np.abs(flat[8])
    if bands > 9:
        x_values[:, idx["seasonal_drop"]] = np.clip(
            np.abs(flat[9]) / (shift_map.ravel() + 1e-3),
            0.0,
            1.0,
        )

    return np.nan_to_num(x_values, nan=0.0, posinf=0.0, neginf=0.0)


def _sample_flat_indices(
    mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
    hardness: np.ndarray | None = None,
) -> np.ndarray:
    flat_locations = np.flatnonzero(mask.ravel())
    if flat_locations.size == 0:
        return np.array([], dtype=np.int64)

    count = min(count, flat_locations.size)
    if hardness is None or count == flat_locations.size:
        return rng.choice(flat_locations, size=count, replace=False)

    random_count = count - count // 2
    hard_count = count - random_count
    random_part = rng.choice(flat_locations, size=random_count, replace=False)

    hard_values = hardness.ravel()[flat_locations].astype(np.float64)
    weights = hard_values - hard_values.min()
    if weights.sum() <= 1e-9:
        return rng.choice(flat_locations, size=count, replace=False)

    hard_part = rng.choice(
        flat_locations,
        size=hard_count,
        replace=False,
        p=weights / weights.sum(),
    )
    merged = np.unique(np.concatenate([random_part, hard_part]))
    if merged.size < count:
        remaining = np.setdiff1d(flat_locations, merged, assume_unique=False)
        if remaining.size:
            fill = rng.choice(
                remaining,
                size=min(count - merged.size, remaining.size),
                replace=False,
            )
            merged = np.concatenate([merged, fill])
    return merged.astype(np.int64)


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
        try:
            _tile_id, yy_text = stem.rsplit("_alert", maxsplit=1)
            year = 2000 + int(yy_text)
        except ValueError:
            continue

        date_path = label_dir / f"gladl_{tile_id}_alertDate{yy_text}.tif"
        if not date_path.exists():
            continue

        with rasterio.open(alert_path) as alert_src, rasterio.open(date_path) as date_src:
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


def _clean_prediction_mask(probability_grid: np.ndarray, threshold: float) -> np.ndarray:
    candidate = probability_grid >= threshold
    if not np.any(candidate):
        return candidate

    try:
        from scipy import ndimage

        local_probability = ndimage.uniform_filter(
            probability_grid.astype(np.float32),
            size=3,
            mode="nearest",
        )
        low = probability_grid >= max(0.05, threshold - LOW_THRESHOLD_MARGIN)
        supported_low = low & (local_probability >= max(0.05, threshold - 0.035))
        candidate = candidate | supported_low

        local_support = (
            ndimage.uniform_filter(candidate.astype(np.float32), size=3, mode="nearest")
            >= LOCAL_SUPPORT_FRACTION
        )
        strong = probability_grid >= min(0.98, threshold + STRONG_PIXEL_MARGIN)
        cleaned = candidate & (local_support | strong)

        labels, count = ndimage.label(cleaned)
        if count == 0:
            return cleaned

        flat_labels = labels.ravel()
        sizes = np.bincount(flat_labels)
        prob_sum = np.bincount(
            flat_labels,
            weights=probability_grid.ravel().astype(np.float32),
            minlength=sizes.size,
        )
        prob_max = ndimage.maximum(
            probability_grid,
            labels=labels,
            index=np.arange(sizes.size),
        )
        prob_mean = prob_sum / np.maximum(sizes, 1)
        seed_counts = np.bincount(
            flat_labels,
            weights=(probability_grid >= threshold).ravel().astype(np.float32),
            minlength=sizes.size,
        )

        keep = (
            (sizes >= MIN_COMPONENT_PIXELS)
            & (seed_counts > 0)
            & (
                (prob_max >= threshold + COMPONENT_MAX_MARGIN)
                | (prob_mean >= threshold + COMPONENT_MEAN_MARGIN)
                | (sizes >= LARGE_COMPONENT_PIXELS)
            )
        )
        keep[0] = False
        return ndimage.binary_fill_holes(keep[labels]).astype(bool)
    except Exception:
        return probability_grid >= threshold


def _temporal_new_disturbance_gate(
    feature_grid: np.ndarray,
    yoy_grid: np.ndarray,
    probability_grid: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Keep detections with year-over-year support, unless very confident."""

    baseline_shift = np.linalg.norm(feature_grid, axis=0)
    yoy_shift = np.linalg.norm(yoy_grid, axis=0)
    yoy_ratio = np.divide(
        yoy_shift,
        baseline_shift + 1e-6,
        out=np.zeros_like(yoy_shift, dtype=np.float32),
        where=baseline_shift > 1e-6,
    )
    return (
        (yoy_shift >= TEMPORAL_NEW_SHIFT_MIN)
        | (yoy_ratio >= TEMPORAL_NEW_SHIFT_RATIO)
        | (probability_grid >= min(0.99, threshold + TEMPORAL_STRONG_MARGIN))
    )


def _remove_tiny_components(mask: np.ndarray, min_pixels: int = MIN_COMPONENT_PIXELS) -> np.ndarray:
    if not np.any(mask):
        return mask
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask)
        if count == 0:
            return mask
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_pixels
        keep[0] = False
        return keep[labels]
    except Exception:
        return mask


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

        year_paths = [(y, p) for y, p in _aef_year_paths(prediction_data_dir, tile_id) if y > 2020]
        if not year_paths:
            continue

        per_year: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[str]]] = []
        prev_path = baseline_path

        for year, current_path in year_paths:
            feature_grid, yoy_grid, profile = _aef_feature_grid(baseline_path, current_path, prev_path)
            prev_path = current_path

            x_values = _feature_matrix_from_grid(feature_grid, yoy_grid)
            probabilities, model_names = _predict_ensemble_probability(members, x_values)
            probability_grid = probabilities.reshape(feature_grid.shape[1], feature_grid.shape[2])
            per_year.append((year, probability_grid, feature_grid, yoy_grid, profile, model_names))

        if not per_year:
            continue

        stack = np.stack([prob for _, prob, _, _, _, _ in per_year], axis=0)
        max_prob = np.max(stack, axis=0)
        temporal_stack = np.stack(
            [
                _temporal_new_disturbance_gate(feature_grid, yoy_grid, probability_grid, threshold)
                for _, probability_grid, feature_grid, yoy_grid, _, _ in per_year
            ],
            axis=0,
        )
        strong_stack = stack >= min(0.99, threshold + TEMPORAL_STRONG_MARGIN)

        # Deforestation is absorbing. Emit at most one prediction year per pixel.
        # Apply the new-disturbance gate before first-hit selection so weak early
        # cumulative changes do not block a later year with stronger YOY support.
        eligible = (stack >= threshold) & (
            stack >= (max_prob[None, :, :] - TEMPORAL_EARLY_MARGIN)
        )
        eligible &= max_prob[None, :, :] >= threshold
        eligible &= temporal_stack | strong_stack
        any_eligible = np.any(eligible, axis=0)
        first_year_idx = np.argmax(eligible, axis=0)

        for year_idx, (year, probability_grid, feature_grid, yoy_grid, profile, model_names) in enumerate(per_year):
            year_mask = any_eligible & (first_year_idx == year_idx)
            if not np.any(year_mask):
                continue

            year_prob = np.where(year_mask, probability_grid, 0.0)
            positive = _clean_prediction_mask(year_prob, threshold)
            if not np.any(positive):
                continue

            temporal_gate = _temporal_new_disturbance_gate(
                feature_grid,
                yoy_grid,
                probability_grid,
                threshold,
            )
            strong_peak = probability_grid >= min(0.99, threshold + TEMPORAL_STRONG_MARGIN)
            positive &= temporal_gate | strong_peak
            positive = _remove_tiny_components(positive, MIN_COMPONENT_PIXELS)
            if not np.any(positive):
                continue

            time_step = (year % 100) * 100 + 7
            features.extend(
                _polygonize_prediction_mask(
                    positive,
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

    feature_list: list[dict[str, Any]] = []
    values = np.where(positive, time_step, 0).astype(np.int32)

    for geometry, value in shapes(values, mask=positive, transform=profile["transform"]):
        feature_time_step = int(value)
        feature_list.append(
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

    return feature_list


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

    with _wall_time_limit(DEFAULT_RUN_TIMEOUT_SECONDS):
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
