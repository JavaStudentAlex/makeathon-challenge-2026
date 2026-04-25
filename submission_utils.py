"""Utilities for converting deforestation prediction rasters into submittable GeoJSON."""

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def is_valid_time_step(time_step: Any) -> bool:
    """Return whether ``time_step`` is allowed by the challenge upload contract."""

    if time_step is None:
        return True
    if isinstance(time_step, bool):
        return False
    if isinstance(time_step, int):
        value = time_step
    elif isinstance(time_step, str) and time_step.isdigit():
        value = int(time_step)
    else:
        return False

    yy = value // 100
    mm = value % 100
    return 0 <= yy <= 99 and 1 <= mm <= 12


def validate_submission_geojson(
    geojson_or_path: dict | str | Path,
    require_geojson_extension: bool = True,
) -> None:
    """Validate challenge GeoJSON upload constraints before submission.

    The upload contract requires a ``.geojson`` file containing a top-level
    ``FeatureCollection``. Every feature geometry must be a ``Polygon`` or
    ``MultiPolygon``. ``properties.time_step`` may be omitted, ``null``, or a
    valid ``YYMM`` value such as ``2204``.
    """

    if isinstance(geojson_or_path, (str, Path)):
        path = Path(geojson_or_path)
        if require_geojson_extension and path.suffix != ".geojson":
            raise ValueError("Submission file extension must be .geojson")
        with open(path) as f:
            geojson = json.load(f)
    else:
        geojson = geojson_or_path

    if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must parse as a top-level FeatureCollection")

    features = geojson.get("features")
    if not isinstance(features, list):
        raise ValueError("FeatureCollection must contain a features list")

    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(f"Feature {index} must be a GeoJSON Feature")

        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"Feature {index} must contain a geometry object")
        geometry_type = geometry.get("type")
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                f"Feature {index} geometry must be Polygon or MultiPolygon, "
                f"got {geometry_type!r}"
            )

        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            raise ValueError(f"Feature {index} properties must be an object")
        if "time_step" in properties and not is_valid_time_step(
            properties["time_step"]
        ):
            raise ValueError(
                f"Feature {index} time_step must be valid YYMM, null, or omitted"
            )


def raster_to_geojson(
    raster_path: str | Path,
    output_path: str | Path | None = None,
    min_area_ha: float = 0.5,
    time_step: int | str | None = None,
    allow_empty: bool = False,
) -> dict:
    """Convert a binary deforestation prediction raster to a GeoJSON FeatureCollection.

    Reads a single-band GeoTIFF where 1 indicates deforestation and 0 indicates
    no deforestation, vectorises the foreground pixels into polygons, removes
    polygons smaller than ``min_area_ha``, reprojects the result to EPSG:4326,
    and returns (and optionally writes) a GeoJSON FeatureCollection.

    The caller is responsible for binarising their model output before passing
    it to this function. This function is designed to be the final step in the
    submission pipeline: take your binarised prediction raster, call this
    function, and upload the resulting ``.geojson`` file to the leaderboard.

    Args:
        raster_path: Path to the input GeoTIFF. Must be a single-band raster
            with binary values (0 = no deforestation, 1 = deforestation).
        output_path: Optional path at which to write the GeoJSON file. Parent
            directories are created automatically. If ``None``, the result is
            returned but not written to disk.
        min_area_ha: Minimum polygon area in hectares. Polygons smaller than
            this threshold are removed before the output is written. Area is
            computed in the appropriate UTM projection so the filter is
            metric-accurate regardless of the raster's native CRS. Defaults
            to ``0.5``.
        time_step: Optional scalar ``YYMM`` value assigned to every output
            feature. Use ``None`` to write the accepted null value.
        allow_empty: If ``True``, return and optionally write an empty
            ``FeatureCollection`` instead of raising when the raster contains
            no deforestation pixels or when all polygons are filtered out.

    Returns:
        A GeoJSON-compatible ``dict`` representing a FeatureCollection. Each
        Feature corresponds to one contiguous deforestation polygon in
        EPSG:4326 (longitude/latitude, WGS-84).

    Raises:
        FileNotFoundError: If ``raster_path`` does not point to an existing file.
        ValueError: If the raster contains no deforestation pixels (all zeros),
            or if all polygons are smaller than ``min_area_ha``.

    Example:
        >>> geojson = raster_to_geojson(
        ...     raster_path="predictions/tile_18NVJ.tif",
        ...     output_path="submission/tile_18NVJ.geojson",
        ...     min_area_ha=0.5,
        ... )
        >>> print(len(geojson["features"]), "deforestation polygons")
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    if not is_valid_time_step(time_step):
        raise ValueError("time_step must be valid YYMM, null, or omitted")

    def _finalize_output(geojson: dict) -> dict:
        if output_path is not None:
            final_output_path = Path(output_path)
            if final_output_path.suffix != ".geojson":
                raise ValueError("Submission file extension must be .geojson")
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(final_output_path, "w") as f:
                json.dump(geojson, f)

            validate_submission_geojson(final_output_path)

        return geojson

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    if data.sum() == 0:
        if allow_empty:
            return _finalize_output({"type": "FeatureCollection", "features": []})
        raise ValueError(
            f"No deforestation pixels (value=1) found in {raster_path}. "
            "Ensure the raster has been binarised before calling this function."
        )

    # Vectorise connected foreground regions into polygons
    polygons = [
        shape(geom)
        for geom, value in shapes(data, mask=data, transform=transform)
        if value == 1
    ]

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf = gdf.to_crs("EPSG:4326")

    # Filter by area: project to UTM for metric-accurate ha calculation
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    gdf = gdf[gdf_utm.area / 10_000 >= min_area_ha].reset_index(drop=True)

    if gdf.empty:
        if allow_empty:
            return _finalize_output({"type": "FeatureCollection", "features": []})
        raise ValueError(
            f"All polygons are smaller than min_area_ha={min_area_ha} ha. "
            "Lower the threshold or check your prediction raster."
        )

    gdf["time_step"] = time_step

    geojson = json.loads(gdf.to_json())
    return _finalize_output(geojson)
