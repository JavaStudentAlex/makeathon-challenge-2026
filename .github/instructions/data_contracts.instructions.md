---
description: "Dataset, raster-shape, label-encoding, and submission-format contracts for the challenge repository."
---

# Data Contracts

## Purpose

Use this file when working on any task that touches:

- dataset layout
- raster shapes or band counts
- label decoding or binarisation
- CRS alignment or reprojection
- submission file generation

## Dataset Root

After download, the challenge data lives under:

`data/makeathon-challenge/`

## Directory Layout

```text
data/makeathon-challenge/
├── sentinel-1/
│   ├── train/{tile_id}__s1_rtc/{tile_id}__s1_rtc_{year}_{month}_{ascending|descending}.tif
│   └── test/...
├── sentinel-2/
│   ├── train/{tile_id}__s2_l2a/{tile_id}__s2_l2a_{year}_{month}.tif
│   └── test/...
├── aef-embeddings/
│   ├── train/{tile_id}_{year}.tiff
│   └── test/...
├── labels/train/
│   ├── gladl/   gladl_{tile_id}_alert{YY}.tif + gladl_{tile_id}_alertDate{YY}.tif
│   ├── glads2/  glads2_{tile_id}_alert.tif + glads2_{tile_id}_alertDate.tif
│   └── radd/    radd_{tile_id}_labels.tif
└── metadata/
    ├── train_tiles.geojson
    └── test_tiles.geojson
```

`tile_id` uses the form `{MGRS_grid}_{x}_{y}`, for example `18NWG_6_6`.

## Modality Contracts

### Sentinel-2

- Monthly time series
- GeoTIFF
- 12 spectral bands
- When read as a stack, expected array shape is `(12, height, width)`
- Notebook examples treat bands 4, 3, and 2 as RGB
- Delivered in a local projected CRS that aligns with Sentinel-1 tiles

### Sentinel-1 RTC

- Monthly time series
- GeoTIFF
- 1 VV backscatter band
- When read as a stack, expected shape is `(1, height, width)`
- `src.read(1)` produces a `(height, width)` array
- Delivered in a local projected CRS aligned with Sentinel-2

### AlphaEarth Foundations Embeddings

- Annual raster
- GeoTIFF / TIFF
- 64 embedding bands
- When read as a stack, expected shape is `(64, height, width)`
- Delivered in `EPSG:4326`
- Must be reprojected if used together with Sentinel-1 or Sentinel-2 rasters

## Label Contracts

Labels are provided only for the training split and remain raw weak-label
encodings unless the task explicitly requires decoding or binarisation.

### RADD

- File shape: one TIFF per tile
- `0` means no alert
- Non-zero values encode both confidence and alert date
- Leading digit:
  - `2` = low confidence
  - `3` = high confidence
- Remaining digits are days since `2014-12-31`

Examples:

- `20001` = low-confidence alert on `2015-01-01`
- `30055` = high-confidence alert on `2015-02-24`

### GLAD-L

- `alertYY.tif` is a yearly alert raster, typically `uint8`
  - `0` = no loss
  - `2` = probable loss
  - `3` = confirmed loss
- `alertDateYY.tif` is a yearly date raster, typically `uint16`
  - `0` = no alert
  - non-zero = day-of-year within year `20YY`

### GLAD-S2

- `alert.tif` is a raw confidence raster, typically `uint8`
  - `0` = no loss
  - `1` = most recent observation only
  - `2` = low confidence
  - `3` = medium confidence
  - `4` = high confidence
- `alertDate.tif` is a raw day-offset raster, typically `uint16`
  - `0` = no alert
  - non-zero = days since `2019-01-01`

## Alignment Rules

- Do not assume all modalities share the same CRS.
- Sentinel-1 and Sentinel-2 are expected to align in a local projected CRS.
- AlphaEarth embeddings are in `EPSG:4326` and must be reprojected before
  direct pixel-wise fusion with projected rasters.
- Preserve CRS, transform, dtype, nodata, width, and height when writing
  derived rasters.

## Submission Contract

`submission_utils.raster_to_geojson` is the current submission boundary.

Input expectations:

- single-band raster
- binary values only
- `1` = deforestation
- `0` = background

Output expectations:

- GeoJSON `FeatureCollection`
- geometries in `EPSG:4326`
- polygons smaller than `0.5` hectares are filtered by default
- each feature currently carries a `time_step` property with a null value

## Testing Guidance

- Use synthetic rasters in tests.
- Do not require the downloaded challenge dataset for automated verification.
- Prefer tiny local GeoTIFFs with explicit CRS and transform metadata.
