# Changelog

All notable changes to `segmenteverygrain` are documented here.

## [Unreleased]

### Added

- `sam_segmentation` raises an informative `ValueError` when called with an
  empty prompt list, and `predict_large_image` emits a warning when no grains
  are detected in the image. Previously a failed U-Net prediction (e.g., a
  model that did not load correctly) could surface as a cryptic downstream
  error such as "attempt to get argmax of an empty sequence" (issue #10).

## [0.5.0]

### Changed

- **Area-weighted grain size distributions are now computed with explicit
  weights instead of by replication.** `get_area_weighted_distribution` built
  the weighted distribution by repeating each grain size
  `int(area / (0.5 * mean_area))` times. Because `int()` truncates, every grain
  smaller than half the mean area was dropped entirely; across the example
  datasets this discarded 39–55% of grains and left the distribution built from
  roughly 75% of the total grain area, truncating the fine tail and biasing
  statistics coarse.

  Anyone passing `area` to `plot_histogram_of_axis_lengths` will see different
  numbers after upgrading. The mode and the coarse half of the distribution are
  essentially unchanged; D50 shifts by 0.12–0.24 phi (5–15% in mm) and the fine
  tail by up to 0.7 phi as the previously dropped grains reappear. Folk & Ward
  mean phi moves by +0.16 to +0.28. Histogram bar heights are now about half
  their previous values, because replication produced roughly twice as many
  copies as there were grains.

- **`interactions.get_histogram` and `interactions.save_histogram` are now
  area-weighted by default**, matching what `notebooks/Segment_every_grain.ipynb`
  already did. Pass `area_weighted=False` for the previous count-based
  behavior. The setting is ignored if the summary has no `area` column.

- `plot_histogram_of_axis_lengths` now labels the y axis `area-weighted count`
  when weighted and `count` otherwise. Weights are normalized to sum to the
  number of grains, so the y axis stays on a count-like scale and is
  independent of the units the areas are given in.

### Added

- `weighted_ecdf(values, weights=None)` — weighted empirical cumulative
  distribution function, returning sorted values, the CDF, and the exceedance
  curve. With equal weights it reproduces exactly the curve the previous code
  plotted; with integer weights it matches the replicated ECDF to within 1e-12.
- `weighted_percentile(values, percentiles, weights=None)` — weighted
  percentiles for area-weighted statistics such as D50 and D84.
- `area_weighted` parameter on `interactions.get_histogram` and
  `interactions.save_histogram`.
- Resumable checkpointing and a quiet mode for `predict_large_image`, so long
  runs over large images can be interrupted and continued.
- Tests covering the weighting functions, including the equivalences above and
  regression tests for the two fixes below.

### Fixed

- **NaN handling misaligned the area weights.** `plot_histogram_of_axis_lengths`
  stripped NaNs from the major and minor axis arrays using separate masks while
  leaving `area` unfiltered, so `zip` inside the weighting silently paired grain
  sizes with the wrong areas. A single shared mask now covers all three arrays,
  and a mismatched `area` length raises `ValueError` instead of being silently
  truncated.
- The exceedance curve was computed as `ecdf[::-1]`, which is only correct for
  uniform weights.
- Polygon overlap detection and merging are substantially faster on large
  images.
- setuptools `license-file` metadata no longer breaks PyPI uploads.
- CI no longer segfaults during pytest collection (triton removed).

### Deprecated

- `get_area_weighted_distribution` still works but emits a `DeprecationWarning`.
  Use `weighted_ecdf` and `weighted_percentile`, or pass the areas directly as
  `weights` to Matplotlib's `hist`.

## [0.4.0]

- Fixed a double softmax bug; the U-Net model now outputs raw logits
  (pre-softmax) and softmax is applied where needed.
- Added label smoothing, a U-Net-only mode, and model evaluation utilities,
  along with a U-Net retrained with label smoothing.
- SAM prompts are filtered using the binary grain mask rather than probability
  thresholds, and a coverage overlay (`H` key) was added to the interactive
  interface.
- Added a `min_grain_area` filter to `label_grains` to skip noisy small regions.
- Updated the SAM 2.1 checkpoint URL and the example image.

## [0.3.0]

- Migrated from SAM 1 to SAM 2.1, with cross-platform GPU support (CUDA, Apple
  MPS, CPU).
- Notebooks updated for the new interactive editing interface and moved to
  `notebooks/`.
- CI fixes: quoted Python `"3.10"` to prevent YAML float parsing, and a disk
  space cleanup step before installing dependencies.
- Added citation information to the README.

## [0.2.5]

- Removed hardcoded blitting when drawing patches.

## [0.2.4]

- New interactive editing interface added as the `interactions` submodule,
  with `GrainPlot` replacing the previous editing workflow.
- `predict_from_prompts` moved into `interactions.py`; improved edge grain
  handling.
- Added `extract_color_features`.
- Notebooks moved to `notebooks/` and updated to the new `GrainPlot` API;
  interactive editing became part of the main workflow, relying on
  `predict_large_image`.

## Earlier releases

Releases before 0.2.4 predate this changelog; see the commit history and the
release tags for details.
