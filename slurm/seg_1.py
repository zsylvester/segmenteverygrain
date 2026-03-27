# to prevent blocking all memory
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("TF GPUs:", tf.config.list_physical_devices("GPU"))

import cv2
from keras.utils import load_img
from keras.saving import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import regionprops, regionprops_table
from tqdm import trange, tqdm

import segmenteverygrain as seg
import segmenteverygrain.interactions as si
import os
import rasterio
import time, traceback
import pandas as pd
import torch

# %matplotlib qt

import torch

# 1) UNet laden (TF)
unet = load_model(
    "../models/seg_model.keras",
    custom_objects={"weighted_crossentropy": seg.weighted_crossentropy},
)
print("UNet loaded")

# 2) SAM2 laden (Torch)
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg  = "configs/sam2.1/sam2.1_hiera_l.yaml"
ckpt = "/dss/dsshome1/0B/di54doz/segmenteverygrain/models/sam2.1_hiera_large.pt"
sam = build_sam2(cfg, ckpt, device=device)
print("SAM2 loaded on", device)

# 3) kurzer Speicher-Check
free, total = torch.cuda.mem_get_info()
print(f"torch reports free {free/1024**3:.1f} GB / total {total/1024**3:.1f} GB")


import os
import glob
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img
import rasterio

import pandas as pd
import torch

import geopandas as gpd
from shapely.geometry import Polygon
from skimage.measure import regionprops_table

# -----------------------------
# USER SETTINGS
# -----------------------------
OUT_ROOT = "/dss/tbyscratch/0B/di54doz/seg"   # alles hier rein
os.makedirs(OUT_ROOT, exist_ok=True)

MAX_SAM_PROMPTS = 3000
PROMPT_SUBSAMPLE_MODE = "random"   # "random" oder "first"
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# Save label mask TIFF always
SAVE_LABEL_TIF = True

# Keep plots
SAVE_PEBBLE_PNG = True

# Histogram only for sample (saves a lot of time)
SAVE_HIST_SAMPLE = True
N_HIST_PER_FOLDER = 100  # random tiles per folder for histogram PNGs

# Save GPKG only for sample
SAVE_GPKG_SAMPLE = True
N_GPKG_PER_FOLDER = 100  # random tiles per folder

# SEG plotting: avoid drawing ellipse axes from segmenteverygrain
SEG_PLOT_IMAGE = False   # <- important: prevents a/b axes overlays

def _gpu_free_gb():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / 1024**3
    return None

def setup_out_folder(in_folder: str, out_root: str):
    folder_name = os.path.basename(os.path.normpath(in_folder))
    out_folder = os.path.join(out_root, folder_name)

    out_gpkg   = os.path.join(out_folder, "ouputgpkg")
    out_csv    = os.path.join(out_folder, "csv_stats")
    out_plot   = os.path.join(out_folder, "pebbleplots")
    out_hist   = os.path.join(out_folder, "histplots")
    out_masks  = os.path.join(out_folder, "masktifs")

    for p in [out_gpkg, out_csv, out_plot, out_hist, out_masks]:
        os.makedirs(p, exist_ok=True)

    return out_folder, out_gpkg, out_csv, out_plot, out_hist, out_masks

def collect_tiles(folder: str):
    inputtiledir = os.path.join(folder, "inputtiles")
    tiles_in_input = sorted(glob.glob(os.path.join(inputtiledir, "*.tif")))
    tiles_in_root  = sorted(glob.glob(os.path.join(folder, "*.tif")))
    tiles = tiles_in_input if len(tiles_in_input) > 0 else tiles_in_root

    source = "inputtiles" if len(tiles_in_input) > 0 else "folder root"
    print(f"Found {len(tiles)} tiles in {folder} ({source})")

    if len(tiles) == 0:
        raise RuntimeError(f"No tiles found in {inputtiledir} or in {folder}")

    return tiles

def process_one_folder(folder: str, out_root: str = OUT_ROOT):
    tiles = collect_tiles(folder)

    out_folder, out_gpkg, out_csv, out_plot, out_hist, out_masks = setup_out_folder(folder, out_root)
    print("OUT_FOLDER:", out_folder)

    tile_ids_all = [os.path.splitext(os.path.basename(p))[0] for p in tiles]

    # sample for GPKG export
    if SAVE_GPKG_SAMPLE:
        if len(tile_ids_all) <= N_GPKG_PER_FOLDER:
            gpkg_tile_ids = set(tile_ids_all)
        else:
            gpkg_tile_ids = set(rng.choice(tile_ids_all, size=N_GPKG_PER_FOLDER, replace=False))
        print(f"GPKG sample: {len(gpkg_tile_ids)} / {len(tile_ids_all)} tiles")
    else:
        gpkg_tile_ids = set()

    # sample for histogram plots
    if SAVE_HIST_SAMPLE:
        if len(tile_ids_all) <= N_HIST_PER_FOLDER:
            hist_tile_ids = set(tile_ids_all)
        else:
            hist_tile_ids = set(rng.choice(tile_ids_all, size=N_HIST_PER_FOLDER, replace=False))
        print(f"HIST sample: {len(hist_tile_ids)} / {len(tile_ids_all)} tiles")
    else:
        hist_tile_ids = set()

    # read pixel size once from first tile
    with rasterio.open(tiles[0]) as src:
        xres, yres = src.res
        crs = src.crs

    gsd_units = float((xres + yres) / 2.0)
    gsd_mm = gsd_units * 1000.0

    minor_mm = 20.0
    minor_px = minor_mm / gsd_mm
    min_area_px = float(np.pi * (minor_px / 2.0) ** 2)

    print("CRS:", crs)
    print(f"Pixel size: {gsd_units:.6f} units/px (~{gsd_mm:.3f} mm/px)")
    print(f"2 cm => minor_px={minor_px:.2f} px -> min_area_px={min_area_px:.1f} px^2")

    rows = []
    metrics_csv = os.path.join(out_folder, "runtime_metrics.csv")
    pd.DataFrame([]).to_csv(metrics_csv, index=False)  # creates empty file early

    for i, fname in enumerate(tiles, start=1):
        tile_id = os.path.splitext(os.path.basename(fname))[0]
        print(f"[{i}/{len(tiles)}] {tile_id}")
        with open(os.path.join(out_folder, "_progress.txt"), "w") as f:
            f.write(f"{i}/{len(tiles)} {tile_id}\n")

        t0 = time.perf_counter()
        gpu_free_before = _gpu_free_gb()

        rec = {
            "folder": os.path.basename(folder),
            "tile_id": tile_id,
            "fname": fname,
            "status": None,

            "crs": str(crs),
            "pixel_size_units_per_px": gsd_units,
            "pixel_size_mm_per_px": gsd_mm,
            "minor_mm_threshold": minor_mm,
            "minor_px_threshold": minor_px,
            "min_area_px": min_area_px,

            "H": None,
            "W": None,
            "n_prompts_before": None,
            "n_prompts_used": None,
            "prompt_cap_used": None,
            "n_grains": None,

            "t_unet_s": None,
            "t_label_s": None,
            "t_sam_s": None,
            "t_export_s": None,
            "t_total_s": None,

            "gpu_free_gb_before": gpu_free_before,
            "gpu_free_gb_after": None,

            "error_msg": None,
            "traceback_head": None,
        }

        try:
            # nodata check
            with rasterio.open(fname) as src:
                m = src.dataset_mask()
                if not np.any(m):
                    print(" -> skipped (100% Nodata)")
                    rec["status"] = "skip_nodata"
                    rec["t_total_s"] = time.perf_counter() - t0
                    rec["gpu_free_gb_after"] = _gpu_free_gb()
                    rows.append(rec)
                    continue

            # load + predict (UNet)
            t = time.perf_counter()
            image = np.array(load_img(fname))
            rec["H"], rec["W"] = int(image.shape[0]), int(image.shape[1])
            image_pred = seg.predict_image(image, unet, I=256)
            rec["t_unet_s"] = time.perf_counter() - t

            # prompts
            t = time.perf_counter()
            labels_pts, coords = seg.label_grains(image, image_pred, dbs_max_dist=10.0)
            rec["t_label_s"] = time.perf_counter() - t

            coords = np.asarray(coords)
            rec["n_prompts_before"] = int(len(coords))
            rec["prompt_cap_used"] = False

            if MAX_SAM_PROMPTS is not None and len(coords) > MAX_SAM_PROMPTS:
                rec["prompt_cap_used"] = True
                n_before = len(coords)

                if PROMPT_SUBSAMPLE_MODE == "first":
                    keep_idx = np.arange(MAX_SAM_PROMPTS)
                else:
                    keep_idx = np.sort(rng.choice(len(coords), size=MAX_SAM_PROMPTS, replace=False))

                coords = coords[keep_idx]

                try:
                    labels_arr = np.asarray(labels_pts)
                    if labels_arr.ndim == 1 and len(labels_arr) == n_before:
                        labels_pts = labels_arr[keep_idx]
                except Exception:
                    pass

                print(f"Prompt cap active: reduced prompts from {n_before} -> {len(coords)}")

            rec["n_prompts_used"] = int(len(coords))

            # SAM segmentation
            t = time.perf_counter()
            all_grains, labels, mask_all, grain_data, fig, ax = seg.sam_segmentation(
                sam,
                image,
                image_pred,
                coords,
                labels_pts,
                min_area=min_area_px,
                plot_image=SEG_PLOT_IMAGE,   # <- no built-in axis overlays
                remove_edge_grains=True,
                remove_large_objects=True,
            )
            rec["t_sam_s"] = time.perf_counter() - t
            rec["n_grains"] = int(len(all_grains))

            # export/post
            t = time.perf_counter()

            # 1) pebble PNG (fallback plot, no axes, no a/b axis overlays)
            if SAVE_PEBBLE_PNG:
                seg_plot_path = os.path.join(out_plot, f"{tile_id}.png")

                if fig is None:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(image)
                    for poly in all_grains:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, linewidth=0.8)
                    ax.axis("off")
                else:
                    try:
                        for a in fig.axes:
                            a.axis("off")
                    except Exception:
                        pass

                fig.savefig(seg_plot_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

            # 2) label mask GeoTIFF (0..N), georeferenced, filtered result from SEG
            if SAVE_LABEL_TIF:
                with rasterio.open(fname) as ds:
                    prof = ds.profile.copy()
                    prof.update(driver="GTiff", count=1, dtype="uint32", compress="lzw")
                    out_mask = os.path.join(out_masks, f"{tile_id}_labels.tif")
                    with rasterio.open(out_mask, "w", **prof) as dst:
                        dst.write(labels.astype("uint32"), 1)

            # 3) stats from labels (fast)
            with rasterio.open(fname) as dataset:
                px_size_x = abs(dataset.transform.a)

                props = regionprops_table(
                    labels.astype("int"),
                    properties=("label", "area", "centroid", "major_axis_length", "minor_axis_length"),
                )
                grain_stats = pd.DataFrame(props)

                if len(grain_stats) > 0:
                    centroid_x, centroid_y = rasterio.transform.xy(
                        dataset.transform,
                        grain_stats["centroid-0"].values,
                        grain_stats["centroid-1"].values,
                    )

                    grain_stats["centroid_x"] = centroid_x
                    grain_stats["centroid_y"] = centroid_y
                    grain_stats.rename(columns={"area": "area_px"}, inplace=True)
                    grain_stats["major_axis_px"] = grain_stats["major_axis_length"]
                    grain_stats["minor_axis_px"] = grain_stats["minor_axis_length"]
                    grain_stats["major_axis_length"] = grain_stats["major_axis_length"] * px_size_x
                    grain_stats["minor_axis_length"] = grain_stats["minor_axis_length"] * px_size_x
                    grain_stats["major_axis_mm"] = grain_stats["major_axis_length"] * 1000.0
                    grain_stats["minor_axis_mm"] = grain_stats["minor_axis_length"] * 1000.0

                    out_cols = [
                        "label", "area_px", "centroid_x", "centroid_y",
                        "major_axis_px", "minor_axis_px",
                        "major_axis_length", "minor_axis_length",
                        "major_axis_mm", "minor_axis_mm",
                    ]
                    grain_stats_out = grain_stats[out_cols].copy()
                else:
                    grain_stats_out = pd.DataFrame(columns=[
                        "label","area_px","centroid_x","centroid_y",
                        "major_axis_px","minor_axis_px",
                        "major_axis_length","minor_axis_length",
                        "major_axis_mm","minor_axis_mm"
                    ])

            csv_path = os.path.join(out_csv, f"{tile_id}_grain_stats.csv")
            grain_stats_out.to_csv(csv_path, index=False)

            # 4) histogram PNG only for sampled tiles
            if SAVE_HIST_SAMPLE and (tile_id in hist_tile_ids) and len(grain_stats_out) > 0:
                fig_hist, ax_hist = seg.plot_histogram_of_axis_lengths(
                    grain_stats_out["major_axis_mm"],
                    grain_stats_out["minor_axis_mm"],
                    binsize=0.25,
                    xlimits=[8, 2 * 256],
                )
                hist_plot_path = os.path.join(out_hist, f"{tile_id}_hist.png")
                fig_hist.savefig(hist_plot_path, dpi=200, bbox_inches="tight")
                plt.close(fig_hist)

            # 5) GPKG only for sampled tiles
            if SAVE_GPKG_SAMPLE and (tile_id in gpkg_tile_ids):
                with rasterio.open(fname) as dataset:
                    projected_polys = []
                    for grain in all_grains:
                        px_x = np.asarray(grain.exterior.xy[0])
                        px_y = np.asarray(grain.exterior.xy[1])
                        x_proj, y_proj = rasterio.transform.xy(dataset.transform, px_y, px_x)
                        poly = Polygon(np.vstack((x_proj, y_proj)).T)
                        projected_polys.append(poly)

                    gdf = gpd.GeoDataFrame({"geometry": projected_polys}, geometry="geometry", crs=dataset.crs)

                gpkg_path = os.path.join(out_gpkg, f"{tile_id}_grains.gpkg")
                gdf.to_file(gpkg_path, driver="GPKG")

            rec["t_export_s"] = time.perf_counter() - t
            rec["status"] = "ok"
            rec["t_total_s"] = time.perf_counter() - t0
            rec["gpu_free_gb_after"] = _gpu_free_gb()
            rows.append(rec)
            pd.DataFrame(rows).to_csv(metrics_csv, index=False)

        except Exception as e:
            rec["status"] = "error"
            rec["t_total_s"] = time.perf_counter() - t0
            rec["gpu_free_gb_after"] = _gpu_free_gb()
            rec["error_msg"] = str(e)
            rec["traceback_head"] = traceback.format_exc(limit=8)
            rows.append(rec)
            pd.DataFrame(rows).to_csv(metrics_csv, index=False)
            print("ERROR on", tile_id, ":", e)

    # runtime metrics for this folder
    df = pd.DataFrame(rows)
    metrics_csv = os.path.join(out_folder, "runtime_metrics.csv")
    df.to_csv(metrics_csv, index=False)
    print("Saved runtime metrics CSV:", metrics_csv)

    # ready table
    ok = df[df["status"] == "ok"].copy()
    n_ok = len(ok)
    total_s = ok["t_total_s"].sum()
    tiles_per_min = (n_ok / (total_s / 60.0)) if total_s > 0 else np.nan

    ready = pd.DataFrame({
        "metric": [
            "n_tiles_total",
            "n_tiles_ok",
            "n_tiles_skipped_nodata",
            "n_tiles_error",
            "total_runtime_min",
            "tiles_per_min",
            "total_s_per_tile (median)",
            "unet_s (median)",
            "label_s (median)",
            "sam_s (median)",
            "export_s (median)",
            "prompts_used (median)",
            "grains (median)",
        ],
        "value": [
            int(len(df)),
            int(n_ok),
            int((df["status"] == "skip_nodata").sum()),
            int((df["status"] == "error").sum()),
            float(total_s / 60.0) if n_ok else np.nan,
            float(tiles_per_min) if n_ok else np.nan,
            float(ok["t_total_s"].median()) if n_ok else np.nan,
            float(ok["t_unet_s"].median()) if n_ok else np.nan,
            float(ok["t_label_s"].median()) if n_ok else np.nan,
            float(ok["t_sam_s"].median()) if n_ok else np.nan,
            float(ok["t_export_s"].median()) if n_ok else np.nan,
            float(ok["n_prompts_used"].median()) if n_ok else np.nan,
            float(ok["n_grains"].median()) if n_ok else np.nan,
        ]
    })

    ready_csv = os.path.join(out_folder, "runtime_summary_ready_table.csv")
    ready.to_csv(ready_csv, index=False)
    print("Saved ready table CSV:", ready_csv)

    return df, ready


import os, glob

BASE = "/dss/dsstbyfs02/pr94no/pr94no-dss-0001/drylands/gravel_leonie_masterthesis/segmenteverygrain"
folders = sorted(glob.glob(os.path.join(BASE, "F*")))

MAX_TO_PROCESS = 50
processed = 0

for folder in folders:
    if processed >= MAX_TO_PROCESS:
        print("Reached MAX_TO_PROCESS, stopping.")
        break

    # 1) skip wenn schon Ergebnisse (GPKG) vorhanden
    gpkg_files = glob.glob(os.path.join(folder, "ouputgpkg", "*.gpkg"))
    if len(gpkg_files) > 0:
        print("SKIP already has GPKG:", os.path.basename(folder), "n_gpkg:", len(gpkg_files))
        continue

    # 2) skip wenn noch keine tiles drin sind (inputtiles/ ODER folder-root)
    tiles = glob.glob(os.path.join(folder, "inputtiles", "*.tif"))
    if len(tiles) == 0:
        tiles = glob.glob(os.path.join(folder, "*.tif"))

    if len(tiles) == 0:
        print("SKIP no tiles yet:", os.path.basename(folder))
        continue

    print("PROCESS:", os.path.basename(folder), "tiles:", len(tiles))
    process_one_folder(folder)
    processed += 1