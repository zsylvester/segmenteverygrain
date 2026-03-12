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

import torch

# %matplotlib qt



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



import glob
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img
import rasterio



import geopandas as gpd
from shapely.geometry import Polygon
from skimage.measure import regionprops_table

# --- optional: prompt cap for SAM (None = no limit) ---
MAX_SAM_PROMPTS = 3000
PROMPT_SUBSAMPLE_MODE = "random"   # "random" oder "first"
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

def setup_folder(folder_path: str):
    os.makedirs(os.path.join(folder_path, "ouputgpkg"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "inputtiles"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "pebbleplots"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "histplots"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "csv_stats"), exist_ok=True)

    inputtiledir  = os.path.join(folder_path, "inputtiles")
    ouputgpkg     = os.path.join(folder_path, "ouputgpkg")
    csvdir        = os.path.join(folder_path, "csv_stats")
    plotdirgravel = os.path.join(folder_path, "pebbleplots")
    plotdirhist   = os.path.join(folder_path, "histplots")

    return inputtiledir, ouputgpkg, csvdir, plotdirgravel, plotdirhist

def _gpu_free_gb():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / 1024**3
    return None

def process_one_folder(folder: str):
    inputtiledir, ouputgpkg, csvdir, plotdirgravel, plotdirhist = setup_folder(folder)

        # --- collect tiles ---
    tiles_in_input = sorted(glob.glob(os.path.join(inputtiledir, "*.tif")))
    tiles_in_root  = sorted(glob.glob(os.path.join(folder, "*.tif")))
    
    tiles = tiles_in_input if len(tiles_in_input) > 0 else tiles_in_root
    
    print(f"Found {len(tiles)} tiles in {folder} "
          f"({'inputtiles' if len(tiles_in_input) > 0 else 'folder root'})")
    
    if len(tiles) == 0:
        raise RuntimeError(f"No tiles found in {inputtiledir} or in {folder}")

 
    # --- read pixel size once from first tile ---
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

    # --- loop ---
    for i, fname in enumerate(tiles, start=1):
        tile_id = os.path.splitext(os.path.basename(fname))[0]
        print(f"[{i}/{len(tiles)}] {tile_id}")

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
            # ---- nodata check ----
            with rasterio.open(fname) as src:
                m = src.dataset_mask()
                if not np.any(m):
                    print(" -> skipped (100% Nodata)")
                    rec["status"] = "skip_nodata"
                    rec["t_total_s"] = time.perf_counter() - t0
                    rec["gpu_free_gb_after"] = _gpu_free_gb()
                    rows.append(rec)
                    continue

            # ---- load + predict (UNet) ----
            t = time.perf_counter()
            image = np.array(load_img(fname))
            rec["H"], rec["W"] = int(image.shape[0]), int(image.shape[1])
            image_pred = seg.predict_image(image, unet, I=256)
            rec["t_unet_s"] = time.perf_counter() - t

            # ---- prompts ----
            t = time.perf_counter()
            labels, coords = seg.label_grains(image, image_pred, dbs_max_dist=10.0)
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
                    labels_arr = np.asarray(labels)
                    if labels_arr.ndim == 1 and len(labels_arr) == n_before:
                        labels = labels_arr[keep_idx]
                except Exception:
                    pass

                print(f"Prompt cap active: reduced prompts from {n_before} -> {len(coords)}")

            rec["n_prompts_used"] = int(len(coords))

            # ---- SAM segmentation ----
            t = time.perf_counter()
            all_grains, labels, mask_all, grain_data, fig, ax = seg.sam_segmentation(
                sam,
                image,
                image_pred,
                coords,
                labels,
                min_area=min_area_px,
                plot_image=True,
                remove_edge_grains=True,
                remove_large_objects=True,
            )
            rec["t_sam_s"] = time.perf_counter() - t
            rec["n_grains"] = int(len(all_grains))

            # ---- export/post ----
            t = time.perf_counter()

            # 1) segmentation plot (fallback if fig None)
            seg_plot_path = os.path.join(plotdirgravel, f"{tile_id}.png")
            if fig is None:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(image)
                for poly in all_grains:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, linewidth=0.8)
                ax.set_title(tile_id)
                ax.axis("off")

            fig.savefig(seg_plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print("Saved segmentation plot")

            # 2) georef polygons + stats
            with rasterio.open(fname) as dataset:
                projected_polys = []
                for grain in all_grains:
                    px_x = np.asarray(grain.exterior.xy[0])
                    px_y = np.asarray(grain.exterior.xy[1])

                    x_proj, y_proj = rasterio.transform.xy(dataset.transform, px_y, px_x)
                    poly = Polygon(np.vstack((x_proj, y_proj)).T)
                    projected_polys.append(poly)

                gdf = gpd.GeoDataFrame({"geometry": projected_polys}, geometry="geometry", crs=dataset.crs)

                props = regionprops_table(
                    labels.astype("int"),
                    properties=("label", "area", "centroid", "major_axis_length", "minor_axis_length"),
                )
                grain_stats = pd.DataFrame(props)

                if len(grain_stats) != len(gdf):
                    nmin = min(len(grain_stats), len(gdf))
                    print(f"Warning: len(gdf)={len(gdf)} != len(grain_stats)={len(grain_stats)} -> truncating to {nmin}")
                    gdf = gdf.iloc[:nmin].copy()
                    grain_stats = grain_stats.iloc[:nmin].copy()

                centroid_x, centroid_y = rasterio.transform.xy(
                    dataset.transform,
                    grain_stats["centroid-0"].values,
                    grain_stats["centroid-1"].values,
                )

                px_size_x = abs(dataset.transform.a)

                gdf["label"] = grain_stats["label"].values
                gdf["area_px"] = grain_stats["area"].values
                gdf["centroid_x"] = centroid_x
                gdf["centroid_y"] = centroid_y
                gdf["major_axis_px"] = grain_stats["major_axis_length"].values
                gdf["minor_axis_px"] = grain_stats["minor_axis_length"].values
                gdf["major_axis_length"] = grain_stats["major_axis_length"].values * px_size_x
                gdf["minor_axis_length"] = grain_stats["minor_axis_length"].values * px_size_x
                gdf["major_axis_mm"] = gdf["major_axis_length"] * 1000.0
                gdf["minor_axis_mm"] = gdf["minor_axis_length"] * 1000.0

            # 3) histogram plot
            if len(gdf) > 0:
                fig_hist, ax_hist = seg.plot_histogram_of_axis_lengths(
                    gdf["major_axis_mm"],
                    gdf["minor_axis_mm"],
                    binsize=0.25,
                    xlimits=[8, 2 * 256],
                )
                hist_plot_path = os.path.join(plotdirhist, f"{tile_id}_hist.png")
                fig_hist.savefig(hist_plot_path, dpi=200, bbox_inches="tight")
                plt.close(fig_hist)
                print("Saved histogram plot")
            else:
                print("No grains found -> skipping histogram plot")

            # 4) write gpkg + csv
            gpkg_path = os.path.join(ouputgpkg, f"{tile_id}_grains.gpkg")
            csv_path = os.path.join(csvdir, f"{tile_id}_grain_stats.csv")

            gdf.to_file(gpkg_path, driver="GPKG")
            gdf.drop(columns="geometry").to_csv(csv_path, index=False)

            print(f"Saved GPKG: {gpkg_path}")
            print(f"Saved stats CSV: {csv_path}")

            rec["t_export_s"] = time.perf_counter() - t

            rec["status"] = "ok"
            rec["t_total_s"] = time.perf_counter() - t0
            rec["gpu_free_gb_after"] = _gpu_free_gb()
            rows.append(rec)

            print("done with segmentation + export")

        except Exception as e:
            rec["status"] = "error"
            rec["t_total_s"] = time.perf_counter() - t0
            rec["gpu_free_gb_after"] = _gpu_free_gb()
            rec["error_msg"] = str(e)
            rec["traceback_head"] = traceback.format_exc(limit=8)
            rows.append(rec)
            print("ERROR on", tile_id, ":", e)

    # ---- save runtime metrics (Excel-friendly CSV) ----
    df = pd.DataFrame(rows)

    metrics_csv = os.path.join(folder, "runtime_metrics.csv")
    df.to_csv(metrics_csv, index=False)
    print("Saved runtime metrics CSV:", metrics_csv)

    # optional: quick describe
    summary = df[df["status"] == "ok"][["t_total_s","t_unet_s","t_label_s","t_sam_s","t_export_s","n_prompts_used","n_grains"]].describe()
    print(summary)

    # ---- paper-ready summary table (median-focused) ----
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
            "pixel_size_mm_per_px (median)",
            "min_area_px (median)",
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
            float(ok["pixel_size_mm_per_px"].median()) if n_ok else np.nan,
            float(ok["min_area_px"].median()) if n_ok else np.nan,
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

    ready_disp = ready.copy()
    ready_disp["value"] = ready_disp["value"].apply(
        lambda v: round(v, 3) if isinstance(v, (float, np.floating)) and pd.notnull(v) else v
    )
    print("\nREADY TABLE:\n")
    print(ready_disp.to_string(index=False))

    ready_csv = os.path.join(folder, "runtime_summary_ready_table.csv")
    ready.to_csv(ready_csv, index=False)
    print("\nSaved ready table CSV:", ready_csv)

    return df, ready




import os, glob

BASE = "/dss/dsstbyfs02/pr94no/pr94no-dss-0001/drylands/gravel_leonie_masterthesis/segmenteverygrain"
folders = sorted(glob.glob(os.path.join(BASE, "F*")))

MAX_TO_PROCESS = 10
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