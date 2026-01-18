"""
segmenteverygrain: A Python package for automated grain segmentation using SAM and U-Net.

This package provides tools for:
- Semantic segmentation of grains using U-Net
- Instance segmentation using the Segment Anything Model (SAM)
- Interactive editing of grain segmentations
- Extraction of individual grain images for analysis
"""

# Core segmentation functionality
from .segmenteverygrain import *

# Interactive editing tools
from .interactions import (
    GrainPlot,
    Grain,
    load_image,
    polygons_to_grains,
    save_grains,
    save_summary,
    save_histogram,
    save_mask,
    get_summary,
)

# Grain extraction utilities
from .grain_utils import (
    extract_grain_image,
    extract_all_grains,
    make_square,
    extract_vgg16_features,
    cluster_grains,
    create_grain_panel_cluster,
    create_grain_panel,
    create_clustered_grain_montage,
    ClusterMontageSelector,
    ClusterMontageLabeler,
    plot_classified_grains,
    extract_color_features,
)

__version__ = "0.2.5"
