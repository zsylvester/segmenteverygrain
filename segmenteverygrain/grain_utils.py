"""
Utility functions for extracting and processing individual grain images.

This module provides functions to extract individual grains from segmented images,
rotate them to a standard orientation, and prepare them for further analysis or
classification tasks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from shapely.affinity import translate
from skimage.measure import regionprops, label as label_image
from skimage.draw import polygon as draw_polygon
from skimage import exposure
from keras.applications import VGG16, ResNet50, InceptionV3
from keras.models import Model
from keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
from keras.applications.inception_v3 import preprocess_input as preprocess_inception
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA


def extract_grain_image(grain_polygon, image, image_pred=None, target_size=224, pad=10):
    """
    Extract a single grain from an image and create a square, unrotated image.
    
    This function takes a grain polygon, extracts the corresponding region from the image,
    applies padding, rotates it to align the major axis vertically, makes it square by
    padding with zeros, and resizes it to the target size.
    
    Parameters
    ----------
    grain_polygon : shapely.Polygon
        The grain polygon from all_grains
    image : numpy.ndarray
        The original image (H, W, 3)
    image_pred : numpy.ndarray, optional
        The U-Net prediction (H, W, 3)
    target_size : int
        Output image size (default: 224x224)
    pad : int
        Padding around the grain bounding box in pixels
        
    Returns
    -------
    grain_image : numpy.ndarray
        The extracted grain image (target_size, target_size, 3)
    grain_mask : numpy.ndarray
        Binary mask of the grain (target_size, target_size)
    grain_pred : numpy.ndarray or None
        The prediction for the grain region (target_size, target_size) if image_pred provided
    original_size : int
        The size of the square crop before resizing
        
    Examples
    --------
    >>> grain_img, grain_mask, grain_pred, orig_size = extract_grain_image(
    ...     all_grains[0], image, image_pred, target_size=224, pad=10
    ... )
    """
    # Get bounding box of the grain
    minx, miny, maxx, maxy = grain_polygon.bounds
    minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
    
    # Add padding (but stay within image bounds)
    img_height, img_width = image.shape[:2]
    minx = max(0, minx - pad)
    miny = max(0, miny - pad)
    maxx = min(img_width, maxx + pad)
    maxy = min(img_height, maxy + pad)
    
    # Create a mask for this specific grain in the cropped region
    crop_width = maxx - minx
    crop_height = maxy - miny
    
    # Translate polygon to crop coordinates
    grain_local = translate(grain_polygon, xoff=-minx, yoff=-miny)
    
    # Create mask
    mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    if grain_local.exterior is not None:
        coords = np.array(grain_local.exterior.coords)
        rr, cc = draw_polygon(coords[:, 1], coords[:, 0], mask.shape)
        mask[rr, cc] = 1
    
    # Extract the grain region from the image
    grain_crop = image[miny:maxy, minx:maxx].copy()
    grain_crop[mask == 0] = 0  # Zero out non-grain pixels
    
    # Extract prediction crop BEFORE squaring (while mask dimensions still match)
    pred_crop_masked = None
    if image_pred is not None:
        pred_crop = image_pred[miny:maxy, minx:maxx, 2].copy()  # Channel 2
        pred_crop_masked = pred_crop.copy()
        pred_crop_masked[mask == 0] = 0  # Use original mask dimensions here
    
    # Get orientation for rotation
    # For simplicity, calculate orientation using regionprops
    temp_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    coords = np.array(grain_polygon.exterior.coords)
    rr, cc = draw_polygon(coords[:, 1], coords[:, 0], temp_mask.shape)
    temp_mask[rr, cc] = 1
    temp_labeled = label_image(temp_mask)
    if temp_labeled.max() > 0:
        props = regionprops(temp_labeled)[0]
        orientation = props.orientation
    else:
        orientation = 0
    
    # Make square by padding with zeros
    grain_crop = make_square(grain_crop, mask)
    mask_squared = make_square(mask, mask)
    if pred_crop_masked is not None:
        pred_crop_masked = make_square(pred_crop_masked, mask)
    
    # Rotate to upright position (expand=True allows the canvas to grow to fit rotated content)
    grain_crop = np.array(Image.fromarray(grain_crop).rotate(
        -np.rad2deg(orientation), fillcolor=0, expand=True
    ))
    mask_rotated = np.array(Image.fromarray((mask_squared * 255).astype(np.uint8)).rotate(
        -np.rad2deg(orientation), fillcolor=0, expand=True
    )) / 255.0
    
    # After rotation with expand=True, the image may no longer be square, so make it square again
    grain_crop = make_square(grain_crop)
    mask_rotated = make_square(mask_rotated)
    
    original_size = grain_crop.shape[0]
    
    # Resize to target size
    grain_image = tf.image.resize(grain_crop, (target_size, target_size), antialias=False)
    grain_image = np.array(grain_image).astype('uint8')
    
    grain_mask = tf.image.resize(mask_rotated[..., None], (target_size, target_size), antialias=True)
    grain_mask = np.array(grain_mask).astype('uint8')
    
    # Handle prediction if provided
    grain_pred = None
    if pred_crop_masked is not None:
        pred_crop_rotated = np.array(Image.fromarray(pred_crop_masked).rotate(
            -np.rad2deg(orientation), fillcolor=0, expand=True
        ))
        pred_crop_rotated = make_square(pred_crop_rotated)
        grain_pred = tf.image.resize(pred_crop_rotated[..., None], (target_size, target_size), antialias=False)
    
    return grain_image, grain_mask, grain_pred, original_size

def make_square(im, reference_mask=None):
    """
    Pad an image to make it square.
    
    This function pads an image with zeros to make its height and width equal.
    The padding is distributed evenly on both sides.
    
    Parameters
    ----------
    im : numpy.ndarray
        Image to pad (H, W) or (H, W, C)
    reference_mask : numpy.ndarray, optional
        If provided, uses this mask's shape for determining padding
        
    Returns
    -------
    im_square : numpy.ndarray
        Square image with zero padding
        
    Examples
    --------
    >>> image = np.random.rand(100, 150, 3)
    >>> square_image = make_square(image)
    >>> square_image.shape
    (150, 150, 3)
    """
    if reference_mask is not None:
        shape = reference_mask.shape
    else:
        shape = im.shape[:2]
    
    r, c = shape[:2]
    
    if r == c:
        return im
    
    # Determine padding
    if r > c:
        # Pad columns
        pad_total = r - c
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        if len(im.shape) == 3:
            pad_width = ((0, 0), (pad_left, pad_right), (0, 0))
        else:
            pad_width = ((0, 0), (pad_left, pad_right))
    else:
        # Pad rows
        pad_total = c - r
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        
        if len(im.shape) == 3:
            pad_width = ((pad_top, pad_bottom), (0, 0), (0, 0))
        else:
            pad_width = ((pad_top, pad_bottom), (0, 0))
    
    return np.pad(im, pad_width, mode='constant', constant_values=0)

def extract_all_grains(all_grains, image, image_pred=None, target_size=224):
    """
    Extract all grains from an image.
    
    This function processes all grains in a list and extracts standardized,
    square images for each grain suitable for classification or analysis tasks.
    
    Parameters
    ----------
    all_grains : list
        List of shapely.Polygon objects representing grains
    image : numpy.ndarray
        The original image (H, W, 3)
    image_pred : numpy.ndarray, optional
        The U-Net prediction (H, W, 3)
    target_size : int
        Output image size (default: 224x224)
    
    Returns
    -------
    grain_images : list
        List of grain images, each of shape (target_size, target_size, 3)
    grain_masks : list
        List of grain masks, each of shape (target_size, target_size, 1)
    grain_preds : list
        List of grain predictions (or None if image_pred not provided)
        
    Examples
    --------
    >>> grain_images, grain_masks, grain_preds = extract_all_grains(
    ...     all_grains, image, image_pred, target_size=224
    ... )
    >>> print(f"Extracted {len(grain_images)} grains")
    """
    grain_images = []
    grain_masks = []
    grain_preds = []
    
    for grain in tqdm(all_grains, desc='Extracting grains'):
        img, mask, pred, _ = extract_grain_image(
            grain, image, image_pred, target_size=target_size
        )
        grain_images.append(img)
        grain_masks.append(mask)
        grain_preds.append(pred)
    
    return grain_images, grain_masks, grain_preds

def extract_vgg16_features(grain_images, model_name='VGG16', layer_name='fc2', 
                          include_preprocessing=True, batch_size=32, verbose=0):
    """
    Extract deep learning features from grain images using a pre-trained CNN.
    
    Images are automatically resized to 224x224 if they are not already that size.
    
    Parameters
    ----------
    grain_images : list or numpy.ndarray
        List of grain images, each of shape (H, W, 3). Images will be 
        automatically resized to 224x224 if necessary.
    model_name : str
        Name of the pre-trained model to use ('VGG16', 'ResNet50', etc.)
    layer_name : str
        Layer to extract features from. Options:
        - 'fc2' or -2: Second-to-last layer (default, 4096 features for VGG16)
        - 'fc1' or -3: Third-to-last layer
        - 'flatten': Flattened convolutional features
    include_preprocessing : bool
        Whether to apply model-specific preprocessing (recommended)
    batch_size : int
        Batch size for prediction (default: 32)
    verbose : int
        Verbosity level for model.predict (default: 0)
    
    Returns
    -------
    features : numpy.ndarray
        Extracted features of shape (n_grains, n_features)
    model : keras.Model
        The feature extraction model (useful for processing more grains later)
        
    Examples
    --------
    >>> features, model = extract_vgg16_features(grain_images)
    >>> print(features.shape)
    (336, 4096)
    
    >>> # Works with any size images - they will be auto-resized
    >>> features_640, model = extract_vgg16_features(grain_images_640x640)
    >>> print(features_640.shape)
    (336, 4096)
    """
    # Load the appropriate model
    if model_name.upper() == 'VGG16':
        base_model = VGG16(weights='imagenet')
        preprocess_fn = preprocess_vgg16
    elif model_name.upper() == 'RESNET50':
        base_model = ResNet50(weights='imagenet')
        preprocess_fn = preprocess_resnet
    elif model_name.upper() == 'INCEPTIONV3':
        base_model = InceptionV3(weights='imagenet')
        preprocess_fn = preprocess_inception
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Select the layer to extract features from
    if isinstance(layer_name, int) or (isinstance(layer_name, str) and layer_name.startswith('-')):
        layer_idx = int(layer_name)
        feature_layer = base_model.layers[layer_idx].output
    else:
        feature_layer = base_model.get_layer(layer_name).output
    
    # Create feature extraction model
    feature_model = Model(inputs=base_model.inputs, outputs=feature_layer)
    
    # Convert grain images to numpy array
    grain_batch = np.array(grain_images).astype('float32')
    
    # Check and resize if necessary
    if grain_batch.shape[1] != 224 or grain_batch.shape[2] != 224:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f'Resizing grain images from {grain_batch.shape[1]}x{grain_batch.shape[2]} to 224x224')
        
        # Resize each image to 224x224
        resized_images = []
        for img in grain_batch:
            resized_img = tf.image.resize(img, (224, 224), antialias=True)
            resized_images.append(np.array(resized_img))
        grain_batch = np.array(resized_images).astype('float32')
    
    if include_preprocessing:
        grain_batch = preprocess_fn(grain_batch)
    
    # Extract features
    features = feature_model.predict(grain_batch, batch_size=batch_size, verbose=verbose)
    
    # Flatten if necessary
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    return features, feature_model

def cluster_grains(features, n_clusters=10, method='kmeans', n_components=25, 
                   random_state=42, **kwargs):
    """
    Cluster grain features using dimensionality reduction and clustering.
    
    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix of shape (n_grains, n_features)
    n_clusters : int
        Number of clusters to create
    method : str
        Clustering method: 'kmeans', 'dbscan', 'hierarchical'
    n_components : int or None
        Number of PCA components. If None, no dimensionality reduction is applied
    random_state : int
        Random seed for reproducibility
    **kwargs : dict
        Additional arguments passed to the clustering algorithm
    
    Returns
    -------
    labels : numpy.ndarray
        Cluster labels for each grain
    reduced_features : numpy.ndarray
        Dimensionality-reduced features (or original if n_components is None)
    pca : sklearn.decomposition.PCA or None
        Fitted PCA object (None if n_components is None)
    clusterer : sklearn clustering object
        Fitted clustering model
        
    Examples
    --------
    >>> labels, reduced_feats, pca, kmeans = cluster_grains(
    ...     features, n_clusters=10, n_components=25
    ... )
    >>> print(f"Found {len(np.unique(labels))} clusters")
    """
    # Dimensionality reduction
    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=random_state)
        reduced_features = pca.fit_transform(features)
        print(f"PCA: {features.shape[1]} features → {n_components} components "
              f"(explained variance: {pca.explained_variance_ratio_.sum():.2%})")
    else:
        pca = None
        reduced_features = features
    
    # Clustering
    if method.lower() == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    elif method.lower() == 'dbscan':
        clusterer = DBSCAN(**kwargs)
    elif method.lower() == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    labels = clusterer.fit_predict(reduced_features)
    
    return labels, reduced_features, pca, clusterer

def create_grain_panel_cluster(cluster_number, labels, all_images):
    """
    Create a visual panel of grain images from a specific cluster.
    
    This function creates a square grid panel containing randomly selected grain images
    from a specified cluster. Each grain image is assumed to be 224x224 pixels.
    
    Parameters
    ----------
    cluster_number : int
        The cluster number to create a panel for
    labels : numpy.ndarray
        Array of cluster labels for each grain
    all_images : numpy.ndarray
        Array of all grain images, shape (n_grains, 224, 224, 3)
    
    Returns
    -------
    cl_panel : numpy.ndarray
        Combined panel image with shape (224*n_rows, 224*n_cols, 3)
    grain_inds : numpy.ndarray
        2D array containing the indices of grains used in the panel, 
        shape (n_rows, n_cols)
    
    Examples
    --------
    >>> panel, indices = create_grain_panel_cluster(0, cluster_labels, grain_images)
    >>> plt.imshow(panel.astype(int))
    >>> plt.show()
    """
    cluster = np.where(labels == cluster_number)[0]
    n_rows = n_cols = int(np.floor(len(cluster)**0.5))
    rand_inds = np.random.choice(cluster, size = n_cols*n_rows, replace = False)
    cl_panel = np.zeros((224*n_rows, 224*n_cols, 3))
    grain_inds = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            cl_panel[i*224:(i+1)*224, j*224:(j+1)*224, :] = all_images[rand_inds[i*n_cols + j]]
            grain_inds[i, j] = int(rand_inds[i*n_cols + j])
    cl_panel = exposure.equalize_adapthist(cl_panel.astype('uint8'))
    return cl_panel, grain_inds

def create_grain_panel(inds, all_images, n_rows, n_cols):
    """
    Create a visual panel of grain images from specified indices.
    
    This function creates a rectangular grid panel containing grain images
    from the provided indices. Each grain image is assumed to be 224x224 pixels.
    
    Parameters
    ----------
    inds : array-like
        Indices of grains to include in the panel
    all_images : numpy.ndarray
        Array of all grain images, shape (n_grains, 224, 224, 3)
    n_rows : int
        Number of rows in the panel grid
    n_cols : int
        Number of columns in the panel grid
    
    Returns
    -------
    cl_panel : numpy.ndarray
        Combined panel image with shape (224*n_rows, 224*n_cols, 3)
    
    Notes
    -----
    The number of indices should equal n_rows * n_cols.
    
    Examples
    --------
    >>> selected_indices = [0, 5, 10, 15]
    >>> panel = create_grain_panel(selected_indices, grain_images, 2, 2)
    >>> plt.imshow(panel.astype(int))
    >>> plt.show()
    """
    cl_panel = np.zeros((224*n_rows, 224*n_cols, 3))
    for i in range(n_rows):
        for j in range(n_cols):
            cl_panel[i*224:(i+1)*224, j*224:(j+1)*224, :] = all_images[inds[i*n_cols + j]]
    return cl_panel

def create_clustered_grain_montage(cluster_labels, grain_images, grid_cols=20,
                                   grain_size=224, padding=2, 
                                   draw_boundaries=True, boundary_thickness=2,
                                   all_grains=None, sort_by_size=False):
    """
    Create a single montage image with all grains arranged by cluster.
    
    Grains are arranged in a compact grid, ordered by cluster (cluster 0, then 1, etc.).
    Within each cluster, grains can optionally be sorted by size. Grains flow continuously 
    across rows to minimize white space. Cluster boundaries can be visualized with colored 
    borders around each grain.
    
    Parameters
    ----------
    cluster_labels : numpy.ndarray
        Array of cluster labels for each grain
    grain_images : list or numpy.ndarray
        Array of all grain images, shape (n_grains, H, W, 3)
    grid_cols : int
        Number of grain columns in the grid (default: 20)
    grain_size : int
        Size of each grain image in pixels (default: 224)
    padding : int
        Padding between grains in pixels (default: 2)
    draw_boundaries : bool
        If True, draws colored borders around each grain to indicate cluster (default: True)
    boundary_thickness : int
        Thickness of cluster boundary in pixels (default: 2)
    all_grains : list of shapely.Polygon, optional
        List of grain polygons. Required if sort_by_size=True
    sort_by_size : bool
        If True, sorts grains within each cluster by size (area) in descending order.
        Requires all_grains to be provided (default: False)
    
    Returns
    -------
    montage : numpy.ndarray
        Single large image containing all grains arranged by cluster
    cluster_info : dict
        Dictionary with information about each cluster:
        - 'n_grains': number of grains in cluster
        - 'grain_indices': indices of grains in this cluster
        - 'positions': list of (row, col) positions in the grid
    
    Examples
    --------
    >>> # With colored borders
    >>> montage, info = create_clustered_grain_montage(labels, grain_images, grid_cols=20)
    >>> plt.figure(figsize=(20, 15))
    >>> plt.imshow(montage.astype('uint8'))
    >>> plt.axis('off')
    >>> plt.show()
    
    >>> # Without borders (pure grid)
    >>> montage, info = create_clustered_grain_montage(
    ...     labels, grain_images, grid_cols=20, draw_boundaries=False
    ... )
    
    >>> # Sorted by size within each cluster
    >>> montage, info = create_clustered_grain_montage(
    ...     labels, grain_images, all_grains=all_grains, sort_by_size=True
    ... )
    
    >>> # Print cluster info
    >>> for cluster_id, data in info.items():
    ...     print(f"Cluster {cluster_id}: {data['n_grains']} grains")
    """
    
    if sort_by_size and all_grains is None:
        raise ValueError("all_grains must be provided when sort_by_size=True")
    grain_images = np.array(grain_images)
    unique_clusters = np.unique(cluster_labels)
    
    # Create ordered list of grain indices by cluster
    ordered_grain_indices = []
    cluster_info = {}
    
    for cluster_id in unique_clusters:
        grain_indices = np.where(cluster_labels == cluster_id)[0]
        
        # Sort by size (area) if requested
        if sort_by_size:
            grain_areas = [all_grains[idx].area for idx in grain_indices]
            # Sort in descending order (largest first)
            sorted_order = np.argsort(grain_areas)[::-1]
            grain_indices = grain_indices[sorted_order]
        
        ordered_grain_indices.extend(grain_indices)
        cluster_info[cluster_id] = {
            'n_grains': len(grain_indices),
            'grain_indices': grain_indices,
            'positions': []
        }
    
    # Calculate montage dimensions
    total_grains = len(ordered_grain_indices)
    n_rows = int(np.ceil(total_grains / grid_cols))
    
    total_height = n_rows * (grain_size + padding) + padding
    total_width = grid_cols * (grain_size + padding) + padding
    
    # Create montage with white background
    montage = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Get cluster colors if drawing boundaries
    if draw_boundaries:
        cluster_colors = {}
        for cluster_id in unique_clusters:
            color = plt.cm.tab10(cluster_id % 10)[:3]
            cluster_colors[cluster_id] = (np.array(color) * 255).astype(np.uint8)
    
    # Place all grains in continuous grid
    for i, grain_idx in enumerate(ordered_grain_indices):
        row = i // grid_cols
        col = i % grid_cols
        
        y_pos = padding + row * (grain_size + padding)
        x_pos = padding + col * (grain_size + padding)
        
        # Get cluster ID for this grain
        cluster_id = cluster_labels[grain_idx]
        
        # Record position for this grain's cluster
        cluster_info[cluster_id]['positions'].append((row, col))
        
        # Place the grain image
        grain_img = grain_images[grain_idx].copy()
        
        # Draw colored border if requested
        if draw_boundaries:
            border_color = cluster_colors[cluster_id]
            # Top and bottom borders
            grain_img[:boundary_thickness, :] = border_color
            grain_img[-boundary_thickness:, :] = border_color
            # Left and right borders
            grain_img[:, :boundary_thickness] = border_color
            grain_img[:, -boundary_thickness:] = border_color
        
        montage[y_pos:y_pos + grain_size, 
               x_pos:x_pos + grain_size] = grain_img
    
    return montage, cluster_info

class ClusterMontageSelector:
    """
    Interactive selector for clustered grain montage.
    
    Allows interactive selection and removal of entire clusters or individual grains
    for quality control. Click on a cluster to select/deselect all grains in it,
    or Shift+Click to select/deselect individual grains.
    
    Parameters
    ----------
    cluster_labels : numpy.ndarray
        Array of cluster labels for each grain
    grain_images : list or numpy.ndarray
        Array of all grain images
    all_grains : list
        List of grain polygons (shapely.Polygon objects)
    grid_cols : int
        Number of grain columns in the grid (default: 20)
    figsize : tuple
        Figure size (default: (20, 15))
    grain_size : int
        Size of each grain image in pixels (default: 224)
    padding : int
        Padding between grains in pixels (default: 2)
    boundary_thickness : int
        Thickness of cluster boundary in pixels (default: 2)
    sort_by_size : bool
        If True, sorts grains within each cluster by size (area) in descending order
        (default: False)
    
    Attributes
    ----------
    selected_grain_indices : set
        Set of grain indices that are selected for removal
    selected_clusters : set
        Set of cluster IDs that are fully selected
    
    Methods
    -------
    activate()
        Activate interactive selection mode
    deactivate()
        Deactivate interactive mode
    get_filtered_grains()
        Get the list of grains after removing selected ones
    get_filtered_labels()
        Get the cluster labels after removing selected grains
    get_filtered_images()
        Get the grain images after removing selected grains
    get_filtered_indices()
        Get the indices of grains that remain after filtering
    
    Interactive Controls
    --------------------
    - Left-click: Select/deselect entire cluster
    - Shift + Left-click: Select/deselect individual grain
    - 'd' key: Delete selected grains/clusters
    - 'r' key: Reset (clear all selections)
    - 'Esc' key: Clear all selections
    
    Examples
    --------
    >>> selector = ClusterMontageSelector(labels, grain_images, all_grains)
    >>> selector.activate()
    >>> # ... interact with the plot ...
    >>> filtered_grains = selector.get_filtered_grains()
    >>> print(f"Kept {len(filtered_grains)} out of {len(all_grains)} grains")
    """
    
    def __init__(self, cluster_labels, grain_images, all_grains, 
                 grid_cols=20, figsize=(20, 15), grain_size=224, 
                 padding=2, boundary_thickness=2, sort_by_size=False):
        
        self.cluster_labels = np.array(cluster_labels)
        self.grain_images = np.array(grain_images)
        self.all_grains = list(all_grains)
        self.grid_cols = grid_cols
        self.grain_size = grain_size
        self.padding = padding
        self.boundary_thickness = boundary_thickness
        self.sort_by_size = sort_by_size
        
        # Selection tracking
        self.selected_grain_indices = set()
        self.selected_clusters = set()
        self.removed_grain_indices = set()
        
        # Create the montage and get cluster info
        self.montage, self.cluster_info = create_clustered_grain_montage(
            self.cluster_labels, self.grain_images, 
            grid_cols=grid_cols, grain_size=grain_size,
            padding=padding, draw_boundaries=True,
            boundary_thickness=boundary_thickness,
            all_grains=self.all_grains, sort_by_size=sort_by_size
        )
        
        # Create position lookup: (row, col) -> grain_index
        self.position_to_grain = {}
        for cluster_id, info in self.cluster_info.items():
            for pos, grain_idx in zip(info['positions'], info['grain_indices']):
                self.position_to_grain[pos] = grain_idx
        
        # Create grain to cluster lookup
        self.grain_to_cluster = {}
        for cluster_id, info in self.cluster_info.items():
            for grain_idx in info['grain_indices']:
                self.grain_to_cluster[grain_idx] = cluster_id
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.display_montage()
        
        # Event handlers
        self.cid_click = None
        self.cid_key = None
        
    def display_montage(self):
        """Display the current montage with selection overlays."""
        self.ax.clear()
        
        # Create display montage
        display_img = self.montage.copy()
        
        # Overlay selection indicators
        for cluster_id, info in self.cluster_info.items():
            cluster_selected = cluster_id in self.selected_clusters
            
            for pos, grain_idx in zip(info['positions'], info['grain_indices']):
                if grain_idx in self.removed_grain_indices:
                    # Mark removed grains with thick red X
                    row, col = pos
                    y_pos = self.padding + row * (self.grain_size + self.padding)
                    x_pos = self.padding + col * (self.grain_size + self.padding)
                    
                    # Draw thick red X
                    line_thickness = 4  # Thickness of the X lines
                    for i in range(self.grain_size):
                        for t in range(-line_thickness//2, line_thickness//2 + 1):
                            # Main diagonal (with thickness)
                            y_main = y_pos + i
                            x_main = x_pos + i + t
                            if 0 <= y_main < display_img.shape[0] and 0 <= x_main < display_img.shape[1]:
                                display_img[y_main, x_main] = [255, 0, 0]
                            
                            # Anti-diagonal (with thickness)
                            y_anti = y_pos + i
                            x_anti = x_pos + self.grain_size - i - 1 + t
                            if 0 <= y_anti < display_img.shape[0] and 0 <= x_anti < display_img.shape[1]:
                                display_img[y_anti, x_anti] = [255, 0, 0]
                
                elif grain_idx in self.selected_grain_indices or cluster_selected:
                    # Overlay semi-transparent red for selections
                    row, col = pos
                    y_pos = self.padding + row * (self.grain_size + self.padding)
                    x_pos = self.padding + col * (self.grain_size + self.padding)
                    
                    overlay = display_img[y_pos:y_pos + self.grain_size, 
                                        x_pos:x_pos + self.grain_size].copy()
                    overlay = (overlay * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
                    display_img[y_pos:y_pos + self.grain_size, 
                              x_pos:x_pos + self.grain_size] = overlay
        
        self.ax.imshow(display_img.astype('uint8'))
        self.ax.axis('off')
        
        # Add status text
        n_selected = len(self.selected_grain_indices) + sum(
            len(self.cluster_info[cid]['grain_indices']) 
            for cid in self.selected_clusters
        )
        n_removed = len(self.removed_grain_indices)
        n_total = len(self.all_grains)
        
        title = (f"Selected: {n_selected} grains | "
                f"Removed: {n_removed} grains | "
                f"Remaining: {n_total - n_removed}/{n_total}\n"
                f"Left-click: select cluster | Shift+Left-click: select grain | "
                f"'d': delete | 'r': reset")
        self.ax.set_title(title, fontsize=10, pad=10)
        
        self.fig.canvas.draw()
    
    def onclick(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax or event.button != 1:
            return
        
        # Convert click position to grid coordinates
        x_click = event.xdata
        y_click = event.ydata
        
        # Find which grain was clicked
        col = int((x_click - self.padding) / (self.grain_size + self.padding))
        row = int((y_click - self.padding) / (self.grain_size + self.padding))
        
        # Check if click is within a grain (not in padding)
        x_in_cell = (x_click - self.padding) % (self.grain_size + self.padding)
        y_in_cell = (y_click - self.padding) % (self.grain_size + self.padding)
        
        if x_in_cell > self.grain_size or y_in_cell > self.grain_size:
            return  # Clicked in padding
        
        if (row, col) not in self.position_to_grain:
            return  # Clicked outside valid grains
        
        grain_idx = self.position_to_grain[(row, col)]
        
        if grain_idx in self.removed_grain_indices:
            return  # Already removed, ignore
        
        cluster_id = self.grain_to_cluster[grain_idx]
        
        # Shift+Click: select individual grain
        if event.key == 'shift':
            if grain_idx in self.selected_grain_indices:
                self.selected_grain_indices.remove(grain_idx)
            else:
                self.selected_grain_indices.add(grain_idx)
        # Regular click: select entire cluster
        else:
            if cluster_id in self.selected_clusters:
                self.selected_clusters.remove(cluster_id)
            else:
                self.selected_clusters.add(cluster_id)
        
        self.display_montage()
    
    def onkey(self, event):
        """Handle keyboard events."""
        if event.key == 'd':
            # Delete selected grains/clusters
            grains_to_remove = set(self.selected_grain_indices)
            
            for cluster_id in self.selected_clusters:
                grains_to_remove.update(self.cluster_info[cluster_id]['grain_indices'])
            
            self.removed_grain_indices.update(grains_to_remove)
            
            # Clear selections
            self.selected_grain_indices.clear()
            self.selected_clusters.clear()
            
            self.display_montage()
            
        elif event.key == 'r' or event.key == 'escape':
            # Reset/clear all selections
            self.selected_grain_indices.clear()
            self.selected_clusters.clear()
            self.display_montage()
    
    def activate(self):
        """Activate interactive selection mode."""
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        plt.show()
    
    def deactivate(self):
        """Deactivate interactive mode."""
        if self.cid_click is not None:
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
        if self.cid_key is not None:
            self.fig.canvas.mpl_disconnect(self.cid_key)
            self.cid_key = None
    
    def get_filtered_grains(self):
        """
        Get the list of grains after removing selected ones.
        
        Returns
        -------
        filtered_grains : list
            List of grain polygons with removed grains excluded
        """
        return [grain for i, grain in enumerate(self.all_grains) 
                if i not in self.removed_grain_indices]
    
    def get_filtered_labels(self):
        """
        Get cluster labels after removing selected grains.
        
        Returns
        -------
        filtered_labels : numpy.ndarray
            Cluster labels for remaining grains
        """
        return np.array([label for i, label in enumerate(self.cluster_labels)
                        if i not in self.removed_grain_indices])
    
    def get_filtered_images(self):
        """
        Get grain images after removing selected grains.
        
        Returns
        -------
        filtered_images : list
            List of grain images with removed grains excluded
        """
        return [img for i, img in enumerate(self.grain_images)
                if i not in self.removed_grain_indices]
    
    def get_filtered_indices(self):
        """
        Get the indices of grains that remain after filtering (not removed).
        
        This is useful for filtering other arrays (like all_grains) that are
        indexed the same way as the original grain_images/cluster_labels.
        
        Returns
        -------
        filtered_indices : numpy.ndarray
            Array of indices for grains that were not removed
            
        Examples
        --------
        >>> selector = ClusterMontageSelector(labels, grain_images, all_grains)
        >>> selector.activate()
        >>> # ... interact with the plot to remove some grains ...
        >>> filtered_idx = selector.get_filtered_indices()
        >>> filtered_all_grains = [all_grains[i] for i in filtered_idx]
        """
        return np.array([i for i in range(len(self.all_grains))
                        if i not in self.removed_grain_indices])


class ClusterMontageLabeler:
    """
    Interactive labeler for clustered grain montage (based on GrainPlot design).
    
    Allows fast interactive labeling of grains or entire clusters with custom labels.
    
    Parameters
    ----------
    cluster_labels : numpy.ndarray
        Array of cluster labels for each grain
    grain_images : list or numpy.ndarray
        Array of all grain images
    all_grains : list
        List of grain polygons (shapely.Polygon objects)
    label_names : list of str
        List of label names (e.g., ['quartz', 'feldspar', 'other'])
    label_colors : list of str or tuple, optional
        Colors for each label. If None, uses default colors.
    grid_cols : int
        Number of columns in the grid (default: 20)
    figsize : tuple
        Figure size (default: (20, 15))
    grain_size : int
        Size of each grain image in pixels (default: 224)
    padding : int
        Padding between grains (default: 2)
    boundary_thickness : int
        Border thickness for labeled grains (default: 2)
    sort_by_size : bool
        Sort grains by size within clusters (default: False)
    blit : bool
        Use blitting for fast updates (default: True)
    
    Interactive Controls
    --------------------
    - Left-click: Label individual grain
    - Shift + Left-click: Label entire cluster
    - Number keys (1-9): Select label by index
    - 'r': Reset all labels
    """
    
    def __init__(self, cluster_labels, grain_images, all_grains, label_names,
                 label_colors=None, grid_cols=20, figsize=(20, 15), 
                 grain_size=224, padding=2, boundary_thickness=2, 
                 sort_by_size=False, blit=True):
        
        # Store parameters
        self.cluster_labels = np.array(cluster_labels)
        self.grain_images = np.array(grain_images)
        self.all_grains = list(all_grains)
        self.label_names = list(label_names)
        self.grid_cols = grid_cols
        self.grain_size = grain_size
        self.padding = padding
        self.boundary_thickness = boundary_thickness
        self.blit = blit
        
        # Default colors
        if label_colors is None:
            default_colors = ['green', 'red', 'blue', 'yellow', 'magenta', 
                            'cyan', 'orange', 'purple', 'lime']
            self.label_colors = {name: default_colors[i % len(default_colors)] 
                                for i, name in enumerate(self.label_names)}
        else:
            self.label_colors = {name: color 
                                for name, color in zip(self.label_names, label_colors)}
        
        # Label tracking
        self.grain_labels = {}  # grain_idx -> label_name
        self.active_label = self.label_names[0] if self.label_names else None
        
        # Create montage
        self.montage, self.cluster_info = create_clustered_grain_montage(
            self.cluster_labels, self.grain_images, 
            grid_cols=grid_cols, grain_size=grain_size,
            padding=padding, draw_boundaries=False,
            boundary_thickness=boundary_thickness,
            all_grains=self.all_grains, sort_by_size=sort_by_size
        )
        
        # Create lookups
        self.position_to_grain = {}
        for cluster_id, info in self.cluster_info.items():
            for pos, grain_idx in zip(info['positions'], info['grain_indices']):
                self.position_to_grain[pos] = grain_idx
        
        self.grain_to_cluster = {}
        for cluster_id, info in self.cluster_info.items():
            for grain_idx in info['grain_indices']:
                self.grain_to_cluster[grain_idx] = cluster_id
        
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.canvas = self.fig.canvas
        
        # Draw montage (not animated - part of static background)
        self.montage_artist = self.ax.imshow(self.montage.astype('uint8'), animated=False)
        self.ax.axis('off')
        
        # Label overlays (animated patches, like grains in GrainPlot)
        self.label_patches = {}  # grain_idx -> list of Rectangle patches
        
        # Flag to prevent recursive background captures during blitting
        self._updating = False
        
        # Capture background (just the montage)
        if self.blit:
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        
        # Event handlers
        self.cid_click = None
        self.cid_key = None
        self.cid_draw = None
        
        # Print instructions to console
        self._print_instructions()
    
    def _print_instructions(self):
        """Print instructions to console."""
        print("\n" + "="*60)
        print("ClusterMontageLabeler - Interactive Controls")
        print("="*60)
        print(f"Active label: [{self.active_label}]")
        print(f"\nAvailable labels:")
        for i, name in enumerate(self.label_names):
            color = self.label_colors[name]
            print(f"  {i+1}: {name} ({color})")
        print(f"\nControls:")
        print(f"  - Left-click: Label individual grain")
        print(f"  - Shift + Left-click: Label entire cluster")
        print(f"  - Number keys (1-{len(self.label_names)}): Select label")
        print(f"  - 'r': Reset all labels")
        print("="*60 + "\n")
    
    def _draw_grain_border(self, grain_idx, label):
        """Draw border patches for a labeled grain (animated, like grain patches in GrainPlot)."""
        
        # Find position
        cluster_id = self.grain_to_cluster[grain_idx]
        pos_idx = np.where(self.cluster_info[cluster_id]['grain_indices'] == grain_idx)[0]
        if len(pos_idx) == 0:
            return []
        
        pos = self.cluster_info[cluster_id]['positions'][pos_idx[0]]
        row, col = pos
        
        y_pos = self.padding + row * (self.grain_size + self.padding)
        x_pos = self.padding + col * (self.grain_size + self.padding)
        
        color = self.label_colors[label]
        thickness = self.boundary_thickness
        
        # Create 4 rectangles for border (all animated for blitting)
        patches = [
            Rectangle((x_pos, y_pos), self.grain_size, thickness,
                     facecolor=color, edgecolor='none', animated=self.blit, zorder=10),
            Rectangle((x_pos, y_pos + self.grain_size - thickness), 
                     self.grain_size, thickness,
                     facecolor=color, edgecolor='none', animated=self.blit, zorder=10),
            Rectangle((x_pos, y_pos), thickness, self.grain_size,
                     facecolor=color, edgecolor='none', animated=self.blit, zorder=10),
            Rectangle((x_pos + self.grain_size - thickness, y_pos), 
                     thickness, self.grain_size,
                     facecolor=color, edgecolor='none', animated=self.blit, zorder=10),
        ]
        
        for patch in patches:
            self.ax.add_patch(patch)
        
        return patches
    
    def ondraw(self, event):
        """Handle draw events (recapture background after zoom/pan)."""
        if not self.blit or self._updating:
            return
        
        # Recapture background after any redraw (zoom, pan, etc.)
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
    
    def update(self):
        """Update display (using blitting like GrainPlot)."""
        if not self.blit:
            self.canvas.draw_idle()
            return
        
        # Set flag to prevent ondraw from recapturing during our blit
        self._updating = True
        
        # Restore background (includes montage)
        self.canvas.restore_region(self.background)
        
        # Get current view limits to only draw visible patches
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Draw only visible animated label patches
        for patches in self.label_patches.values():
            for patch in patches:
                # Quick visibility check - only draw if patch might be visible
                x, y = patch.get_xy()
                width = patch.get_width()
                height = patch.get_height()
                
                # Check if patch is within view bounds (with some margin)
                if (x + width >= xlim[0] and x <= xlim[1] and
                    y + height >= ylim[1] and y <= ylim[0]):  # Note: y-axis is inverted
                    self.ax.draw_artist(patch)
        
        # Blit
        self.canvas.blit(self.ax.bbox)
        self.canvas.flush_events()
        
        # Reset flag
        self._updating = False
    
    def onclick(self, event):
        """Handle mouse clicks."""
        if event.inaxes != self.ax or event.button != 1:
            return
        
        if self.active_label is None:
            return
        
        # Convert click to grid position
        x_click = event.xdata
        y_click = event.ydata
        
        if x_click is None or y_click is None:
            return
        
        col = int((x_click - self.padding) / (self.grain_size + self.padding))
        row = int((y_click - self.padding) / (self.grain_size + self.padding))
        
        # Check if click is within grain (not padding)
        x_in_cell = (x_click - self.padding) % (self.grain_size + self.padding)
        y_in_cell = (y_click - self.padding) % (self.grain_size + self.padding)
        
        if x_in_cell > self.grain_size or y_in_cell > self.grain_size:
            return
        
        if (row, col) not in self.position_to_grain:
            return
        
        grain_idx = self.position_to_grain[(row, col)]
        cluster_id = self.grain_to_cluster[grain_idx]
        
        # Determine grains to label
        if event.key == 'shift':
            grains_to_label = list(self.cluster_info[cluster_id]['grain_indices'])
        else:
            grains_to_label = [grain_idx]
        
        # Update labels
        for idx in grains_to_label:
            # Remove old patches if exists
            if idx in self.label_patches:
                for patch in self.label_patches[idx]:
                    patch.remove()
                del self.label_patches[idx]
            
            # Add new label
            self.grain_labels[idx] = self.active_label
            
            # Draw new border
            patches = self._draw_grain_border(idx, self.active_label)
            if patches:
                self.label_patches[idx] = patches
        
        # Fast blit update
        self.update()
        
        # Print status to console (after update to avoid slowing down the click)
        n_labeled = len(self.grain_labels)
        print(f"Labeled {n_labeled}/{len(self.all_grains)} grains | Active: [{self.active_label}]")
    
    def onkey(self, event):
        """Handle keyboard events."""
        if event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            idx = int(event.key) - 1
            if idx < len(self.label_names):
                self.active_label = self.label_names[idx]
                print(f"Active label changed to: [{self.active_label}]")
                # No update needed - just changes active label for next click
        
        elif event.key == 'r':
            # Reset all labels
            self.grain_labels.clear()
            
            for patches in self.label_patches.values():
                for patch in patches:
                    patch.remove()
            self.label_patches.clear()
            
            print("All labels reset!")
            self.update()
    
    def activate(self):
        """Activate interactive mode."""
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.cid_draw = self.fig.canvas.mpl_connect('draw_event', self.ondraw)
        plt.show()
    
    def deactivate(self):
        """Deactivate interactive mode."""
        if self.cid_click is not None:
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
        if self.cid_key is not None:
            self.fig.canvas.mpl_disconnect(self.cid_key)
            self.cid_key = None
        if self.cid_draw is not None:
            self.fig.canvas.mpl_disconnect(self.cid_draw)
            self.cid_draw = None
    
    def get_labeled_data(self):
        """Get labeled grain images and labels."""
        if len(self.grain_labels) == 0:
            return np.array([]), np.array([]), np.array([])
        
        indices = sorted(self.grain_labels.keys())
        labeled_images = np.array([self.grain_images[i] for i in indices])
        labels = np.array([self.grain_labels[i] for i in indices])
        
        return labeled_images, labels, np.array(indices)
    
    def get_unlabeled_indices(self):
        """Get indices of unlabeled grains."""
        return np.array([i for i in range(len(self.all_grains))
                        if i not in self.grain_labels])
    
    def get_filtered_indices(self):
        """Get indices of labeled grains."""
        return np.array(sorted(self.grain_labels.keys()))
    
    def export_labels(self, filepath):
        """Export labels to CSV."""
        
        if len(self.grain_labels) == 0:
            print("No labels to export!")
            return
        
        data = {
            'grain_index': list(self.grain_labels.keys()),
            'label': list(self.grain_labels.values()),
            'cluster_id': [self.grain_to_cluster[idx] for idx in self.grain_labels.keys()]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Exported {len(df)} labels to {filepath}")
    
    def save_labeled_images(self, output_dir):
        """
        Save labeled grain images to disk, organized by label.
        
        Creates subdirectories for each label and saves images as:
        {output_dir}/{label_name}/grain_{grain_index}.png
        
        Parameters
        ----------
        output_dir : str
            Base directory where labeled images will be saved
        
        Examples
        --------
        >>> labeler.save_labeled_images('labeled_grains')
        # Creates:
        # labeled_grains/quartz/grain_0.png
        # labeled_grains/quartz/grain_5.png
        # labeled_grains/feldspar/grain_1.png
        # etc.
        """
        if len(self.grain_labels) == 0:
            print("No labeled grains to save!")
            return
        
        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectory for each label
        for label_name in self.label_names:
            label_dir = os.path.join(output_dir, label_name)
            os.makedirs(label_dir, exist_ok=True)
        
        # Save each labeled image
        saved_count = {label: 0 for label in self.label_names}
        
        for grain_idx, label in tqdm(self.grain_labels.items(), desc='Saving labeled images'):
            # Get the grain image
            grain_img = self.grain_images[grain_idx]
            
            # Create output path
            label_dir = os.path.join(output_dir, label)
            output_path = os.path.join(label_dir, f'grain_{grain_idx}.png')
            
            # Save image
            img_pil = Image.fromarray(grain_img.astype('uint8'))
            img_pil.save(output_path)
            
            saved_count[label] += 1
        
        # Print summary
        print(f"\nSaved {len(self.grain_labels)} labeled images to '{output_dir}':")
        for label, count in saved_count.items():
            if count > 0:
                print(f"  {label}: {count} images")


def plot_classified_grains(image, all_grains, classifications, class_colors=None, 
                           ax=None, plot_image=True, im_alpha=1.0, 
                           edge_color='black', edge_width=0.5, fill_alpha=0.5,
                           legend=True):
    """
    Plot image with grain masks colored by their classification.
    
    Parameters
    ----------
    image : numpy.ndarray
        The input image to be plotted
    all_grains : list
        List of shapely Polygon objects representing grain masks
    classifications : list or numpy.ndarray
        Classification label for each grain (same length as all_grains)
    class_colors : dict, optional
        Dictionary mapping class names to colors. If None, uses default colors.
        Colors can be matplotlib color names or RGB tuples.
        Example: {'quartz': 'blue', 'feldspar': 'red', 'other': 'green'}
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure
    plot_image : bool
        Whether to plot the background image (default: True)
    im_alpha : float
        Alpha value for background image (default: 1.0)
    edge_color : str or tuple
        Color for grain edges (default: 'black')
    edge_width : float
        Width of grain edges (default: 0.5)
    fill_alpha : float
        Alpha value for grain fill colors (default: 0.5)
    legend : bool
        Whether to add a legend (default: True)
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects
    
    Examples
    --------
    >>> # After training classifier
    >>> predictions = clf.predict(features)
    >>> fig, ax = plot_classified_grains(
    ...     image, all_grains, predictions,
    ...     class_colors={'quartz': 'blue', 'feldspar': 'red', 'other': 'green'}
    ... )
    >>> plt.show()
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
    else:
        fig = ax.figure
    
    # Default colors if not provided
    if class_colors is None:
        unique_classes = np.unique(classifications)
        default_colors = ['blue', 'red', 'green', 'orange', 'purple', 
                         'cyan', 'magenta', 'yellow', 'lime', 'pink']
        class_colors = {cls: default_colors[i % len(default_colors)] 
                       for i, cls in enumerate(unique_classes)}
    
    # Plot background image
    if plot_image:
        ax.imshow(image, alpha=im_alpha)
    
    # Plot each grain with its classification color
    from matplotlib.patches import Patch
    
    for i, grain in enumerate(tqdm(all_grains, desc='Plotting classified grains')):
        classification = classifications[i]
        color = class_colors.get(classification, 'gray')
        
        ax.fill(
            grain.exterior.xy[0],
            grain.exterior.xy[1],
            facecolor=color,
            edgecolor=edge_color,
            linewidth=edge_width,
            alpha=fill_alpha,
        )
    
    # Add legend
    if legend:
        # Count grains per class
        unique_classes, counts = np.unique(classifications, return_counts=True)
        
        # Create legend patches
        legend_elements = [
            Patch(facecolor=class_colors.get(cls, 'gray'), 
                  edgecolor=edge_color, 
                  label=f'{cls} (n={count})')
            for cls, count in zip(unique_classes, counts)
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.15, 1.0), framealpha=0.9)
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis('off')
    
    return fig, ax