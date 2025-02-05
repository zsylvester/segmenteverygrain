import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mpl_Polygon
from tqdm import tqdm, trange
import networkx as nx
import rasterio
from rasterio.features import rasterize
import warnings

from skimage import measure
from skimage.measure import regionprops, regionprops_table, label, find_contours
from skimage.morphology import binary_erosion, binary_dilation
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from shapely.geometry import Polygon, Point, MultiPolygon, mapping, shape
from shapely.affinity import translate
import scipy.ndimage as ndi
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import rtree
import itertools
from glob import glob 
import os
from PIL import Image
import json

import tensorflow as tf
from keras import Model
from keras.layers import Input, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.utils import load_img
from keras.saving import load_model
from keras.optimizers import Adam

from segment_anything import SamPredictor

def predict_image_tile(im_tile, model):
    """
    Predicts one image tile using a Unet model.

    Parameters
    ----------
    im_tile : 3D array
        The image tile for which the prediction will be done.
    model
        Tensorflow model used for semantic segmentation.

    Returns
    -------
    im_tile_pred : 3D array
        Predicted tile.
    """
    
    # Check for invalid input
    if not isinstance(im_tile, np.ndarray):
        raise ValueError("Input image tile must be a numpy array.")
    if len(im_tile.shape) != 3:
        raise ValueError("Input image tile must be a 3D array.")
    if len(im_tile.shape) == 3 and im_tile.shape[2] != 3:
        raise ValueError("3D input image tile must have 3 channels.")
    im_tile = np.expand_dims(im_tile, axis=0) # add batch dimension
    im_tile_pred = model.predict(im_tile, verbose=0) # make prediction
    im_tile_pred = im_tile_pred[0] # remove batch dimension
    return im_tile_pred

def predict_image(image, model, I):
    """
    Segmantic segmentation of the entire image using a Unet model.

    Parameters
    ----------
    image : 2D or 3D array
        The image that is being segmented. Can have one or more channels.
    model
        Tensorflow model used for semantic segmentation.
    I : int
        Size of the square-shaped image tiles in pixels.

    Returns
    -------
    image_pred : 3D array
        Semantic segmentation result for the input image.
    """

    # Check for invalid input
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array.")
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be a 2D or 3D array.")
    if image.ndim == 3 and image.shape[2] not in [1, 3]:
        raise ValueError("3D input image must have 1 or 3 channels.")

    pad_rows = I - np.mod(image.shape[0], I)
    pad_cols = I - np.mod(image.shape[1], I)
    if image.ndim == 2:
        image = np.stack((image, image, image), axis=-1) # convert to 3 channels
    if image.ndim == 3:
        image = np.vstack((image, np.zeros((pad_rows, image.shape[1], image.shape[2]))))
        image = np.hstack((image, np.zeros((image.shape[0], pad_cols, image.shape[2]))))
    r = int(np.floor(image.shape[0]/I)) # number of rows of image tiles
    c = int(np.floor(image.shape[1]/I)) # number of columns of image tiles
    
    I2 = int(I/2)
    W = np.hanning(I) * np.hanning(I)[:, np.newaxis]
    Wup = W.copy()
    Wup[:I2, :] = np.tile(np.hanning(I), (I2, 1))
    Wdown = W.copy()
    Wdown[I2:, :] = np.tile(np.hanning(I), (I2, 1))

    image = np.hstack((np.zeros((r*I, I2, 3)), image, np.zeros((r*I, I2, 3)))) # padding on the left and right sides
    image_pred = np.zeros((image.shape[0], image.shape[1], 3))
    print('segmenting image tiles...')
    for i in trange(c+1): # rows, no offset
        for j in range(1,2*r-2): # columns
            im_tile = image[j*I2:(j+2)*I2, i*I:(i+1)*I, :]/255.0
            im_tile_pred = predict_image_tile(im_tile, model)
            for layer in range(3):
                image_pred[j*I2:(j+2)*I2, i*I:(i+1)*I, layer] += im_tile_pred[:, :, layer] * W
    for i in range(c+1): # first row
        im_tile = image[:2*I2, i*I:(i+1)*I, :]/255.0
        im_tile_pred = predict_image_tile(im_tile, model)
        for layer in range(3):
            image_pred[:2*I2, i*I:(i+1)*I, layer] += im_tile_pred[:, :, layer] * Wup
    for i in range(c+1): # last row
        im_tile = image[(2*r-2)*I2:2*r*I2, i*I:(i+1)*I, :]/255.0
        im_tile_pred = predict_image_tile(im_tile,model)
        for layer in range(3):
            image_pred[(2*r-2)*I2:2*r*I2, i*I:(i+1)*I, layer] += im_tile_pred[:, :, layer] * Wdown
    for i in trange(c): # rows, half offset
        for j in range(1,2*r-2): # columns
            im_tile = image[j*I2:(j+2)*I2, i*I+I2:(i+1)*I+I2, :]/255.0
            im_tile_pred = predict_image_tile(im_tile,model)
            for layer in range(3):
                image_pred[j*I2:(j+2)*I2, i*I+I2:(i+1)*I+I2, layer] += im_tile_pred[:, :, layer] * W
    for i in range(c): # first row
        im_tile = image[:2*I2, i*I+I2:(i+1)*I+I2, :]/255.0
        im_tile_pred = predict_image_tile(im_tile,model)
        for layer in range(3):
            image_pred[:2*I2, i*I+I2:(i+1)*I+I2, layer] += im_tile_pred[:, :, layer] * Wup
    for i in range(c): # last row
        im_tile = image[(2*r-2)*I2:2*r*I2, i*I+I2:(i+1)*I+I2, :]/255.0
        im_tile_pred = predict_image_tile(im_tile,model)
        for layer in range(3):
            image_pred[(2*r-2)*I2:2*r*I2, i*I+I2:(i+1)*I+I2, layer] += im_tile_pred[:, :, layer] * Wdown

    image_pred = image_pred[:, I2:-I2, :] # crop the left and right side padding
    image_pred = image_pred[:-pad_rows, :-pad_cols, :] # get rid of padding
    return image_pred

def label_grains(image, image_pred, dbs_max_dist=20.0):
    """
    Label grains in semantic segmentation result and generate prompts for SAM model.

    Parameters
    ----------
    image : 2D or 3d array
        image that was segmented
    image_pred : 3D array
        semantic segmentation result
    dbs_max_dist : float
        DBSCAN distance parameter; decreasing it results in more SAM prompts and longer processing times

    Returns
    -------
    labels_simple
        the labels as an image
    all_coords
        pixel coordinates of the prompts
    """

    grains = image_pred[:,:,1].copy() # grain prediction from semantic segmentation result
    grains[grains >= 0.5] = 1
    grains[grains < 0.5] = 0
    grains = grains.astype('bool')
    labels_simple, n_elems = measure.label(grains, return_num = True, connectivity=1)
    props = regionprops_table(labels_simple, intensity_image = image, properties=('label', 'area', 'centroid'))
    grain_data_simple = pd.DataFrame(props)
    coords_simple = np.vstack((grain_data_simple['centroid-1'], grain_data_simple['centroid-0'])).T # use the centroids of the Unet grains as 'simple' prompts
    coords_simple = coords_simple.astype('int32')
    background_probs = image_pred[:,:,0][coords_simple[:,1], coords_simple[:,0]]
    inds = np.where(background_probs < 0.3)[0] # get rid of prompts that are likely to be background
    coords_simple = coords_simple[inds, :]

    bounds = image_pred[:,:,2].copy() # grain boundary prediction
    bounds[bounds >= 0.5] = 1
    bounds[bounds < 0.5] = 0
    bounds = bounds.astype('bool')
    temp_labels, n_elems = measure.label(bounds, return_num = True, connectivity=1)
    # Find the object with the largest area:
    label_counts = np.bincount(temp_labels.ravel())
    labels = np.where(label_counts > 100)[0][1:]
    if len(label_counts[1:]) > 0:
        largest_label = np.argmax(label_counts[1:]) + 1
        for label in labels:
            temp_labels[temp_labels == label] = largest_label
        bounds[temp_labels != largest_label] = 0
        bounds = bounds-1
        bounds[bounds < 0] = 1
        bounds = bounds.astype('bool')
        distance = ndi.distance_transform_edt(bounds)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=bounds.astype('bool'))
        background_probs = image_pred[:,:,0][coords[:,0], coords[:,1]]
        inds = np.where(background_probs < 0.3)[0]
        coords = coords[inds, :]
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=bounds)
        props = regionprops_table(labels, intensity_image = image, properties=('label', 'area', 'centroid', 'major_axis_length', 'minor_axis_length', 
                                                                                        'orientation', 'perimeter', 'max_intensity', 'mean_intensity', 'min_intensity'))
        grain_data = pd.DataFrame(props)
        if len(grain_data) > 0:
            coords = np.vstack((grain_data['centroid-1'].values, grain_data['centroid-0'].values)).T
            # Create a DBSCAN clustering object
            dbscan = DBSCAN(eps=dbs_max_dist, min_samples=2)
            # Fit the data to the DBSCAN object
            dbscan.fit(np.vstack((grain_data['centroid-1'], grain_data['centroid-0'])).T)
            # Get the cluster labels for each point (-1 represents noise/outliers)
            db_labels = dbscan.labels_
            coords_ws = coords[np.where(db_labels == -1)[0]]
            for i in np.unique(db_labels):
                xy = np.mean(coords[np.where(db_labels == i)[0]], axis=0)
                coords_ws = np.vstack((coords_ws, xy))
            coords_ws = coords_ws.astype('int32')
            background_probs = image_pred[:,:,0][coords_ws[:,1], coords_ws[:,0]]
            inds = np.where(background_probs < 0.3)[0] # get rid of prompts that are likely to be background
            coords_ws = coords_ws[inds, :]
            all_coords = np.vstack((coords_ws, coords_simple))
        else:
            all_coords = coords_simple
    else:
        all_coords = coords_simple
    return labels_simple, all_coords

def one_point_prompt(x, y, image, predictor, ax=False):
    """
    Perform SAM segmentation using a single point prompt.

    Parameters
    ----------
        x : float
            the x-coordinate of the point
        y : float
            the y-coordinate of the point
        image : numpy.ndarray)
            the input image
        predictor 
            the SAM predictor
        ax : bool, default True
            whether to plot the segmentation result on an axis

    Returns
    -------
        sx
            the x-coordinates of the contour points
        sy
            the y-coordinates of the contour points
        mask
            the segmented mask
    """
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    new_masks = []
    new_scores = []
    if len(masks) >= 1:
        for ind in range(len(masks)):
            if np.sum(masks[ind])/(image.shape[0]*image.shape[1]) <= 0.5: # if mask is very large compared to size of the image
                new_scores.append(scores[ind])
                if masks.ndim > 2:
                    new_masks.append(masks[ind, :, :])
                else:
                    new_masks.append(masks)
    if len(new_masks) > 0:
        masks = new_masks
        scores = new_scores
        ind = np.argmax(scores)
        temp_labels, n_elems = measure.label(masks[ind], return_num = True, connectivity=1)
        if n_elems > 1: # if the mask has more than one element, find the largest one and delete the rest
            mask = masks[ind]
            # Find the object with the largest area
            label_counts = np.bincount(temp_labels.ravel())
            largest_label = np.argmax(label_counts[1:]) + 1
            mask[temp_labels != largest_label] = 0
        else:
            mask = masks[ind]
        contours = measure.find_contours(mask, 0.5)
        if len(contours) > 0:
            sx = contours[0][:,1]
            sy = contours[0][:,0]
        else:
            sx = []; sy = []
        if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[0, -1]):
            mask = np.pad(mask, 1, mode='constant')
            contours = measure.find_contours(mask, 0.5)
            sx = contours[0][:,1]
            sy = contours[0][:,0]
            if np.any(mask[1, :]):
                sy = sy-1
            if np.any(mask[:,1]):
                sx = sx-1
            mask = mask[1:-1, 1:-1]
        if len(sx) > 0 and ax:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            ax.fill(sx, sy, facecolor=color, edgecolor='k', alpha=0.5)
    else: sx = []; sy = []; mask = np.zeros_like(image[:,:,0])
    return sx, sy, mask

def two_point_prompt(x1, y1, x2, y2, image, predictor, ax=False):
    """
    Perform a two-point-prompt-based segmentation using the SAM model. 
    Second point is used as background (label=0).

    Parameters
    ----------
        x1 : float
            x-coordinate of the first point
        y1 : float
            y-coordinate of the first point
        x2 : float 
            x-coordinate of the second point
        y2 : float 
            y-coordinate of the second point
        ax : matplotlib.axes.Axes
            The axes to plot the segmentation result
        image : numpy.ndarray
            the input image
        predictor 
            the SAM predictor

    Returns
    -------
        sx: numpy.ndarray
            The x-coordinates of the contour points
        sy : numpy.ndarray
            The y-coordinates of the contour points
    """
    input_point = np.array([[x1, y1], [x2, y2]])
    input_label = np.array([1, 0])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    ind = np.argmax(scores)
    if np.sum(masks[ind])/(image.shape[0]*image.shape[1]) > 0.1:
        scores = np.delete(scores, ind)
        ind = np.argmax(scores)
    contours = measure.find_contours(masks[ind], 0.5)
    sx = contours[0][:,1]
    sy = contours[0][:,0]
    if ax:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        ax.fill(sx, sy, facecolor=color, edgecolor='k', alpha=0.5)
    return sx, sy

def find_overlapping_polygons(polygons, min_overlap=0.4):
    """
    Finds and returns a list of overlapping polygons from the given list of polygons using spatial indexing.

    Parameters
    ----------
    polygons : list
        A list of polygons.

    Returns
    -------
    overlapping_polygons : list
        A list of tuples representing the indices of overlapping polygons.

    """
    overlapping_polygons = []
    # Create an R-tree index
    idx = rtree.index.Index()
    # Insert polygons into the index
    for i, poly in enumerate(polygons):
        bounds = poly.bounds
        idx.insert(i, bounds)
    # Find overlapping polygons using the index
    for i, poly1 in tqdm(enumerate(polygons)):
        bounds1 = poly1.bounds
        overlapping_indices = list(idx.intersection(bounds1))
        for j in overlapping_indices:
            poly2 = polygons[j]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if not poly1.is_valid:
                    poly1 = poly1.buffer(0)
                if not poly2.is_valid:
                    poly2 = poly2.buffer(0)
                if i != j and poly1.intersects(poly2)\
                    and poly1.intersection(poly2).area > min_overlap*(min(poly1.area, poly2.area))\
                    and (j, i) not in overlapping_polygons:
                    overlapping_polygons.append((i, j))
    return overlapping_polygons

def Unet():
    """
    Creates a U-Net model for image segmentation.

    Returns
    -------
    model : tensorflow.keras.Model
        The U-Net model.
    """

    tf.keras.backend.clear_session()

    inputs = Input((256, 256, 3), name='input')
    
    conv1 = Conv2D(16, (3,3), activation='relu', padding = 'same')(inputs)
    conv1 = Conv2D(16, (3,3), activation='relu', padding = 'same')(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3,3), activation='relu', padding = 'same')(pool1)
    conv2 = Conv2D(32, (3,3), activation='relu', padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (3,3), activation='relu', padding = 'same')(pool2)
    conv3 = Conv2D(64, (3,3), activation='relu', padding = 'same')(conv3)
    conv3 = BatchNormalization()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, (3,3), activation='relu', padding = 'same')(pool3)
    conv4 = Conv2D(128, (3,3), activation='relu', padding = 'same')(conv4)
    conv4 = BatchNormalization()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(256, (3,3), activation='relu', padding = 'same')(pool4)
    conv5 = Conv2D(256, (3,3), activation='relu', padding = 'same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(128, (3,3), activation='relu', padding = 'same')(up6)
    conv6 = Conv2D(128, (3,3), activation='relu', padding = 'same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(64, (3,3), activation='relu', padding = 'same')(up7)
    conv7 = Conv2D(64, (3,3), activation='relu', padding = 'same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = 'same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(32, (3,3), activation='relu', padding = 'same')(up8)
    conv8 = Conv2D(32, (3,3), activation='relu', padding = 'same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(16, (3, 3), strides = (2, 2), padding = 'same')(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = Conv2D(16, (3,3), activation='relu', padding = 'same')(up9)
    conv9 = Conv2D(16, (3,3), activation='relu', padding = 'same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(3, (1,1), activation='softmax')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def weighted_crossentropy(y_true, y_pred):
    """
    Calculates the weighted cross-entropy loss between the true labels and predicted labels.

    Parameters
    ----------
    y_true : tensor
        True labels.
    y_pred : tensor
        Predicted labels.

    Returns
    -------
    loss : tensor
        Weighted cross-entropy loss.
    """
    class_weights = tf.constant([[[[0.6, 1.0, 5.0]]]]) # increase the weight on the grains and the grain boundaries
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    weighted_losses = weights * unweighted_losses
    loss = tf.reduce_mean(weighted_losses)
    return loss

def plot_images_and_labels(img, label):
    """
    Plot the input image and its corresponding label side by side.
    The third subplot shows the input image with the label overlayed.

    Parameters
    ----------
    img : numpy.ndarray
        The input image to be plotted.
    label : numpy.ndarray
        The label image to be plotted.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131)
    ax1.imshow(img)
    ax1.set_title("Grains")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = fig.add_subplot(132)
    ax2.imshow(label[:, :, 0], cmap='Reds')
    ax2.set_title("Label")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3 = fig.add_subplot(133)
    ax3.imshow(img)
    ax3.imshow(label[:, :, 0], alpha=0.3, cmap='Reds')
    ax3.set_title("Blending")
    ax3.set_xticks([])
    ax3.set_yticks([])

def calculate_iou(poly1, poly2):
    """
    Calculate the Intersection over Union (IoU) between two polygons.

    Parameters
    ----------
    poly1 : Polygon
        The first polygon.
    poly2 : Polygon
        The second polygon.

    Returns
    -------
    iou : float
        The IoU value between the two polygons.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    iou = intersection_area / union_area
    return iou

def pick_most_similar_polygon(polygons):
    """
    Picks the 'most similar' polygon from a list of polygons based on the average IoU scores.

    Parameters
    ----------
    polygons : list
        A list of polygons.

    Returns
    -------
    most_similar_polygon : Polygon
        The most similar polygon.

    """
    # Calculate the average IoU for each polygon
    avg_iou_scores = []
    for i, poly1 in enumerate(polygons):
        iou_scores = []
        for j, poly2 in enumerate(polygons):
            if i != j:
                iou_scores.append(calculate_iou(poly1, poly2))
        avg_iou_scores.append(sum(iou_scores) / len(iou_scores))
    # Find the polygon with the highest average IoU score
    most_similar_index = avg_iou_scores.index(max(avg_iou_scores))
    most_similar_polygon = polygons[most_similar_index]
    return most_similar_polygon

def sam_segmentation(sam, image, image_pred, coords, labels, min_area, plot_image=False, remove_edge_grains=False, remove_large_objects=False):
    """
    Perform segmentation using the Segment Anything Model (SAM).

    Parameters
    ----------
    sam : SamPredictor
        The SAM model.
    image : numpy.ndarray
        The input image.
    image_pred : numpy.ndarray
        The output of the Unet segmentation.
    coords : numpy.ndarray
        The coordinates of the SAM prompts.
    labels : numpy.ndarray
        The labeled image that comes from the 'label_grains' function.
    min_area : int
        The minimum area of the grains, in pixels.
    plot_image : bool, optional
        Whether to plot the segmented image. Default is False.
    remove_edge_grains : bool, optional
        Whether to remove grains that are touching the edge of the image. Default is False.
    remove_large_objects : bool, optional
        Whether to remove large objects. Default is False. This is useful when the segmentation result is not very good.

    Returns
    -------
    all_grains : list
        List of polygons representing the segmented grains.
    labels : numpy.ndarray
        The labeled image.
    mask_all : numpy.ndarray
        The mask of all grains.
    grain_data : pandas.DataFrame
        DataFrame containing properties of each grain.
    fig : matplotlib.figure.Figure or None
        The figure object if plot_image is True, otherwise None.
    ax : matplotlib.axes.Axes or None
        The axes object if plot_image is True, otherwise None.
    """
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    all_grains = []
    print('creating masks using SAM...')
    for i in trange(len(coords[:,0])):
        x = coords[i,0]
        y = coords[i,1]
        sx, sy, mask = one_point_prompt(x, y, image, predictor)
        if np.max(mask) > 0:
            if remove_edge_grains and np.sum(np.hstack([mask[:4, :], mask[-4:, :], mask[:, :4].T, mask[:, -4:].T])) == 0: # if the mask is not touching too much the edge of the image
                all_grains = collect_polygon_from_mask(labels, mask, image_pred, all_grains, sx, sy)
            if not remove_edge_grains:
                all_grains = collect_polygon_from_mask(labels, mask, image_pred, all_grains, sx, sy)

    print('finding overlapping polygons...')
    new_grains, comps, g = find_connected_components(all_grains, min_area)

    if remove_large_objects:
        N_neighbors_before, N_neighbors_after, Nodes = [], [], []
        for i in range(len(comps)):
            g_small = nx.subgraph(g, comps[i]).copy()
            nodes = []
            n_neighbors_before = []
            for node in g_small.nodes:
                n_neighbors_before.append(len(list(nx.neighbors(g_small, node))))
                nodes.append(node)
            
            edges_to_remove = []
            for edge in g_small.edges:
                iou = calculate_iou(all_grains[edge[0]], all_grains[edge[1]])
                if iou < 0.8: # probably this shouldn't be hardcoded!
                    edges_to_remove.append(edge)
            for edge in edges_to_remove:
                g_small.remove_edge(edge[0], edge[1])
            
            n_neighbors_after = []
            for node in g_small.nodes:
                n_neighbors_after.append(len(list(nx.neighbors(g_small, node))))
            N_neighbors_before += n_neighbors_before
            N_neighbors_after += n_neighbors_after
            Nodes += nodes

        classifications = classify_points(N_neighbors_before, N_neighbors_after, 10, 0, 60, 25)
        if len(np.unique(classifications)) > 1:
            cluster_0_mean = np.nanmean(np.array(N_neighbors_after)[np.array(classifications)==0])
            cluster_1_mean = np.nanmean(np.array(N_neighbors_after)[np.array(classifications)==1])
            if cluster_0_mean > cluster_1_mean: # the cluster with the larger number of neighbors after pruning the components is the one that we want to keep
                all_grains = np.array(all_grains)[np.array(Nodes)[np.array(classifications)==0]]
            else:
                all_grains = np.array(all_grains)[np.array(Nodes)[np.array(classifications)==1]]

        all_grains = list(all_grains) + new_grains

        print('finding overlapping polygons...')
        new_grains, comps, g = find_connected_components(all_grains, min_area)

    print('finding best polygons...')
    all_grains = merge_overlapping_polygons(all_grains, new_grains, comps, min_area, image_pred)
    if len(all_grains) > 0:
        print('creating labeled image...')
        labels, mask_all = create_labeled_image(all_grains, image)
    else:
        labels = np.zeros_like(image[:,:,0])
        mask_all = np.zeros_like(image)
    if plot_image:
        fig, ax = plt.subplots(figsize=(15,10))
        ax.imshow(image)
        plot_image_w_colorful_grains(image, all_grains, ax, cmap='Paired')
        plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10)
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, np.shape(image)[1]])
        plt.ylim([np.shape(image)[0], 0])
        plt.tight_layout()
    else:
        fig, ax = None, None                                                                    
    props = regionprops_table(labels, intensity_image = image, properties=('label', 'area', 'centroid', 'major_axis_length', 'minor_axis_length', 
                                                                                    'orientation', 'perimeter', 'max_intensity', 'mean_intensity', 'min_intensity'))
    grain_data = pd.DataFrame(props)
    return all_grains, labels, mask_all, grain_data, fig, ax

def collect_polygon_from_mask(labels, mask, image_pred, all_grains, sx, sy, min_area=100, max_n_large_grains=10, max_bg_fraction=0.7):
    """
    Collect polygon from a mask and append it to a list of grains.

    Parameters
    ----------
    labels : ndarray
        Array of labels for each pixel in the image.
    mask : ndarray
        Boolean mask indicating the region of interest.
    image_pred : ndarray
        Predicted image from the Unet model.
    all_grains : list
        List to append the resulting polygons to.
    sx : ndarray
        X-coordinates of the polygon vertices.
    sy : ndarray
        Y-coordinates of the polygon vertices.
    min_area : int, optional
        Minimum area for a label to be considered significant (default is 100).
    max_n_large_grains : int, optional
        Maximum number of large grains allowed in the mask (default is 10).
    max_bg_fraction : float, optional
        Maximum fraction of the mask that can be background (default is 0.7).

    Returns
    -------
    list
        Updated list of polygons representing grains.
    """
    labels_in_mask = np.unique(labels[mask])
    large_labels_in_mask = [label for label in labels_in_mask if len(labels[mask][labels[mask]==label]) >= min_area]
    # skip masks that (1) cover too many objects of significant size in the Unet output and (2) masks that are mostly background:
    if len(large_labels_in_mask) < max_n_large_grains and np.mean(image_pred[:,:,0][mask]) < max_bg_fraction:
        poly = Polygon(np.vstack((sx, sy)).T)
        if not poly.is_valid:
            poly = poly.buffer(0)
        all_grains.append(poly)
    return all_grains

def find_connected_components(all_grains, min_area):
    """
    Finds connected components in a graph of overlapping polygons.

    Parameters
    ----------
    all_grains : list
        List of polygons representing all grains.
    min_area : float
        Minimum area threshold for valid grains.

    Returns
    -------
    new_grains : list
        List of polygons that do not overlap and have an area greater than min_area.
    comps : list
        List of sets, where each set represents a connected component of overlapping polygons.
    g : networkx.Graph
        The graph of overlapping polygons.
    """
    overlapping_polygons = find_overlapping_polygons(all_grains)
    g = nx.Graph(overlapping_polygons)
    comps = list(nx.connected_components(g))
    connected_grains = set()
    for comp in comps:
        connected_grains.update(comp)
    new_grains = []
    for i in range(len(all_grains)): # collect the objects that do not overlap each other
        if i not in connected_grains and all_grains[i].area > min_area:
            if not all_grains[i].is_valid:
                all_grains[i] = all_grains[i].buffer(0)
            new_grains.append(all_grains[i])
    return new_grains, comps, g

def merge_overlapping_polygons(all_grains, new_grains, comps, min_area, image_pred):
    """
    Merge overlapping polygons in a connected component.

    This function takes a list of all polygons, a list of polygons that do not overlap with other polygons,
    a list of connected components, a minimum area threshold, and the Unet prediction.
    It iterates over each connected component and merges the overlapping polygons within that component.
    The most similar polygon is selected as the representative polygon for the merged region.
    If the area of the representative polygon is greater than the minimum area threshold, it is added to the new polygons list.

    Parameters
    ----------
    all_grains : list
        List of all polygons.
    new_grains : list
        List of polygons that do not overlap each other.
    comps : list
        List of connected components.
    min_area : float
        Minimum area threshold.
    image_pred : numpy.ndarray
        The Unet prediction.

    Returns
    -------
    all_grains : list
        List of merged polygons.
    """
    for j in trange(len(comps)): # deal with the overlapping objects, one connected component at a time
        polygons = [] # polygons in the connected component
        for i in comps[j]:
            polygons.append(all_grains[i])
        most_similar_polygon = pick_most_similar_polygon(polygons)
        # this next section is needed so that grains that are not covered by the 'most similar polygon' are not lost
        diff_polys = []
        for polygon in polygons:
            if polygon != most_similar_polygon:
                diff_polygon = polygon.difference(most_similar_polygon)
                if diff_polygon.area > min_area:
                    diff_raster = rasterize_grains([diff_polygon], image_pred)
                    diff_grain = image_pred[diff_raster == 1][:,1]
                    if len(diff_grain) > 0:
                        # if a large fraction of the pixels in the difference polygon are grains:
                        if len(np.where(diff_grain > 0.5)[0])/len(diff_grain) > 0.1:
                            diff_polys.append(diff_polygon)
        # deal with the cases when the difference polygons are MultiPolygons:
        polys = []
        for poly in diff_polys:
            if type(poly) == MultiPolygon:
                areas = []
                for geom in poly.geoms:
                    areas.append(geom.area)
                poly = poly.geoms[np.argmax(areas)]
                polys.append(poly)
            elif type(poly) == Polygon:
                polys.append(poly)
        diff_polys = polys
        # find the polygons that are not overlapping with the most similar polygon among the 
        # difference polygons and not overlapping too much with each other:
        selected_polygons = []
        if len(diff_polys) > 1:
            most_similar_polygon1 = pick_most_similar_polygon(diff_polys)
            selected_polygons = [most_similar_polygon1]
            for poly1, poly2 in itertools.combinations(diff_polys, 2):
                iou = calculate_iou(poly1, poly2)
                if iou < 0.1:
                    iou1 = 0
                    iou2 = 0
                    iou1s = []
                    iou2s = []
                    for poly in selected_polygons:
                        iou1s.append(calculate_iou(poly1, poly))
                        iou2s.append(calculate_iou(poly2, poly))
                    if len(iou1s)>0:
                        iou1 = iou1s[np.argmax(iou1s)]
                    if len(iou2s)>0:
                        iou2 = iou2s[np.argmax(iou2s)]
                    if iou1 < 0.1 and poly1.area > min_area:
                        selected_polygons.append(poly1)
                    if iou2 < 0.1 and poly2.area > min_area:
                        selected_polygons.append(poly2)
        elif len(diff_polys) == 1:
            if diff_polys[0].area > min_area:
                selected_polygons = [diff_polys[0]]
        opened_polygons = [] # get rid of thin grain margins that are not real grains
        for poly in selected_polygons:
            erosion_distance = -5  # Negative value for erosion
            dilation_distance = 5  # Positive value for dilation
            eroded_polygon = poly.buffer(erosion_distance)
            opened_polygon = eroded_polygon.buffer(dilation_distance)
            if opened_polygon.area > min_area and type(opened_polygon) == Polygon:
                opened_polygons.append(opened_polygon)
        selected_polygons = opened_polygons
        if most_similar_polygon.area > min_area and most_similar_polygon not in new_grains:
            new_grains.append(most_similar_polygon)
        if len(selected_polygons) > 0:
            new_grains += selected_polygons
    all_grains = new_grains # replace original list of polygons
    return all_grains

def rasterize_grains(all_grains, image):
    """
    Rasterizes a list of polygons representing grains into an array of labels.

    Parameters
    ----------
    all_grains : list
        A list of polygons representing grains.
    image : numpy.ndarray
        The input image.

    Returns
    -------
    numpy.ndarray
        The rasterized array of labels.

    """
    labels = np.arange(1, len(all_grains)+1)
    # Combine polygons and labels into a tuple of (polygon, label) pairs
    shapes_with_labels = zip(all_grains, labels)
    # Define the shape and resolution of the rasterized output
    out_shape = image.shape[:2]  # Output array shape (height, width)
    bounds = (-0.5, image.shape[0]-0.5, image.shape[1]-0.5, -0.5)  # Left, bottom, right, top of the array (bounding box)
    # Define the transformation from pixel coordinates to spatial coordinates
    transform = rasterio.transform.from_bounds(*bounds, out_shape[1], out_shape[0])
    # Rasterize the polygons into an array of labels
    rasterized = rasterize(
        ((poly, label) for poly, label in shapes_with_labels),
        out_shape=out_shape,
        transform=transform,
        fill=0,  # Background value (for pixels outside polygons)
        dtype='int32'
    )
    return rasterized

def create_labeled_image(all_grains, image):
    """
    Create a labeled image based on the provided grains and input image.

    Parameters
    ----------
    all_grains : list
        List of shapely Polygon objects representing the grains.
    image : numpy.ndarray
        Input image.

    Returns
    -------
    rasterized : numpy.ndarray
        Labeled image where each grain is assigned a unique label.
    mask_all : numpy.ndarray
        Binary mask indicating the presence of grains and their boundaries.
    """
    rasterized = rasterize_grains(all_grains, image) # rasterize grains
    boundaries = []
    for grain in all_grains:
        boundaries.append(grain.boundary.buffer(2))
    boundaries_rasterized = rasterize_grains(boundaries, image) # rasterize grain boundaries
    mask_all = np.zeros(image.shape[:-1]) # combine grains and grain boundaries into a single mask
    mask_all[rasterized > 0] = 1
    mask_all[boundaries_rasterized >= 1] = 2
    rasterized = rasterized.astype('int')
    return rasterized, mask_all

def predict_large_image(fname, model, sam, min_area, patch_size=2000, overlap=300, remove_large_objects=False):
    """
    Predicts the location of grains in a large image using a patch-based approach.

    Parameters
    ----------
    fname : str
        The file path of the input image.
    model : tensorflow.keras.Model
        The Unet model used for the preliminary grain prediction.
    sam : SamPredictor
        The SAM model used for grain segmentation.
    min_area : int
        The minimum area threshold for valid grains.
    patch_size : int, optional
        The size of each patch. Defaults to 2000.
    overlap : int, optional
        The overlap between patches. Defaults to 300.
    remove_large_objects : bool, optional
        Whether to remove large objects from the segmentation. Defaults to False.

    Returns
    -------
    All_Grains : list
        A list of grains represented as polygons.
    image_pred : numpy.ndarray
        The Unet predictions for the entire image.
    all_coords : numpy.ndarray
        The coordinates of the SAM prompts.
    """
    step_size = patch_size - overlap  # step size for overlapping patches
    image = np.array(load_img(fname))
    img_height, img_width = image.shape[:2]  # get the height and width of the image 
    # loop over the image and extract patches:
    All_Grains = []
    total_patches = ((img_height - patch_size + step_size) // step_size + 1) * ((img_width - patch_size + step_size) // step_size + 1)
    # Initialize an array to store the Unet predictions for the entire image
    image_pred = np.zeros((img_height, img_width, 3), dtype=np.float32)

    for i in range(0, img_height - patch_size + step_size + 1, step_size):
        for j in range(0, img_width - patch_size + step_size + 1, step_size):
            patch = image[i:min(i + patch_size, img_height), j:min(j + patch_size, img_width)]
            patch_pred = predict_image(patch, model, I=256) # use the Unet model to predict the mask on the patch
            
            # Define the weights for blending the overlapping regions
            weights = np.ones_like(patch_pred)
            if i > 0:
                weights[:overlap, :] *= np.linspace(0, 1, overlap)[:, None, None]
            if j > 0:
                weights[:, :overlap] *= np.linspace(0, 1, overlap)[None, :, None]
            if i + patch_size < img_height:
                weights[-overlap:, :] *= np.linspace(1, 0, overlap)[:, None, None]
            if j + patch_size < img_width:
                weights[:, -overlap:] *= np.linspace(1, 0, overlap)[None, :, None]

            # Update image_pred with the weighted sum
            image_pred[i:min(i + patch_size, img_height), j:min(j + patch_size, img_width)] += patch_pred * weights
            labels, coords = label_grains(patch, patch_pred, dbs_max_dist=20.0)
            if len(coords) > 0: # run the SAM algorithm only if there are grains in the patch
                all_grains, labels, mask_all, grain_data, fig, ax = sam_segmentation(sam, patch, patch_pred, coords, labels, \
                                    min_area=min_area, plot_image=False, remove_edge_grains=True, remove_large_objects=remove_large_objects)
                for grain in all_grains:
                    All_Grains += [translate(grain, xoff=j, yoff=i)] # translate the grains to the original image coordinates
            patch_num = i//step_size*((img_width - patch_size + step_size)//step_size + 1) + j//step_size + 1
            if len(coords) > 0:
                coords[:,0] = coords[:,0] + j
                coords[:,1] = coords[:,1] + i
                if patch_num > 1:
                    all_coords = np.vstack((all_coords, coords))
                else:
                    all_coords = coords.copy()
            print(f"processed patch #{patch_num} out of {total_patches} patches")
    new_grains, comps, g = find_connected_components(All_Grains, min_area)
    All_Grains = merge_overlapping_polygons(All_Grains, new_grains, comps, min_area, patch_pred)
    return All_Grains, image_pred, all_coords

def load_and_preprocess(image_path, mask_path, augmentations=False):
    """
    Load and preprocess an image and its corresponding mask.

    Parameters
    ----------
    image_path : str
        The file path of the image.
    mask_path : str
        The file path of the mask.
    augmentations : bool, optional
        Whether to apply augmentations to the image. Defaults to False.

    Returns
    -------
    image : numpy.ndarray
        the preprocessed image.
    mask : numpy.ndarray
        the preprocessed mask.
    """
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)

    # Load mask and one-hot encode it
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.int32)
    mask = tf.one_hot(mask, depth=3, axis=-1)
    mask = tf.reshape(mask, (256, 256, 3))

    # Normalize images
    image = tf.cast(image, tf.float32) / 255.0

    # Apply augmentations
    if augmentations:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.where(image < 0, tf.zeros_like(image), image)  # clipping negative values
        image = tf.where(image > 1, tf.ones_like(image), image)  # clipping values larger than 1
        seed = tf.random.uniform(shape=[], minval=0, maxval=1)
        image = tf.cond(seed < 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        mask = tf.cond(seed < 0.5, lambda: tf.image.flip_left_right(mask), lambda: mask)
        image = tf.cond(seed < 0.5, lambda: tf.image.flip_up_down(image), lambda: image)
        mask = tf.cond(seed < 0.5, lambda: tf.image.flip_up_down(mask), lambda: mask)

    return image, mask

def onclick(event, ax, coords, image, predictor):
    """
    Run the SAM segmentation based on the prompt that comes from a mouse click event.
    If left mouse button is clicked, the point is used as an object (label=1) and a mask is added.
    If right mouse button is clicked, the point is used as background (label=0) and the 
    current mask is adjusted accordingly (if possible).

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        The mouse click event object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    coords : list
        A list to store the coordinates of the clicked points.
    image : numpy.ndarray
        The image data.
    predictor : SamPredictor
        The predictor object for segmentation.

    Returns
    -------
    None
    """
    x, y = event.xdata, event.ydata
    if event.button == 1: # left mouse button for object
        coords.append((x, y))
        sx, sy, mask = one_point_prompt(coords[-1][0], coords[-1][1], image, predictor, ax=ax)
        ax.figure.canvas.draw()
    if event.button == 3: # right mouse button for background
        ax.patches[-1].remove()
        coords.append((x, y))
        sx, sy = two_point_prompt(coords[-2][0], coords[-2][1], coords[-1][0], coords[-1][1], image, predictor, ax=ax)
        ax.figure.canvas.draw()
        
def onpress(event, ax, fig):
    """
    Handle key press events for deleting or merging polygons.

    Parameters
    ----------
    event : matplotlib.backend_bases.KeyEvent
        The key press event object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    sys.stdout.flush()
    if event.key == 'x': # delete last polygon
        ax.patches[-1].remove()
        fig.canvas.draw()
    if event.key == 'm': # merge last two polygons
        path1 = ax.patches[-1].get_path()
        path2 = ax.patches[-2].get_path()
        poly = Polygon(path1.vertices).union(Polygon(path2.vertices))
        ax.patches[-1].remove()
        ax.patches[-1].remove()
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        ax.fill(poly.exterior.xy[0], poly.exterior.xy[1], facecolor=color, edgecolor='k', alpha=0.5)
        fig.canvas.draw()
    if event.key == 'g': # make all polygons visible or invisible
        for patch in ax.patches:
            patch.set_visible(not patch.get_visible())
        fig.canvas.draw()

def onclick2(event, all_grains, grain_inds, ax, select_only=False):
    """
    Event handler function for selecting and highlighting grains in a plot,
    based on mouse click events. The selected grains then are either deleted or merged,
    using the 'onpress2' function.

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        The mouse click event object.
    all_grains : list
        A list of all the grains (polygons).
    grain_inds : list
        A list to store the indices of the selected grains.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object representing the plot.
    """
    x, y = event.xdata, event.ydata
    point = Point(x, y)
    for i in range(len(all_grains)):
        if all_grains[i].contains(point):
            grain_inds.append(i)
            if not select_only:
                ax.fill(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], color='r', alpha=0.5)
                ax.figure.canvas.draw()

def onpress2(event, all_grains, grain_inds, fig, ax):
    """
    Handle key press events when deleting or merging grains.

    Parameters
    ----------
    event : matplotlib.backend_bases.KeyEvent
        The key press event object.
    all_grains : list
        A list of all grains (polygons).
    grain_inds : list
        A list of indices corresponding to the grains.
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    sys.stdout.flush()
    if event.key == 'x': # delete last polygon
        ax.patches[-1].remove()
        ax.patches[grain_inds[-1]].remove()
        all_grains.remove(all_grains[grain_inds[-1]])
        fig.canvas.draw()
    if event.key == 'm': # merge last two polygons
        path1 = ax.patches[-1].get_path()
        path2 = ax.patches[-2].get_path()
        poly = Polygon(path1.vertices).union(Polygon(path2.vertices))
        poly = poly.buffer(10).buffer(-10)
        ax.patches[-1].remove()
        ax.patches[-1].remove()
        ax.patches[grain_inds[-1]].remove()
        if grain_inds[-2] < grain_inds[-1]:
            ax.patches[grain_inds[-2]].remove()
        else:
            ax.patches[grain_inds[-2]-1].remove()
        all_grains.remove(all_grains[grain_inds[-1]])
        if grain_inds[-2] < grain_inds[-1]:
            all_grains.remove(all_grains[grain_inds[-2]])
        else:
            all_grains.remove(all_grains[grain_inds[-2]-1])
        all_grains.append(poly)  # add merged polygon to 'all_grains'
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        ax.fill(poly.exterior.xy[0], poly.exterior.xy[1], 
                facecolor=color, edgecolor='k', linewidth=2, alpha=0.5)
        fig.canvas.draw()
    if event.key == 'g': # make all polygons visible or invisible
        for patch in ax.patches:
            patch.set_visible(not patch.get_visible())
        fig.canvas.draw()

def extract_patch(image, center, patch_size):
    """
    Extract a patch from the image centered on the given coordinates.

    Parameters
    ----------
    image : np.ndarray
        The large image from which to extract the patch.
    center : tuple
        The (x, y) coordinates of the center of the patch.
    patch_size : int
        The size of the patch (assumed to be square).

    Returns
    -------
    np.ndarray
        The extracted patch.
    """
    x, y = center
    half_size = patch_size // 2
    x_start = max(x - half_size, 0)
    y_start = max(y - half_size, 0)
    x_end = min(x + half_size, image.shape[1])
    y_end = min(y + half_size, image.shape[0])
    patch = image[y_start:y_end, x_start:x_end]
    return patch, (x_start, y_start)

def convert_to_large_image_coords(sx, sy, patch_origin):
    """
    Convert the coordinates from the patch to the large image.

    Parameters
    ----------
    sx : int
        The x-coordinate in the patch.
    sy : int
        The y-coordinate in the patch.
    patch_origin : tuple
        The (x, y) coordinates of the top-left corner of the patch in the large image.

    Returns
    -------
    tuple
        The (x, y) coordinates in the large image.
    """
    x_start, y_start = patch_origin
    x_large = x_start + sx
    y_large = y_start + sy
    return x_large, y_large

def onclick_large_image(event, ax, coords, image, predictor, patch_size=1000):
    """
    Handles mouse click events on a large image for segmentation purposes.

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        The mouse event that triggered the function.
    ax : matplotlib.axes.Axes
        The axes object where the image is displayed.
    coords : list of tuple
        List of coordinates where the user has clicked.
    image : numpy.ndarray
        The large image on which segmentation is performed.
    predictor : object
        The predictor object used for segmentation.
    patch_size : int, optional
        The size of the patch to extract around the clicked point (default is 1000).

    Notes
    -----
    - Left mouse button (event.button == 1) is used to add an object.
    - Right mouse button (event.button == 3) is used to remove the last added object.
    - The function updates the display with the segmented region.
    """
    x, y = event.xdata, event.ydata
    if event.button == 1: # left mouse button for object
        coords.append((x, y))
        patch, patch_origin = extract_patch(image, (int(np.round(coords[-1][0])), int(np.round(coords[-1][1]))), patch_size)
        predictor.set_image(patch)
        sx, sy, mask = one_point_prompt(coords[-1][0]-patch_origin[0], coords[-1][1]-patch_origin[1], patch, predictor, ax=False)
        x_large, y_large = convert_to_large_image_coords(sx, sy, patch_origin)
        if len(x_large) > 0:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            ax.fill(x_large, y_large, facecolor=color, edgecolor='k', alpha=0.5)
            ax.figure.canvas.draw()
    if event.button == 3: # right mouse button for background
        ax.patches[-1].remove()
        coords.append((x, y))
        patch, patch_origin = extract_patch(image, (int(np.round(coords[-1][0])), int(np.round(coords[-1][1]))), patch_size)
        predictor.set_image(patch)
        sx, sy = two_point_prompt(coords[-2][0]-patch_origin[0], coords[-2][1]-patch_origin[1], coords[-1][0]-patch_origin[0], coords[-1][1]-patch_origin[1], patch, predictor, ax=False)
        x_large, y_large = convert_to_large_image_coords(sx, sy, patch_origin)
        if len(x_large) > 0:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            ax.fill(x_large, y_large, facecolor=color, edgecolor='k', alpha=0.5)
            ax.figure.canvas.draw()

def click_for_scale(event, ax):
    """
    Handles mouse click events to measure the distance between two points on a plot.
    Prints the distance between the two points in number of pixels.

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        The mouse click event object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object representing the plot.
    """
    global x1, x2, y1, y2, dist
    if event.button == 1: # left mouse button for start point of scale
        x1, y1 = event.xdata, event.ydata
        ax.plot(x1, y1, 'ro')
        ax.figure.canvas.draw()
    if event.button == 3: # right mouse button for end point of scale
        x2, y2 = event.xdata, event.ydata
        ax.plot(x2, y2, 'ro')
        ax.plot([x1, x2], [y1, y2], 'r')
        ax.figure.canvas.draw()
        dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        print('number of pixels: ' + str(np.round(dist, 2)))

def get_grains_from_patches(ax, image, plotting=False):
    """
    Extract grains from patches on a plot and create a labeled image based on the updated set of grains.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the patches.
    image : numpy.ndarray
        The input image.

    Returns
    -------
    all_grains : list
        A list of Polygon objects representing the extracted grains.
    labels : numpy.ndarray
        The labeled image where each grain is assigned a unique label.
    mask_all : numpy.ndarray
        The binary mask image where grains and their boundaries are marked.
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    """
    all_grains = []
    for i in trange(len(ax.patches)):
        x = ax.patches[i].get_path().vertices[:,0]
        y = ax.patches[i].get_path().vertices[:,1]
        all_grains.append(Polygon(np.vstack((x, y)).T))

    ol_polys = find_overlapping_polygons(all_grains)
    polys_to_remove = []
    for polys in ol_polys:
        if all_grains[polys[0]].area >= all_grains[polys[1]].area:
            polys_to_remove.append(polys[1])
        else:
            polys_to_remove.append(polys[0])
    polys_to_remove.sort(reverse=True)
    for ind in polys_to_remove:
        all_grains.remove(all_grains[ind])
        
    # create labeled image
    rasterized = rasterize_grains(all_grains, image)
    boundaries = []
    for grain in all_grains:
        boundaries.append(grain.boundary.buffer(2))
    boundaries_rasterized = rasterize_grains(boundaries, image)
    mask_all = np.zeros(image.shape[:-1])
    mask_all[rasterized > 0] = 1
    mask_all[boundaries_rasterized >= 1] = 2
    if plotting:
        plt.figure()
        plt.imshow(rasterized)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        ax.imshow(mask_all, alpha=0.5)
        for i in trange(len(all_grains)):
            ax.fill(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 
                    facecolor=(0,0,1), edgecolor='none', linewidth=0.5, alpha=0.4)
    return all_grains, rasterized, mask_all

def plot_image_w_colorful_grains(image, all_grains, ax, cmap='viridis', plot_image=True, im_alpha=1.0):
    """
    Plot image with randomly colored grain masks.

    Parameters
    ----------
    image : numpy.ndarray
        The input image to be plotted.
    all_grains : list
        A list of shapely Polygon objects representing the grain masks.
    ax : matplotlib.axes.Axes
        The axes object on which to plot the image and grain masks.
    cmap : str, optional
        The name of the colormap to use for coloring the grain masks. Default is 'viridis'.
    plot_image : bool, optional
        Whether to plot the image. Default is True.

    Returns
    -------
    None
    """
    # Get the colormap object
    cmap = plt.cm.get_cmap(cmap)
    # Generate random indices for colors
    num_colors = len(all_grains)  # Number of colors to choose
    color_indices = np.random.randint(0, cmap.N, num_colors)
    # Get the individual colors
    colors = [cmap(i) for i in color_indices]
    if plot_image:
        ax.imshow(image, alpha=im_alpha)
    for i in trange(len(all_grains)):
        color = colors[i]
        ax.fill(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 
                facecolor=color, edgecolor='none', linewidth=1, alpha=0.5)
        ax.plot(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 
                color='k', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    plt.tight_layout()

def plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10):
    """
    Plot the axes and centroids of each grain on the given axis.

    Parameters
    ----------
    all_grains : list
        List of all grains.
    labels : numpy.ndarray
        Array of labeled regions.
    ax : matplotlib.axes.Axes
        The axis object to plot on.
    linewidth : int, optional
        Width of the lines to plot (default is 1).
    markersize : int, optional
        Size of the markers to plot (default is 10).

    Returns
    -------
    None
    """
    regions = regionprops(labels.astype('int'))
    for ind in range(len(all_grains)-1):
        y0, x0 = regions[ind].centroid
        orientation = regions[ind].orientation
        x1 = x0 + np.cos(orientation) * 0.5 * regions[ind].minor_axis_length
        y1 = y0 - np.sin(orientation) * 0.5 * regions[ind].minor_axis_length
        x2 = x0 - np.sin(orientation) * 0.5 * regions[ind].major_axis_length
        y2 = y0 - np.cos(orientation) * 0.5 * regions[ind].major_axis_length
        ax.plot((x0, x1), (y0, y1), '-k', linewidth=linewidth)
        ax.plot((x0, x2), (y0, y2), '-k', linewidth=linewidth)
        ax.plot(x0, y0, '.k', markersize=markersize)

def classify_points(feature1, feature2, x1, y1, x2, y2):
    """
    Classifies points based on their position relative to a line.

    Parameters
    ----------
    feature1 : list
        List of x-coordinates of the points.
    feature2 : list
        List of y-coordinates of the points.
    x1 : float
        x-coordinate of the first point on the line.
    y1 : float
        y-coordinate of the first point on the line.
    x2 : float
        x-coordinate of the second point on the line.
    y2 : float
        y-coordinate of the second point on the line.

    Returns
    -------
    list
        List of classifications for each point. Each classification is either 0 (on or one side of the line) or 1 (the other side of the line).

    """
    # Line equation coefficients
    A = y2 - y1
    B = x1 - x2
    C = (x2 * y1) - (x1 * y2)
    # Classify each point
    classifications = []
    for i in range(len(feature1)):
        x = feature1[i]
        y = feature2[i]
        position = A * x + B * y + C
        if position > 0:
            classifications.append(1)  # One side of the line
        elif position < 0:
            classifications.append(0)  # The other side of the line
        else:
            classifications.append(0)  # On the line
    return classifications

def compute_curvature(x,y):
    """
    Compute first derivatives and curvature of a curve.

    Parameters
    ----------
    x : 1D array
        x-coordinates of the curve
    y : 1D array
        y-coordinates of the curve

    Returns
    -------
    curvature : 1D array
        curvature of the curve (in 1/units of x and y)
    """
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)      
    ddx = np.gradient(dx) # second derivatives 
    ddy = np.gradient(dy) 
    curvature = (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
    return curvature

def patchify_training_data(input_dir, patch_dir):
    """
    Extracts patches from training images and labels, and saves them to the specified directory.

    Parameters
    ----------
    input_dir : str
        The directory containing the input images and labels.
    patch_dir : str
        The directory where the patches will be saved.

    Returns
    -------
    tuple
        A tuple containing the paths to the directories where the image patches and label patches are saved.

    Notes
    -----
    - The function expects the input directory to contain files with 'image' and 'mask' in their filenames.
    - The patches are extracted with a size of 256x256 pixels and a stride of 128 pixels.
    - The function creates subdirectories 'images' and 'labels' within the specified patch directory to save the patches.
    - If the input directory does not exist or contains no matching files, the function will print a warning and return.
    """
    if not os.path.exists(input_dir):
        print(f"Warning: The directory {input_dir} does not exist.")
        return

    images = sorted(glob(input_dir + "*image*"))
    labels = sorted(glob(input_dir + "*mask*"))

    if not images or not labels:
        print(f"Warning: No files containing 'image' or 'mask' found in {input_dir}.")
        return

    # Create directories for patches
    patches_dir = os.path.join(patch_dir, "Patches")
    images_dir = os.path.join(patches_dir, "images")
    labels_dir = os.path.join(patches_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    start_no = 0
    for image in tqdm(images):
        # Load the large image
        large_image = load_img(image)
        # Convert the image to a tensor
        large_image = tf.keras.preprocessing.image.img_to_array(large_image)
        # Reshape the tensor to have a batch size of 1
        large_image = tf.reshape(large_image, [1, *large_image.shape])
        # Extract patches from the large image
        patches = tf.image.extract_patches(
            images=large_image,
            sizes=[1, 256, 256, 1],
            strides=[1, 128, 128, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # Reshape the patches tensor to have a batch size of -1
        patches = tf.reshape(patches, [-1, 256, 256, 3])
        # Write patches to files
        for i in range(patches.shape[0]):
            im = np.asarray(patches[i,:,:,:]).astype('uint8')
            imname = os.path.join(images_dir, 'im%03d.png' % (start_no + i))
            im = Image.fromarray(im.astype(np.uint8))
            im.save(imname)
        start_no = start_no + patches.shape[0]

    start_no = 0
    for image in tqdm(labels):
        # Load the large image
        large_image = load_img(image)
        # Convert the image to a tensor
        large_image = tf.keras.preprocessing.image.img_to_array(large_image)
        large_image = large_image[:,:,0,np.newaxis] # only keep one layer and add a new axis
        # Reshape the tensor to have a batch size of 1
        large_image = tf.reshape(large_image, [1, *large_image.shape])
        # Extract patches from the large image
        patches = tf.image.extract_patches(
            images=large_image,
            sizes=[1, 256, 256, 1],
            strides=[1, 128, 128, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # Reshape the patches tensor to have a batch size of -1
        patches = tf.reshape(patches, [-1, 256, 256, 1])
        # Write patches to files
        for i in range(patches.shape[0]):
            im = np.asarray(patches[i,:,:,0]).astype('uint8')
            imname = os.path.join(labels_dir, 'im%03d.png' % (start_no + i))
            im = Image.fromarray(im.astype(np.uint8))
            im.save(imname)
        start_no = start_no + patches.shape[0]
    return images_dir, labels_dir

def create_train_val_test_data(image_dir, mask_dir, augmentation=True):
    """
    Splits image and mask data into training, validation, and test datasets, with optional augmentation.

    Parameters
    ----------
    image_dir : str
        Directory containing the image files.
    mask_dir : str
        Directory containing the mask files.
    augmentation : bool, optional
        If True, applies data augmentation to the training dataset (default is True).

    Returns
    -------
    train_dataset : tf.data.Dataset
        TensorFlow dataset for training.
    val_dataset : tf.data.Dataset
        TensorFlow dataset for validation.
    test_dataset : tf.data.Dataset
        TensorFlow dataset for testing.
    """

    image_files = sorted(glob(os.path.join(image_dir, '*.png')))
    mask_files = sorted(glob(os.path.join(mask_dir, '*.png')))

    batch_size = 32
    shuffle_buffer_size = 1000

    # First, split the data into training + validation and test sets
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        image_files,
        mask_files,
        test_size=0.15,  # 15% of the data for testing
        random_state=42  
    )
    # Then split the training + validation set into training and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images,
        train_val_masks,
        test_size=0.25,  # 25% of the remaining data for validation
        random_state=42  
    )

    if not augmentation:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks, tf.Variable([True] * len(train_images), dtype=tf.bool)))
    train_dataset = train_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
    test_dataset = test_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def create_and_train_model(train_dataset, val_dataset, test_dataset, model_file=None, epochs=100):
    """
    Create and train a U-Net model.

    Parameters
    ----------
    model_file : str
        Path to the file containing the model weights.
    train_dataset : tf.data.Dataset
        Training dataset.
    val_dataset : tf.data.Dataset
        Validation dataset.
    test_dataset : tf.data.Dataset
        Test dataset.
    epochs : int, optional
        Number of epochs to train the model (default is 100).

    Returns
    -------
    model : tf.keras.Model
        Trained U-Net model.

    Notes
    -----
    The function will plot the training and validation loss and accuracy over epochs.
    """
    if model_file:
        model = load_model(model_file, custom_objects={'weighted_crossentropy': weighted_crossentropy})
    else:
        model = Unet()
        model.compile(optimizer=Adam(), loss=weighted_crossentropy, metrics=["accuracy"])
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[1].plot(history.history['accuracy'])
    axes[1].plot(history.history['val_accuracy'])
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    model.evaluate(test_dataset)
    return model

def save_polygons(polygons, fname):
    """
    Save a list of polygons to a file in GeoJSON format.

    Parameters
    ----------
    polygons : list
        A list of Shapely polygon objects to be saved.
    fname : str
        The filename where the GeoJSON data will be saved.

    Returns
    -------
    None
    """
    # Convert polygons to GeoJSON-like format
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    for polygon in polygons:
        feature = {
            "type": "Feature",
            "geometry": mapping(polygon),  # Convert Shapely polygon to GeoJSON format
            "properties": {}
        }
        geojson_data["features"].append(feature)
    # Save to a file
    with open(fname, 'w') as f:
        json.dump(geojson_data, f)

def read_polygons(fname):
    """
    Reads polygons from a GeoJSON file.

    Parameters
    ----------
    fname : str
        The file path to the GeoJSON file.

    Returns
    -------
    list
        A list of Shapely Polygon objects extracted from the GeoJSON file.
    """
    # Load the GeoJSON file
    with open(fname, 'r') as f:
        geojson_data = json.load(f)
    # Extract the polygons
    polygons = []
    for feature in geojson_data['features']:
        geometry = feature['geometry']
        polygons.append(shape(geometry))  # Convert GeoJSON geometry to Shapely Polygon
    return polygons

def get_area_weighted_distribution(grain_sizes, areas):
    area_weighted_grain_size = []
    mean_area = np.mean(areas)
    for i in range(len(grain_sizes)):
        for j in range(int(areas[i]/(0.5*mean_area))):
            area_weighted_grain_size.append(grain_sizes[i])
    return area_weighted_grain_size

def plot_histogram_of_axis_lengths(major_axis_length, minor_axis_length, area=[], binsize=0.1, xlimits=None):

    """
    Plots a histogram of the major and minor axis lengths in phi scale.

    Parameters
    ----------
    major_axis_length : array-like
        The lengths of the major axes of the grains in millimeters.
    minor_axis_length : array-like
        The lengths of the minor axes of the grains in millimeters.
    area : array-like, optional
        The areas of the grains in square millimeters. If provided, the axis lengths will be weighted by the area.
    binsize : float, optional
        The size of the bins for the histogram. Default is 0.1.
    xlimits : tuple, optional
        The limits for the x-axis in millimeters. If not provided, the limits will be determined from the data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    if len(area)>0:
        major_axis_length = get_area_weighted_distribution(major_axis_length, area)
        minor_axis_length = get_area_weighted_distribution(minor_axis_length, area)
    phi_major = -np.log2(major_axis_length) # major axis length in phi scale
    phi_minor = -np.log2(minor_axis_length)
    if xlimits:
        phi_max = int(np.ceil(-np.log2(xlimits[0])))
        phi_min = int(np.floor(-np.log2(xlimits[1])))
    else:
        phi_max = int(np.ceil(max(np.max(phi_minor), np.max(phi_major))))
        phi_min = int(np.floor(min(np.min(phi_minor), np.min(phi_major))))
    fig, ax = plt.subplots(figsize=(8,6))
    n, bins, patches = plt.hist(phi_major, np.arange(phi_min, phi_max, binsize), alpha=0.5, zorder=2)
    ax.hist(phi_minor, np.arange(phi_min, phi_max, binsize), alpha=0.5)
    y_loc = max(n) - 0.2*max(n)
    grain_size_classes = {
        'very fine silt': [7, 8], 'fine silt': [6, 7], 'medium silt': [5, 6], 'coarse silt': [4, 5],
        'very fine sand': [3, 4], 'fine sand': [2, 3], 'medium sand': [1, 2], 'coarse sand': [0, 1],
        'very coarse sand': [-1, 0], 'granule': [-2, -1], 'pebble': [-6, -2], 'cobble': [-8, -6], 'boulder': [-12, -8]
    }
    matching_classes, bounds = find_grain_size_classes(grain_size_classes, phi_min, phi_max)
    bounds = np.array(sorted(bounds)[::-1])
    if xlimits:
        bounds = bounds[(bounds >= phi_min) & (bounds <= phi_max)]
    for i in range(len(bounds)-1):
            ax.text(bounds[i]*0.5 + bounds[i+1]*0.5 + 0.05, y_loc, matching_classes[i], rotation='vertical')
            ax.plot([bounds[i], bounds[i]], [0, max(n)+0.1*max(n)], 'k', linewidth=0.3)
    if xlimits:
        ax.set_xlim(phi_max, phi_min)
    else:
        ax.set_xlim(bounds[0], bounds[-1])
    ax.set_ylim(0, max(n) + 0.1*max(n))
    ax.set_xticks(np.arange(bounds[-1], bounds[0]+1))
    ax.set_xticklabels([np.round(2**i, 3) for i in range(-bounds[-1], -bounds[0]-1, -1)])
    ax.set_xlabel('grain axis length (mm)')
    ax.set_ylabel('count')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(bounds)
    ax2.set_xticklabels([i for i in bounds])
    ax2.set_xlabel('phi scale')
    phi_major_sorted = np.sort(phi_major)
    ecdf_major = np.arange(1, len(phi_major_sorted) + 1) / len(phi_major_sorted)
    phi_minor_sorted = np.sort(phi_minor)
    ecdf_minor = np.arange(1, len(phi_minor_sorted) + 1) / len(phi_minor_sorted)
    ax3 = ax.twinx()
    ax3.plot(phi_major_sorted, ecdf_major[::-1], color='tab:blue', linewidth=2, zorder=3)
    ax3.plot(phi_minor_sorted, ecdf_minor[::-1], color='tab:orange', linewidth=2, zorder=3)
    ax3.set_ylim(0, 1)
    return fig, ax

def find_grain_size_classes(grain_size_classes, value1, value2, xlimits=None):
    """
    Find grain size classes that overlap with a given range.

    Parameters
    ----------
    grain_size_classes : dict
        A dictionary where keys are grain size class names and values are tuples
        of (lower_bound, upper_bound) representing the size range of each class.
    value1 : float
        One end of the range to check for overlapping grain size classes.
    value2 : float
        The other end of the range to check for overlapping grain size classes.

    Returns
    -------
    matching_classes : list
        A list of grain size class names that overlap with the given range.
    bounds : list
        A list of unique bounds (both lower and upper) from the matching classes.
    """
    min_value = min(value1, value2)
    max_value = max(value1, value2)
    matching_classes = []
    bounds = []
    for grain_class, (lower_bound, upper_bound) in grain_size_classes.items():
        if lower_bound < max_value and upper_bound > min_value:
            matching_classes.append(grain_class)
            if lower_bound not in bounds:
                bounds.append(lower_bound)
            if upper_bound not in bounds:
                bounds.append(upper_bound)
    return matching_classes, bounds