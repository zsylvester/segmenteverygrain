import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import scipy.ndimage as ndi
from sklearn.cluster import DBSCAN
from tqdm import trange

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import load_img

from segment_anything import SamPredictor

def predict_image_tile(im_tile,model):
    """
    Predicts one image tile using a Unet model.

    Parameters
    ----------
    im_tile : 2D or 3D array
        The image tile for which the prediction will be done. Can have one or more channels.
    model :
        Tensorflow model used for semantic segmentation.

    Returns
    -------
    im_tile_pred : 3D array
        Predicted tile.
    """

    if len(np.shape(im_tile)) == 2:
        im_tile = np.expand_dims(im_tile, 2)
    im_tile = model.predict(np.stack((im_tile, im_tile), axis=0), verbose=0)
    im_tile_pred = im_tile[0] 
    return im_tile_pred

def predict_big_image(big_im, model, I):
    """
    Segmantic segmentation of the entire image using a Unet model.

    Parameters
    ----------
    big_im : 2D or 3D array
        The image that is being segmented. Can have one or more channels.
    model :
        Tensorflow model used for semantic segmentation.
    I : int
        Size of the square-shaped image tiles in pixels.

    Returns
    -------
    big_im_pred : 3D array
        Semantic segmentation result for the input image.
    """

    pad_rows = I - np.mod(big_im.shape[0], I)
    pad_cols = I - np.mod(big_im.shape[1], I)
    if len(np.shape(big_im)) == 2:
        big_im = np.vstack((big_im, np.zeros((pad_rows, big_im.shape[1]))))
        big_im = np.hstack((big_im, np.zeros((big_im.shape[0], pad_cols))))
    if len(np.shape(big_im)) == 3:
        big_im = np.vstack((big_im, np.zeros((pad_rows, big_im.shape[1], big_im.shape[2]))))
        big_im = np.hstack((big_im, np.zeros((big_im.shape[0], pad_cols, big_im.shape[2]))))
    r = int(np.floor(big_im.shape[0]/I)) # number of rows of image tiles
    c = int(np.floor(big_im.shape[1]/I)) # number of columns of image tiles
    
    I2 = int(I/2)
    W = np.hanning(I) * np.hanning(I)[:, np.newaxis]
    Wup = W.copy()
    Wup[:I2, :] = np.tile(np.hanning(I), (I2, 1))
    Wdown = W.copy()
    Wdown[I2:, :] = np.tile(np.hanning(I), (I2, 1))

    big_im = np.hstack((np.zeros((r*I, I2, 3)), big_im, np.zeros((r*I, I2, 3)))) # padding on the left and right sides
    big_im_pred = np.zeros((big_im.shape[0], big_im.shape[1], 3))
    print('segmenting image tiles...')
    for i in trange(c+1): # rows, no offset
        for j in range(1,2*r-2): # columns
            im_tile = big_im[j*I2:(j+2)*I2, i*I:(i+1)*I, :]/255.0
            im_tile_pred = predict_image_tile(im_tile, model)
            for layer in range(3):
                big_im_pred[j*I2:(j+2)*I2, i*I:(i+1)*I, layer] += im_tile_pred[:, :, layer] * W
    for i in range(c+1): # first row
        im_tile = big_im[:2*I2, i*I:(i+1)*I, :]/255.0
        im_tile_pred = predict_image_tile(im_tile, model)
        for layer in range(3):
            big_im_pred[:2*I2, i*I:(i+1)*I, layer] += im_tile_pred[:, :, layer] * Wup
    for i in range(c+1): # last row
        im_tile = big_im[(2*r-2)*I2:2*r*I2, i*I:(i+1)*I, :]/255.0
        im_tile_pred = predict_image_tile(im_tile,model)
        for layer in range(3):
            big_im_pred[(2*r-2)*I2:2*r*I2, i*I:(i+1)*I, layer] += im_tile_pred[:, :, layer] * Wdown
    for i in trange(c): # rows, half offset
        for j in range(1,2*r-2): # columns
            im_tile = big_im[j*I2:(j+2)*I2, i*I+I2:(i+1)*I+I2, :]/255.0
            im_tile_pred = predict_image_tile(im_tile,model)
            for layer in range(3):
                big_im_pred[j*I2:(j+2)*I2, i*I+I2:(i+1)*I+I2, layer] += im_tile_pred[:, :, layer] * W
    for i in range(c): # first row
        im_tile = big_im[:2*I2, i*I+I2:(i+1)*I+I2, :]/255.0
        im_tile_pred = predict_image_tile(im_tile,model)
        for layer in range(3):
            big_im_pred[:2*I2, i*I+I2:(i+1)*I+I2, layer] += im_tile_pred[:, :, layer] * Wup
    for i in range(c): # last row
        im_tile = big_im[(2*r-2)*I2:2*r*I2, i*I+I2:(i+1)*I+I2, :]/255.0
        im_tile_pred = predict_image_tile(im_tile,model)
        for layer in range(3):
            big_im_pred[(2*r-2)*I2:2*r*I2, i*I+I2:(i+1)*I+I2, layer] += im_tile_pred[:, :, layer] * Wdown

    big_im_pred = big_im_pred[:, I2:-I2, :] # crop the left and right side padding
    big_im_pred = big_im_pred[:-pad_rows, :-pad_cols, :] # get rid of padding
    return big_im_pred

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

def label_grains(big_im, big_im_pred, dbs_max_dist=20.0):
    """
    Label grains in semantic segmentation result and generate prompts for SAM model.

    Parameters
    ----------
    big_im : 2D or 3d array
        image that was segmented
    big_im_pred : 3D array
        semantic segmentation result
    dbs_max_dist : float
        DBSCAN distance parameter; decreasing it results in more SAM prompts and longer processing times

    Returns
    -------
    labels_simple : the labels as an image
    all_coords : pixel coordinates of the prompts
    """

    grains = big_im_pred[:,:,1].copy() # grain prediction from semantic segmentation result
    grains[grains >= 0.5] = 1
    grains[grains < 0.5] = 0
    grains = grains.astype('bool')
    labels_simple, n_elems = measure.label(grains, return_num = True, connectivity=1)
    props = regionprops_table(labels_simple, intensity_image = big_im, properties=('label', 'area', 'centroid'))
    grain_data_simple = pd.DataFrame(props)
    coords_simple = np.vstack((grain_data_simple['centroid-1'], grain_data_simple['centroid-0'])).T # use the centroids of the Unet grains as 'simple' prompts
    coords_simple = coords_simple.astype('int32')
    background_probs = big_im_pred[:,:,0][coords_simple[:,1], coords_simple[:,0]]
    inds = np.where(background_probs < 0.3)[0] # get rid of prompts that are likely to be background
    coords_simple = coords_simple[inds, :]

    bounds = big_im_pred[:,:,2].copy() # grain boundary prediction
    bounds[bounds >= 0.5] = 1
    bounds[bounds < 0.5] = 0
    bounds = bounds.astype('bool')
    temp_labels, n_elems = measure.label(bounds, return_num = True, connectivity=1)
    # Find the object with the largest area:
    label_counts = np.bincount(temp_labels.ravel())
    labels = np.where(label_counts > 100)[0][1:]
    largest_label = np.argmax(label_counts[1:]) + 1
    for label in labels:
        temp_labels[temp_labels == label] = largest_label
    bounds[temp_labels != largest_label] = 0
    bounds = bounds-1
    bounds[bounds < 0] = 1
    bounds = bounds.astype('bool')
    distance = ndi.distance_transform_edt(bounds)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=bounds.astype('bool'))
    background_probs = big_im_pred[:,:,0][coords[:,0], coords[:,1]]
    inds = np.where(background_probs < 0.3)[0]
    coords = coords[inds, :]
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=bounds)
    props = regionprops_table(labels, intensity_image = big_im, properties=('label', 'area', 'centroid', 'major_axis_length', 'minor_axis_length', 
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
        background_probs = big_im_pred[:,:,0][coords_ws[:,1], coords_ws[:,0]]
        inds = np.where(background_probs < 0.3)[0] # get rid of prompts that are likely to be background
        coords_ws = coords_ws[inds, :]
        all_coords = np.vstack((coords_ws, coords_simple))
    else:
        all_coords = coords_simple
    return labels_simple, all_coords

def one_point_prompt(x, y, image, predictor, ax=False):
    """
    Perform SAM segmentation using a single point prompt.

    Args:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
        image (numpy.ndarray): The input image.
        predictor: The SAM predictor.
        ax (bool, optional): Whether to plot the segmentation result on an axis. Defaults to False.

    Returns:
        sx: the x-coordinates of the contour points.
        sy: the y-coordinates of the contour points.
        mask: the segmented mask.

    """
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    ind = np.argmax(scores)
    if np.sum(masks[ind])/(image.shape[0]*image.shape[1]) > 0.1: # if mask is very large compared to size of the image
        scores = np.delete(scores, ind)
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
    return sx, sy, mask

def two_point_prompt(x1, y1, x2, y2, ax, image, predictor):
    """
    Perform a two-point-prompt-based segmentation using the SAM model. 
    Second point is used as background (label=0).

    Args:
        x1 (int): x-coordinate of the first point.
        y1 (int): y-coordinate of the first point.
        x2 (int): x-coordinate of the second point.
        y2 (int): y-coordinate of the second point.
        ax (matplotlib.axes.Axes): The axes to plot the segmentation result.
        image (numpy.ndarray): The input image.
        predictor: The segmentation predictor.

    Returns:
        numpy.ndarray: The x-coordinates of the contour points.
        numpy.ndarray: The y-coordinates of the contour points.
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
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    ax.fill(sx, sy, facecolor=color, edgecolor='k', alpha=0.5)
    return sx, sy

def find_overlapping_polygons(polygons):
    """
    Finds and returns a list of overlapping polygons from the given list of polygons.

    Args:
        polygons (list): A list of polygons.

    Returns:
        overlapping_polygons (list): A list of tuples representing the indices of overlapping polygons.

    """
    overlapping_polygons = []
    for i, poly1 in tqdm(enumerate(polygons)):
        for j, poly2 in enumerate(polygons):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if not poly1.is_valid:
                    poly1 = poly1.buffer(0)
                if not poly2.is_valid:
                    poly2 = poly2.buffer(0)
                if i != j and poly1.intersects(poly2) and poly1.intersection(poly2).area > 0.4*(min(poly1.area, poly2.area)):
                # if i != j and poly1.intersects(poly2) and calculate_iou(poly1, poly2) > 0.4:
                    overlapping_polygons.append((i, j))
    return overlapping_polygons

def Unet():
    """
    Creates a U-Net model for image segmentation.

    Returns:
    model: The U-Net model.
    """

    tf.keras.backend.clear_session()

    image = tf.keras.Input((256, 256, 3), name='input')
    
    conv1 = Conv2D(16, (3,3), activation='relu', padding = 'same')(image)
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
    model = Model(inputs=[image], outputs=[conv10])

    return model

def weighted_crossentropy(y_true, y_pred):
    """
    Calculates the weighted cross-entropy loss between the true labels and predicted labels.

    Args:
        y_true (tensor): True labels.
        y_pred (tensor): Predicted labels.

    Returns:
        loss: Weighted cross-entropy loss.

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

    Parameters:
    - img: The input image to be plotted.
    - label: The label image to be plotted.

    Returns:
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
    Calculates the Intersection over Union (IoU) between two polygons.

    Parameters:
    poly1 (Polygon): The first polygon.
    poly2 (Polygon): The second polygon.

    Returns:
    iou (float): The IoU value between the two polygons.
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

    Args:
        polygons (list): A list of polygons.

    Returns:
        most_similar_polygon (polygon): The most similar polygon.

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

def sam_segmentation(sam, big_im, big_im_pred, coords, labels, min_area, plot_image=False, remove_edge_grains=False, remove_large_objects=False):
    """
    Perform segmentation using the SAM algorithm.

    Parameters:
    - sam (SamPredictor): The SAM model.
    - big_im (numpy.ndarray): The input image.
    - big_im_pred (numpy.ndarray): The output of the Unet segmentation.
    - coords (numpy.ndarray): The coordinates of the SAM prompts.
    - labels (numpy.ndarray): The labeled image that comes from the 'label_grains' function.
    - min_area (int): The minimum area of the grains, in pixels.
    - plot_image (bool): Whether to plot the segmented image. Default is False.
    - remove_edge_grains (bool): Whether to remove grains that are touching the edge of the image. Default is False.
    - remove_large_objects (bool): Whether to remove large objects. Default is False. This is useful when the segmentation result is not very good.

    Returns:
    - all_grains (list): List of polygons representing the segmented grains.
    - labels (numpy.ndarray): The labeled image.
    - mask_all (numpy.ndarray): The mask of all grains.
    - grain_data (pandas.DataFrame): DataFrame containing properties of each grain.
    - fig (matplotlib.figure.Figure): The figure object if plot_image is True, otherwise None.
    - ax (matplotlib.axes.Axes): The axes object if plot_image is True, otherwise None.
    """
    predictor = SamPredictor(sam)
    predictor.set_image(big_im)
    all_grains = []
    print('creating masks using SAM...')
    for i in trange(len(coords[:,0])):
        x = coords[i,0]
        y = coords[i,1]
        sx, sy, mask = one_point_prompt(x, y, big_im, predictor)
        if remove_edge_grains and np.sum(np.hstack([mask[:4, :], mask[-4:, :], mask[:, :4].T, mask[:, -4:].T])) == 0: # if the mask is not touching too much the edge of the image
            labels_in_mask = np.unique(labels[mask])
            large_labels_in_mask = [label for label in labels_in_mask if len(labels[mask][labels[mask]==label]) >= 100] # if the mask contains a large grain
            if len(large_labels_in_mask) < 10 and np.mean(big_im_pred[:,:,0][mask]) < 0.7: # skip masks that are mostly background
                poly = Polygon(np.vstack((sx, sy)).T)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                all_grains.append(poly)
        if not remove_edge_grains:
            labels_in_mask = np.unique(labels[mask])
            large_labels_in_mask = [label for label in labels_in_mask if len(labels[mask][labels[mask]==label]) >= 100]
            if len(large_labels_in_mask) < 10 and np.mean(big_im_pred[:,:,0][mask]) < 0.7: # skip masks that are mostly background
                poly = Polygon(np.vstack((sx, sy)).T)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                all_grains.append(poly)

    print('finding overlapping polygons...')
    new_grains, comps = find_connected_components(all_grains, min_area)

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
                if iou < 0.8:
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
        cluster_0_mean = np.nanmean(np.array(N_neighbors_after)[np.array(classifications)==0])
        cluster_1_mean = np.nanmean(np.array(N_neighbors_after)[np.array(classifications)==1])
        if cluster_0_mean > cluster_1_mean: # the cluster with the larger number of neighbors after pruning the components is the one that we want to keep
            all_grains = np.array(all_grains)[np.array(Nodes)[np.array(classifications)==0]]
        else:
            all_grains = np.array(all_grains)[np.array(Nodes)[np.array(classifications)==1]]

        all_grains = list(all_grains) + new_grains

        print('finding overlapping polygons...')
        new_grains, comps = find_connected_components(all_grains, min_area)

    print('finding best polygons...')
    all_grains = merge_overlapping_polygons(all_grains, new_grains, comps, min_area)

    print('creating labeled image...')
    labels, mask_all = create_labeled_image(all_grains, big_im, big_im_pred, min_area)
    if plot_image:
        fig, ax = plt.subplots(figsize=(15,10))
        ax.imshow(big_im)
        plot_image_w_colorful_grains(big_im, all_grains, ax, cmap='Paired')
        plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10)
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, np.shape(big_im)[1]])
        plt.ylim([np.shape(big_im)[0], 0])
        plt.tight_layout()
    else:
        fig, ax = None, None
    props = regionprops_table(labels, intensity_image = big_im, properties=('label', 'area', 'centroid', 'major_axis_length', 'minor_axis_length', 
                                                                                    'orientation', 'perimeter', 'max_intensity', 'mean_intensity', 'min_intensity'))
    grain_data = pd.DataFrame(props)
    return all_grains, labels, mask_all, grain_data, fig, ax

def find_connected_components(all_grains, min_area):
    """
    Finds connected components in a graph of overlapping polygons.

    Args:
        all_grains (list): List of polygons representing all grains.
        min_area (float): Minimum area threshold for valid grains.

    Returns:
        new_grains (list): List of polygons that do not overlap and have an area greater than min_area.
        comps (list): List of sets, where each set represents a connected component of overlapping polygons.
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
    return new_grains, comps

def merge_overlapping_polygons(all_grains, new_grains, comps, min_area):
    """
    Merge overlapping polygons in a connected component.

    This function takes a list of all polygons, a list of polygons that do not overlap with other polygons, 
    a list of connected components, and a minimum area threshold.
    It iterates over each connected component and merges the overlapping polygons within that component.
    The most similar polygon is selected as the representative polygon for the merged region.
    If the area of the representative polygon is greater than the minimum area threshold, it is added to the new polygons list.

    Args:
        all_grains (list): List of all polygons.
        new_grains (list): List of polygons that do not overlap each other.
        comps (list): List of connected components.
        min_area (float): Minimum area threshold.

    Returns:
        list: List of merged polygons.

    """
    for j in trange(len(comps)): # deal with the overlapping objects, one connected component at a time
        polygons = [] # polygons in the connected component
        for i in comps[j]:
            polygons.append(all_grains[i])
        most_similar_polygon = pick_most_similar_polygon(polygons)
        if most_similar_polygon.area > min_area:
            new_grains.append(most_similar_polygon)
    all_grains = new_grains # replace original list of polygons
    return all_grains

def create_labeled_image(all_grains, big_im, big_im_pred, min_area):
    """
    Create a labeled image based on the provided grains and input images.

    Parameters:
    - all_grains (list): List of shapely Polygon objects representing the grains.
    - big_im (numpy.ndarray): Input image.
    - big_im_pred (numpy.ndarray): Predicted image (based on Unet model).
    - min_area (int): Minimum area threshold for filtering grains.

    Returns:
    - labels (numpy.ndarray): Labeled image where each grain is assigned a unique label.
    - mask_all (numpy.ndarray): Binary mask indicating the presence of grains and their boundaries.
    """
    # first rasterization of the grains:
    labels = np.zeros(big_im.shape[:-1])
    for i in trange(len(all_grains)):
        mask = rasterize(
            shapes=[all_grains[i]],
            out_shape=big_im.shape[:-1],
            fill=0,
            out=None,
            transform=rasterio.Affine(1.0, 0.0, 0.0,
               0.0, 1.0, 0.0),
            all_touched=False,
            default_value=1,
            dtype=None,
        )
        labels[(mask==1) & (labels==0)] = i+1
    # try to find the remaining grains:
    remaining_grains_im = big_im_pred[:,:,1].copy()
    remaining_grains_im[labels > 0] = 0
    remaining_grains_im[remaining_grains_im>0.8] = 1
    remaining_grains_im[remaining_grains_im<1] = 0
    remaining_grains_im = binary_erosion(remaining_grains_im, footprint=np.ones((9, 9)))
    remaining_grains_im = binary_dilation(remaining_grains_im, footprint=np.ones((12, 12)))
    labels_remaining_grains, n_elems = label(remaining_grains_im, return_num=True, connectivity=1)
    for i in range(1, n_elems):
        if np.sum(mask) > min_area:
            mask = np.zeros(np.shape(labels_remaining_grains))
            mask[labels_remaining_grains == i] = 1
            if np.sum(np.hstack([mask[:4, :], mask[-4:, :], mask[:, :4].T, mask[:, -4:].T])) == 0:
                contours = find_contours(mask, 0.5)
                sx = contours[0][:,1]
                sy = contours[0][:,0]
                if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[0, -1]):
                    mask = np.pad(mask, 1, mode='constant')
                    contours = find_contours(mask, 0.5)
                    sx = contours[0][:,1]
                    sy = contours[0][:,0]
                    if np.any(mask[1, :]):
                        sy = sy-1
                    if np.any(mask[:,1]):
                        sx = sx-1
                    mask = mask[1:-1, 1:-1]
                all_grains.append(Polygon(np.vstack((sx, sy)).T))
    # redo the rasterization with the updated set of grains:
    labels = np.zeros(big_im.shape[:-1])
    mask_all = np.zeros(big_im.shape[:-1])
    for i in trange(len(all_grains)):
        mask = rasterize(
            shapes=[all_grains[i]],
            out_shape=big_im.shape[:-1],
            fill=0,
            out=None,
            transform=rasterio.Affine(1.0, 0.0, 0.0,
               0.0, 1.0, 0.0),
            all_touched=False,
            default_value=1,
            dtype=None,
        )
        boundary_mask = rasterize(
            shapes = [all_grains[i].boundary.buffer(2)],
            out_shape=big_im.shape[:-1],
            fill = 0,
            out = None,
            transform=rasterio.Affine(1.0, 0.0, 0.0,
               0.0, 1.0, 0.0),
            all_touched=False,
            default_value=1,
            dtype=None,
        )
        mask_all[mask == 1] = 1
        mask_all[boundary_mask == 1] = 2
        labels[(mask==1) & (labels==0)] = i+1
    labels = labels.astype('int')
    return labels, mask_all

def predict_large_image(fname, model, sam, min_area, patch_size=4000, overlap=400):
    """
    Predicts the location of grains in a large image using a patch-based approach.

    Args:
        fname (str): The file path of the input image.
        model: The Unet model used forthe preliminary grain prediction.
        sam: The SAM model used for grain segmentation.
        min_area (int): The minimum area threshold for valid grains.
        patch_size (int, optional): The size of each patch. Defaults to 4000.
        overlap (int, optional): The overlap between patches. Defaults to 400.

    Returns:
        list: A list of grains represented as polygons.

    """
    step_size = patch_size - overlap  # step size for overlapping patches
    big_im = np.array(load_img(fname))
    img_height, img_width = big_im.shape[:2]  # get the height and width of the image 
    # loop over the image and extract patches:
    All_Grains = []
    for i in range(0, img_height - patch_size + step_size + 1, step_size):
        for j in range(0, img_width - patch_size + step_size + 1, step_size):
            patch = big_im[i:min(i + patch_size, img_height), j:min(j + patch_size, img_width)]
            patch_pred = predict_big_image(patch, model, I=256) # use the Unet model to predict the mask on the patch
            labels, coords = label_grains(patch, patch_pred, dbs_max_dist=20.0)
            if len(coords) > 0: # run the SAM algorithm only if there are grains in the patch
                all_grains, labels, mask_all, grain_data, fig, ax = sam_segmentation(sam, patch, patch_pred, coords, labels, \
                                    min_area=min_area, plot_image=False, remove_edge_grains=True, remove_large_objects=False)
                for grain in all_grains:
                    All_Grains += [translate(grain, xoff=j, yoff=i)] # translate the grains to the original image coordinates
    new_grains, comps = find_connected_components(All_Grains, min_area)
    All_Grains = merge_overlapping_polygons(All_Grains, new_grains, comps, min_area)
    return All_Grains

def load_and_preprocess(image_path, mask_path, augmentations=False):
    """
    Load and preprocess an image and its corresponding mask.

    Args:
        image_path (str): The file path of the image.
        mask_path (str): The file path of the mask.
        augmentations (bool, optional): Whether to apply augmentations to the image. Defaults to False.

    Returns:
        tuple: A tuple containing the preprocessed image and its corresponding mask.
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

    Parameters:
    - event: The mouse click event object.
    - ax: The matplotlib Axes object.
    - coords: A list to store the coordinates of the clicked points.
    - image: The image data.
    - predictor: The predictor object for segmentation.
    """
    x, y = event.xdata, event.ydata
    if event.button == 1: # left mouse button for object
        coords.append((x, y))
        sx, sy, mask = one_point_prompt(coords[-1][0], coords[-1][1], image, predictor, ax=ax)
        ax.figure.canvas.draw()
    if event.button == 3: # right mouse button for background
        ax.patches[-1].remove()
        coords.append((x, y))
        sx, sy = two_point_prompt(coords[-2][0], coords[-2][1], coords[-1][0], coords[-1][1], ax, image, predictor)
        ax.figure.canvas.draw()
        
def onpress(event, ax, fig):
    """
    Handle key press events for deleting or merging polygons.

    Parameters:
    - event: The key press event object.
    - ax: The matplotlib Axes object.
    - fig: The matplotlib Figure object.
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

def onclick2(event, all_grains, grain_inds, ax):
    """
    Event handler function for selecting and highlighting grains in a plot,
    based on mouse click events. The selected grains then are either deleted or merged.

    Parameters:
    - event: The mouse click event object.
    - all_grains: A list of all the grains (polygons).
    - grain_inds: A list to store the indices of the selected grains.
    - ax: The matplotlib Axes object representing the plot.
    """
    x, y = event.xdata, event.ydata
    point = Point(x, y)
    for i in range(len(all_grains)):
        if all_grains[i].contains(point):
            grain_inds.append(i)
            ax.fill(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], color='r', alpha=0.5)
            ax.figure.canvas.draw()

def onpress2(event, all_grains, grain_inds, fig, ax):
    """
    Handle key press events when deleting or merging grains.

    Parameters:
    - event: The key press event object.
    - all_grains: A list of all grains (polygons).
    - grain_inds: A list of indices corresponding to the grains.
    - fig: The figure object.
    - ax: The axes object.
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

def click_for_scale(event, ax):
    """
    Handles mouse click events to measure the distance between two points on a plot.
    Prints the distance between the two points in number of pixels.

    Parameters:
    - event: The mouse click event object.
    - ax: The matplotlib Axes object representing the plot.
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

def get_grains_from_patches(ax, image):
    """
    Extracts grains from patches on a plot and creates a labeled image based on the updated set of grains.

    Parameters:
    - ax: The matplotlib Axes object containing the patches.
    - image: The input image.

    Returns:
    - all_grains: A list of Polygon objects representing the extracted grains.
    - labels: The labeled image where each grain is assigned a unique label.
    - mask_all: The binary mask image where grains and their boundaries are marked.
    - fig: The matplotlib Figure object.
    - ax: The matplotlib Axes object.

    """
    all_grains = []
    for i in range(len(ax.patches)):
        x = ax.patches[i].get_path().vertices[:,0]
        y = ax.patches[i].get_path().vertices[:,1]
        all_grains.append(Polygon(np.vstack((x, y)).T))
        
    # create labeled image
    labels = np.zeros(image.shape[:-1])
    mask_all = np.zeros(image.shape[:-1])
    for i in trange(len(all_grains)):
        mask = rasterize(
            shapes=[all_grains[i]],
            out_shape=image.shape[:-1],
            fill=0,
            out=None,
            transform=rasterio.Affine(1.0, 0.0, 0.0,
               0.0, 1.0, 0.0),
            all_touched=False,
            default_value=1,
            dtype=None,
        )
        boundary_mask = rasterize(
            shapes = [all_grains[i].boundary.buffer(2)],
            out_shape=image.shape[:-1],
            fill = 0,
            out = None,
            transform=rasterio.Affine(1.0, 0.0, 0.0,
               0.0, 1.0, 0.0),
            all_touched=False,
            default_value=1,
            dtype=None,
        )
        mask_all[mask == 1] = 1
        mask_all[boundary_mask == 1] = 2
        labels[(mask==1) & (labels==0)] = i+1
    plt.figure()
    plt.imshow(labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.imshow(mask_all, alpha=0.5)
    for i in range(len(all_grains)):
        ax.fill(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 
                facecolor=(0,0,1), edgecolor='none', linewidth=0.5, alpha=0.4)
    return all_grains, labels, mask_all, fig, ax

def plot_image_w_colorful_grains(image, all_grains, ax, cmap='viridis', transparency=0.0):
    """
    Plot image with randomly colored grain masks.

    Parameters:
    - image: numpy.ndarray
        The input image to be plotted.
    - all_grains: list
        A list of shapely Polygon objects representing the grain masks.
    - ax: matplotlib.axes.Axes
        The axes object on which to plot the image and grain masks.
    - cmap: str, optional
        The name of the colormap to use for coloring the grain masks. Default is 'viridis'.
    - transparency: float, optional
        The transparency level of the image. Default is 0.0 (fully opaque).
    """
    # Choose a colormap
    colormap = cmap
    # Get the colormap object
    cmap = plt.cm.get_cmap(colormap)
    # Generate random indices for colors
    num_colors = len(all_grains)  # Number of colors to choose
    color_indices = np.random.randint(0, cmap.N, num_colors)
    # Get the individual colors
    colors = [cmap(i) for i in color_indices]
    ax.imshow(image, alpha=transparency)
    for i in range(len(all_grains)):
        color = colors[i]
        ax.fill(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 
                facecolor=color, edgecolor='none', linewidth=1, alpha=0.4)
        ax.plot(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 
                color='k', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10):
    """
    Plot the axes and centroids of each grain on the given axis.

    Parameters:
    - all_grains: List of all grains.
    - labels: Array of labeled regions.
    - ax: The axis object to plot on.
    - linewidth: Width of the lines to plot (default: 1).
    - markersize: Size of the markers to plot (default: 10).
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

    Args:
        feature1 (list): List of x-coordinates of the points.
        feature2 (list): List of y-coordinates of the points.
        x1 (float): x-coordinate of the first point on the line.
        y1 (float): y-coordinate of the first point on the line.
        x2 (float): x-coordinate of the second point on the line.
        y2 (float): y-coordinate of the second point on the line.

    Returns:
        list: List of classifications for each point. Each classification is either 0 (on or one side of the line) or 1 (the other side of the line).

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