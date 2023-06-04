import time
import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm, trange
from itertools import combinations
import cv2
import networkx as nx
import rasterio
from rasterio.features import rasterize

from skimage import measure, morphology
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import label, binary_dilation, binary_opening, reconstruction
from skimage.segmentation import watershed
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.feature import peak_local_max

from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.affinity import scale
from shapely.ops import unary_union
import scipy.ndimage as ndi
import scipy.interpolate
from sklearn.cluster import DBSCAN

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def predict_image_tile(im_tile,model):
    if len(np.shape(im_tile)) == 2:
        im_tile = np.expand_dims(im_tile, 2)
    im_tile = model.predict(np.stack((im_tile, im_tile), axis=0), verbose=0)
    im_tile_pred = im_tile[0] 
    return im_tile_pred

def predict_big_image(big_im, model, I):
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
    """function for computing first derivatives and curvature of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve
    curvature - curvature of the curve (in 1/units of x and y)"""
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)      
    ddx = np.gradient(dx) # second derivatives 
    ddy = np.gradient(dy) 
    curvature = (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
    return curvature

def get_grain_axes(poly):
    if (type(poly) == Polygon) and (type(poly.minimum_rotated_rectangle) == Polygon):
        mbr_points = list(zip(*poly.minimum_rotated_rectangle.exterior.coords.xy))
        # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]
        # get major/minor axis measurements
        minor_axis = min(mbr_lengths)
        major_axis = max(mbr_lengths)
        mrb = poly.minimum_rotated_rectangle
    else:
        minor_axis = np.nan
        major_axis = np.nan
        mrb = None
    return minor_axis, major_axis, mrb

def label_grains(big_im, big_im_pred, dbs_max_dist=20.0):
    grains = big_im_pred[:,:,1].copy() # grain prediction
    grains[grains >= 0.5] = 1
    grains[grains < 0.5] = 0
    grains = grains.astype('bool')
    labels_simple, n_elems = measure.label(grains, return_num = True, connectivity=1)
    props = regionprops_table(labels_simple, intensity_image = big_im, properties=('label', 'area', 'centroid', 'major_axis_length', 'minor_axis_length', 
                                                                                    'orientation', 'perimeter', 'max_intensity', 'mean_intensity', 'min_intensity'))
    grain_data_simple = pd.DataFrame(props)
    coords_simple = np.vstack((grain_data_simple['centroid-1'], grain_data_simple['centroid-0'])).T # use the centroids of the unet grains as 'simple' prompts'
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
    return labels_simple, grains, all_coords

def one_point_prompt(x, y, ax, image, predictor):
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
    sx = contours[0][:,1]
    sy = contours[0][:,0]
    r, c = np.shape(mask)
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
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    ax.fill(sx, sy, facecolor=color, edgecolor='k', alpha=0.5)
    return sx, sy, mask

def multi_point_prompt(x, y, ax, image, predictor):
    input_point = np.array([[x, y]])
    input_label = np.array([1])
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
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    ax.fill(sx, sy, facecolor=color, edgecolor='k', alpha=0.5)
    return sx, sy, masks[ind]

def two_point_prompt(x1, y1, x2, y2, ax, image, predictor):
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

def find_overlapping_polygons(polygons, min_overlap_area):
    overlapping_polygons = []
    overlap_areas = []
    # polys_to_be_removed = []
    for i, poly1 in tqdm(enumerate(polygons)):
        for j, poly2 in enumerate(polygons):
            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)
            if i != j and poly1.intersects(poly2) and poly1.intersection(poly2).area > min_overlap_area:
                overlapping_polygons.append((i, j))
                overlap_areas.append(poly1.intersection(poly2).area)
    return overlapping_polygons, overlap_areas #, polys_to_be_removed

def Unet():
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
    class_weights = tf.constant([[[[0.6, 1.0, 5.0]]]]) # increase the weight on the grains and the grain boundaries
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    weighted_losses = weights * unweighted_losses
    loss = tf.reduce_mean(weighted_losses)
    return loss

def plot_images_and_labels(img, label):
    fig = plt.figure(figsize = (12, 4))
    ax1 = fig.add_subplot(131)
    ax1.imshow(img)
    ax1.set_title("Grains")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = fig.add_subplot(132)
    ax2.imshow(label[:,:,0], cmap='Reds')
    ax2.set_title("Label")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3 = fig.add_subplot(133)
    ax3.imshow(img)
    ax3.imshow(label[:,:,0], alpha=0.3, cmap='Reds')
    ax3.set_title("Blending")
    ax3.set_xticks([])
    ax3.set_yticks([])

def calculate_iou(poly1, poly2):
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    iou = intersection_area / union_area
    return iou

def pick_most_similar_polygon(polygons):
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

def sam_segmentation(sam, big_im, big_im_pred, coords, labels, min_area):
    predictor = SamPredictor(sam)
    predictor.set_image(big_im)
    fig, ax = plt.subplots(figsize=(15,10))
    ax.imshow(big_im) #, alpha=0.5)
    all_grains = []
    masks = []
    for i in trange(len(coords[:,0])):
        x = coords[i,0]
        y = coords[i,1]
        sx, sy, mask = one_point_prompt(x, y, ax, big_im, predictor)
        labels_per_mask = len(np.unique(labels[mask]))
        if (labels_per_mask < 10) and (np.mean(big_im_pred[:,:,0][mask]) < 0.7): # skip masks that are mostly background
            poly = Polygon(np.vstack((sx, sy)).T)
            if not poly.is_valid:
                poly = poly.buffer(0)
            all_grains.append(poly)
            masks.append(mask)
    ax.clear()
    r = big_im.shape[0]
    c = big_im.shape[1]
    overlapping_polygons, overlap_areas = find_overlapping_polygons(all_grains, min_overlap_area=min_area)
    g = nx.Graph(overlapping_polygons)
    comps = list(nx.connected_components(g))
    connected_grains = set()
    for comp in comps:
        connected_grains.update(comp)
    new_grains = []
    for i in range(len(all_grains)): # first collect the objects that do not overlap each other
        if i not in connected_grains and all_grains[i].area > min_area:
            if not all_grains[i].is_valid:
                all_grains[i] = all_grains[i].buffer(0)
            new_grains.append(all_grains[i])
    for j in trange(len(comps)): # second deal with the overlapping objects, one connected component at a time
        polygons = [] # polygons in the connected component
        for i in comps[j]:
            polygons.append(all_grains[i])
        most_similar_polygon = pick_most_similar_polygon(polygons)
        if most_similar_polygon.area > min_area:
            new_grains.append(most_similar_polygon)
    all_grains = new_grains # replace original list of polygons
    # create labeled image:
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
    remaining_grains_im = big_im_pred[:,:,1].copy()
    remaining_grains_im[labels > 0] = 0
    remaining_grains_im[remaining_grains_im>0.8] = 1
    remaining_grains_im[remaining_grains_im<1] = 0
    remaining_grains_im = binary_opening(remaining_grains_im, footprint=np.ones((9, 9)))
    labels_remaining_grains, n_elems = measure.label(remaining_grains_im, return_num = True, connectivity=1)
    for i in range(1, n_elems):
        if np.sum(mask) > min_area:
            mask = np.zeros(np.shape(labels_remaining_grains))
            mask[labels_remaining_grains == i] = 1
            contours = measure.find_contours(mask, 0.5)
            sx = contours[0][:,1]
            sy = contours[0][:,0]
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
            all_grains.append(Polygon(np.vstack((sx, sy)).T))
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
    ax.imshow(big_im)
    plot_image_w_colorful_grains(big_im, all_grains, ax, cmap='Paired')
    plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, np.shape(big_im)[1]])
    plt.ylim([np.shape(big_im)[0], 0])
    plt.tight_layout()
    return all_grains, labels, mask_all, fig, ax

def load_and_preprocess(image_path, mask_path):
    # Load images
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
    seed = tf.random.experimental.stateless_split(tf.zeros([2], dtype=tf.int32), num=2)[0]
    image = tf.image.stateless_random_brightness(image, max_delta=0.1, seed=seed)
    image = tf.image.stateless_random_contrast(image, lower=0.9, upper=1.1, seed=seed)
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    image = tf.image.stateless_random_flip_up_down(image, seed=seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=seed)
    mask = tf.image.stateless_random_flip_up_down(mask, seed=seed)
    # this doesn't work for some reason (validation loss is too high)
    # if np.random.random() > 0.5: # only do this half the time 
    #     image = tf.image.stateless_random_crop(image, (128, 128, 3), seed=seed)
    #     image = tf.image.resize(image, (256, 256))
    #     mask = tf.image.stateless_random_crop(mask, (128, 128, 3), seed=seed)
    #     mask = tf.image.resize(mask, (256, 256), method='nearest')
    return image, mask

def onclick(event, ax, coords, image, predictor):
    x, y = event.xdata, event.ydata
    if event.button == 1: # left mouse button for object
        coords.append((x, y))
        sx, sy, mask = one_point_prompt(coords[-1][0], coords[-1][1], ax, image, predictor)
        ax.figure.canvas.draw()
    if event.button == 3: # right mouse button for background
        ax.patches[-1].remove()
        coords.append((x, y))
        sx, sy = two_point_prompt(coords[-2][0], coords[-2][1], coords[-1][0], coords[-1][1], ax, image, predictor)
        ax.figure.canvas.draw() 
        
def onpress(event, ax, fig):
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

def onclick2(event, all_grains, grain_inds, ax):
    x, y = event.xdata, event.ydata
    point = Point(x, y)
    for i in range(len(all_grains)):
        if all_grains[i].contains(point):
            grain_inds.append(i)
            ax.fill(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], color='r', alpha=0.5)
            ax.figure.canvas.draw()

def onpress2(event, all_grains, grain_inds, fig, ax):
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

def click_for_scale(event, ax):
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

def plot_image_w_colorful_grains(image, all_grains, ax, cmap='viridis'):
    # Choose a colormap
    colormap = cmap
    # Get the colormap object
    cmap = plt.cm.get_cmap(colormap)
    # Generate random indices for colors
    num_colors = len(all_grains)  # Number of colors to choose
    color_indices = np.random.randint(0, cmap.N, num_colors)
    # Get the individual colors
    colors = [cmap(i) for i in color_indices]
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    ax.imshow(image)
    for i in range(len(all_grains)):
        color = colors[i]
        ax.fill(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 
                facecolor=color, edgecolor='none', linewidth=1, alpha=0.4)
        ax.plot(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 
                color='k', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10):
    regions = regionprops(labels.astype('int'))
    for ind in range(len(all_grains)):
        y0, x0 = regions[ind].centroid
        orientation = regions[ind].orientation
        x1 = x0 + np.cos(orientation) * 0.5 * regions[ind].minor_axis_length
        y1 = y0 - np.sin(orientation) * 0.5 * regions[ind].minor_axis_length
        x2 = x0 - np.sin(orientation) * 0.5 * regions[ind].major_axis_length
        y2 = y0 - np.cos(orientation) * 0.5 * regions[ind].major_axis_length
        ax.plot((x0, x1), (y0, y1), '-k', linewidth=linewidth)
        ax.plot((x0, x2), (y0, y2), '-k', linewidth=linewidth)
        ax.plot(x0, y0, '.k', markersize=markersize)