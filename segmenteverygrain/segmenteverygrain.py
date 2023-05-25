import time
import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm, trange
import cv2
import networkx as nx
import rasterio
from rasterio.features import rasterize

from skimage import measure, morphology
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import label, binary_dilation, binary_erosion, reconstruction
from skimage.segmentation import watershed
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize

from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import scale
import scipy.ndimage as ndi
import scipy.interpolate

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


#  functions for assembling big image from prediction tiles:
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

def split_grains(im, selem_size):
    n_elems = 1
    eroded = im.copy()
    while (n_elems<2) & (1 in np.unique(eroded)):
        selem = np.ones((selem_size,selem_size))
        eroded = binary_erosion(eroded, footprint = selem)
        temp_labels, n_elems = measure.label(eroded, return_num = True)
        obj_features = regionprops(temp_labels)
        for i in range(len(obj_features)):
            if obj_features[i]['area'] < 10:
                eroded[temp_labels==i+1] = 0
        temp_labels, n_elems = measure.label(eroded, return_num = True)
        selem_size += 1
    if np.max(eroded)>0:
        distance = ndi.distance_transform_edt(eroded)
        rs = []; cs = []
        # for i in range(n_elems+1):
        for i in range(n_elems):
            dist1 = distance.copy()
            dist1[temp_labels != i+1] = 0
            r, c = np.unravel_index(np.argmax(dist1), np.shape(im))
            rs.append(r)
            cs.append(c)
        markers = np.zeros(np.shape(im))
        # for i in range(n_elems+1):
        for i in range(n_elems):
            markers[rs[i],cs[i]] = i+1
        distance = ndi.distance_transform_edt(im)
        gr_labels = watershed(-distance, markers, mask=im)
    else:
        gr_labels = im.copy()
    return gr_labels, eroded

def split_all_grains(ind, object_features, filled, labels, selem_size):
    minrow, mincol, maxrow, maxcol = object_features[ind]["bbox"]
    im = filled[max(0,minrow-2):maxrow+2,max(0,mincol-2):maxcol+2].copy()
    im[labels[max(0,minrow-2):maxrow+2,max(0,mincol-2):maxcol+2] != ind+1] = 0
    if maxcol >= filled.shape[1]:
        im = np.hstack((im, np.zeros((im.shape[0],2))))
    result, eroded = split_grains(im, selem_size)
    im[result==2] = 2
    start = time.time()
    while np.max(result)>1:
        if len(result[result==2]) > len(result[result==1]):
            result[result==1] = 0
            result[result==2] = 1
        else:
            result[result==1] = 1
            result[result==2] = 0
        result, eroded = split_grains(result, selem_size)
        im[result==2] = np.max(im)+1
        if time.time() - start > 1.0: # There are objects where this loop gets stuck. We don't want that.
            break
    return im

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

def label_and_dilate_grains(big_im, big_im_pred, small_grain_threshold, dilate_size, selem_size, splitting):
    # separate objects through labeling image regions
    boundary = (big_im_pred[:,:,2] > 0.5).astype(np.uint8) # set this to 0.95 to minimize false positives
    boundary[big_im_pred[:,:,0] > 0.5] = 1 # set this to 0.05 to minimize false positives

    labels, n_labels = label(boundary, background=1, return_num=True) # object labeling
    object_features = regionprops(labels)
    object_areas = [objf["area"] for objf in object_features]
    # find the indices of small objects:
    small_obj_inds = np.where(np.array(object_areas)<small_grain_threshold)[0]+1
    count = 0
    for ind in tqdm(small_obj_inds):  # this can take a long time for a large image!
        labels[labels == ind] = 0
        count += 1

    # fill holes:
    labels[labels>0] = 1
    seed = np.copy(labels)
    seed[1:-1, 1:-1] = labels.max()
    filled = reconstruction(seed, labels, method='erosion')

    labels, n_labels = label(filled, return_num = True) # re-label image
    object_features = regionprops(labels)

    n_grains = n_labels
    new_labels = labels.copy() # new labels for split grain clusters
    if splitting == True:
        for ind in trange(n_labels):
            minrow, mincol, maxrow, maxcol = object_features[ind]["bbox"]
            im = filled[max(0,minrow-2):maxrow+2,max(0,mincol-2):maxcol+2].copy()
            im[labels[max(0,minrow-2):maxrow+2,max(0,mincol-2):maxcol+2] != ind+1] = 0
            if maxcol >= filled.shape[1]:
              im = np.hstack((im, np.zeros((im.shape[0],2))))

            im = split_all_grains(ind, object_features, filled, labels, selem_size)
            if np.max(im)>1:
              for i in range(2,int(np.max(im)+1)):
                if maxcol >= filled.shape[1]:
                  im = im[:,:-2]
                new_labels[max(0,minrow-2):maxrow+2,max(0,mincol-2):maxcol+2][im == i] = n_grains+1
                n_grains += 1

    object_features = regionprops(new_labels)

    new_labels_dilated = new_labels.copy()
    for ind in trange(len(object_features)):
        minrow, mincol, maxrow, maxcol = object_features[ind]["bbox"]
        pad = 20
        im = new_labels[max(0,minrow-pad):maxrow+pad,max(0,mincol-pad):maxcol+pad].copy()
        im[new_labels[max(0,minrow-pad):maxrow+pad,max(0,mincol-pad):maxcol+pad] != object_features[ind]["label"]] = 0
        im[im>0] = 1
        grain_dist = ndi.distance_transform_edt(1 - im)
        temp = np.ones(np.shape(im))
        temp[grain_dist > dilate_size] = 0
        im = temp
        im[im==1] = ind+1
        new_labels_dilated[max(0,minrow-pad):maxrow+pad,max(0,mincol-pad):maxcol+pad][im==ind+1] = im[im==ind+1]

    props = regionprops_table(new_labels_dilated, intensity_image = big_im, properties=('label', 'area', 'centroid', 'major_axis_length', 'minor_axis_length', 
                                                                                        'orientation', 'perimeter', 'max_intensity', 'mean_intensity', 'min_intensity'))
    grain_data = pd.DataFrame(props)

    # print('plotting...')
    # create colormap with random colors, first color white (for background)
    # colors = []
    # for i in range(n_grains):
    #     l = list(np.random.random(3))
    #     l.append(1)
    #     colors.append(l)
    # colors[0] = [1,1,1,0]
    # cmap = ListedColormap(colors)

    # xs, ys = create_grain_outlines(big_im, new_labels_dilated)

    # fig = plt.figure(figsize=(30,20))
    # plt.imshow(big_im, cmap='gray', interpolation=None)
    # plt.imshow(new_labels_dilated, cmap=cmap, interpolation=None, alpha=0.4);
    # for i in range(len(grain_data['label'])):
    #     plt.plot(xs[i], ys[i],'k', linewidth = 1.0)
    # plt.xticks([])
    # plt.yticks([])
    # plt.xlim([0, np.shape(big_im)[1]])
    # plt.ylim([np.shape(big_im)[0], 0])
    # plt.tight_layout()
 
    return new_labels_dilated, grain_data

def create_grain_outlines(big_im, new_labels_dilated):
    object_features = regionprops(new_labels_dilated)
    filled = new_labels_dilated.copy()
    filled[filled>0] = 1
    n_grains = len(object_features)
    xs = []
    ys = []
    for ind in range(n_grains):
        minrow, mincol, maxrow, maxcol = object_features[ind]["bbox"]
        im = filled[max(0,minrow-2):maxrow+2,max(0,mincol-2):maxcol+2].copy()
        im[new_labels_dilated[max(0,minrow-2):maxrow+2,max(0,mincol-2):maxcol+2] != object_features[ind]["label"]] = 0
        if maxcol >= filled.shape[1]:
            im = np.hstack((im, np.zeros((im.shape[0],2))))
        contours = measure.find_contours(im, 0.5)
        sx = contours[0][:,1]
        sy = contours[0][:,0]
        xs.append(sx + mincol - 2 + 0.5)
        ys.append(sy + minrow - 2 + 0.5)
    return xs, ys

def compute_s_coordinates(x, y):
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)   
    ds = np.sqrt(dx**2+dy**2)
    s = np.hstack((0,np.cumsum(ds[1:])))
    return s

def get_grain_curvature(big_im, labels, regions, ind):
    filled = labels.copy()
    filled[filled>0] = 1
    minrow, mincol, maxrow, maxcol = regions[ind].bbox
    im = filled[max(0,minrow-2):maxrow+2,max(0,mincol-2):maxcol+2].copy()
    im[labels[max(0,minrow-2):maxrow+2,max(0,mincol-2):maxcol+2] != regions[ind]["label"]] = 0
    if maxcol >= filled.shape[1]:
        im = np.hstack((im, np.zeros((im.shape[0],2))))
    contours = measure.find_contours(im, 0.5)
    x = contours[0][:,1] + mincol - 2 + 0.5
    y = contours[0][:,0] + minrow - 2 + 0.5
    s = compute_s_coordinates(x, y)
    deltas = 1.0
    tck, u = scipy.interpolate.splprep([x,y],s=20, per=True) 
    unew = np.linspace(0,1,1+int(np.round(s[-1]/deltas))) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    sx, sy = out[0], out[1]
    curv = compute_curvature(sx,sy)
    s = compute_s_coordinates(sx, sy)
    y0, x0 = regions[ind].centroid
    orientation = regions[ind].orientation
    x1 = x0 + np.cos(orientation) * 0.5 * regions[ind].minor_axis_length
    y1 = y0 - np.sin(orientation) * 0.5 * regions[ind].minor_axis_length
    x2 = x0 - np.sin(orientation) * 0.5 * regions[ind].major_axis_length
    y2 = y0 - np.cos(orientation) * 0.5 * regions[ind].major_axis_length
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(big_im[max(0,minrow-10):maxrow+10, max(0,mincol-10):maxcol+10], cmap='gray', extent = [max(0,mincol-10), maxcol+10, maxrow+10, max(0,minrow-10)])
    # plt.scatter(sx, sy, s=60, c=curv, cmap='bwr_r', vmin = -0.1, vmax = 0.1)
    plt.plot(sx, sy, 'r', linewidth=2)
    ax.plot((x0, x1), (y0, y1), '-r', linewidth=1)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=1)
    ax.plot(x0, y0, '.g', markersize=15)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(max(0,mincol-10), maxcol+10)
    plt.ylim(maxrow+10, max(0,minrow-10))
    # plt.colorbar();
    return curv, sx, sy

def one_point_prompt(x, y, ax, image, predictor):
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
    r, c = np.shape(masks[ind])
    if masks[ind][0,0]: # if the mask contains the upper left corner of the image
        sx = np.hstack((-0.5, sx, -0.5))
        sy = np.hstack((-0.5, sy, -0.5))
    if masks[ind][0,-1]: # if the mask contains the upper right corner of the image
        sx = np.hstack((c-0.5, sx, c-0.5))
        sy = np.hstack((-0.5, sy, -0.5))
    if masks[ind][-1,0]: # if the mask contains the lower left corner of the image
        sx = np.hstack((-0.5, sx, -0.5))
        sy = np.hstack((r-0.5, sy, r-0.5))
    if masks[ind][-1,-1]: # if the mask contains the lower right corner of the image
        sx = np.hstack((c-0.5, sx, c-0.5))
        sy = np.hstack((r-0.5, sy, r-0.5))
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
    polys_to_be_removed = []
    for i, poly1 in tqdm(enumerate(polygons)):
        for j, poly2 in enumerate(polygons):
            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)
            if i != j and poly1.intersects(poly2) and poly1.intersection(poly2).area > min_overlap_area:
                if poly1.contains(poly2):
                    polys_to_be_removed.append(j)
                elif poly2.contains(poly1):
                    polys_to_be_removed.append(i)
                else:
                    overlapping_polygons.append((i, j))
                    overlap_areas.append(poly1.intersection(poly2).area)
    return overlapping_polygons, overlap_areas, polys_to_be_removed

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

def sam_segmentation(sam, big_im, big_im_pred, grain_data):
    predictor = SamPredictor(sam)
    predictor.set_image(big_im)
    fig, ax = plt.subplots(figsize=(15,10))
    ax.imshow(big_im) #, alpha=0.5)
    all_grains = []
    masks = []
    for i in trange(len(grain_data['centroid-1'])):
        x = grain_data['centroid-1'].iloc[i]
        y = grain_data['centroid-0'].iloc[i]
        sx, sy, mask = one_point_prompt(x, y, ax, big_im, predictor)
        if np.mean(big_im_pred[:,:,0][mask]) < 0.3: # skip masks that are mostly background
        # skip masks that hvae too much background:
        # if np.sum(big_im_pred[:,:,0][mask])/(np.sum(big_im_pred[:,:,1][mask]) + \
        #                                      np.sum(big_im_pred[:,:,2][mask])) < 0.3:
            all_grains.append(Polygon(np.vstack((sx, sy)).T))
            masks.append(mask)
    ax.clear()
    overlap_threshold = 20
    min_area = 20
    r = big_im.shape[0]
    c = big_im.shape[1]
    overlapping_polygons, overlap_areas, polys_to_be_removed = find_overlapping_polygons(all_grains, overlap_threshold)
    g = nx.Graph(overlapping_polygons)
    comps = list(nx.connected_components(g))
    connected_grains = set()
    for comp in comps:
        connected_grains.update(comp)
    for j in trange(len(comps)):
        # j = 0
        comb_masks_all = np.zeros((big_im.shape[0], big_im.shape[1]))
        count = 1
        for i in comps[j]:
            comb_masks_all += masks[i]*count
            count += 1
        if len(np.unique(comb_masks_all)) < 100:
            for i in np.unique(comb_masks_all):
                if i != 0:
                    if len(comb_masks_all[comb_masks_all == i]) > overlap_threshold:
                        new_mask = np.zeros(np.shape(comb_masks_all))
                        new_mask[comb_masks_all == i] = 1
                        # Label the objects in the binary image
                        labeled_image, num_labels = measure.label(new_mask, return_num=True)
                        # Find the object with the largest area
                        label_counts = np.bincount(labeled_image.ravel())
                        largest_label = np.argmax(label_counts[1:]) + 1
                        new_mask[labeled_image != largest_label] = 0
                        # Define a disk-shaped structuring element with radius 3
                        selem = morphology.disk(3)
                        # Erode the image using the structuring element
                        new_mask = morphology.binary_erosion(new_mask, selem)
                        # Dilate the eroded image using the same structuring element
                        new_mask = morphology.binary_dilation(new_mask, selem)
                        if np.max(new_mask):
                            contours = measure.find_contours(new_mask, 0.5)
                            sx = contours[0][:,1]
                            sy = contours[0][:,0]
                            if new_mask[0,0]:
                                sx = np.hstack((-0.5, sx, -0.5))
                                sy = np.hstack((-0.5, sy, -0.5))
                            if new_mask[0,-1]:
                                sx = np.hstack((c-0.5, sx, c-0.5))
                                sy = np.hstack((-0.5, sy, -0.5))
                            if new_mask[-1,0]:
                                sx = np.hstack((-0.5, sx, -0.5))
                                sy = np.hstack((r-0.5, sy, r-0.5))
                            if new_mask[-1,-1]:
                                sx = np.hstack((c-0.5, sx, c-0.5))
                                sy = np.hstack((r-0.5, sy, r-0.5))
                            all_grains.append(Polygon(np.vstack((sx, sy)).T))                       
    all_grains_new = []
    for i in range(len(all_grains)):
        if i not in connected_grains and all_grains[i].area > min_area and i not in polys_to_be_removed:
            all_grains_new.append(all_grains[i])
    all_grains = all_grains_new
    ax.imshow(big_im)
    # create labeled image
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
    ax.imshow(mask_all, alpha=0.5)  
    for i in range(len(all_grains)):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        ax.fill(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 
                facecolor=color, edgecolor='none', linewidth=0.5, alpha=0.4)
        # ax.plot(all_grains[i].exterior.xy[0], all_grains[i].exterior.xy[1], 'k', linewidth=0.5)
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

# def on_key(event, arg1, arg2, arg3):

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
    plt.figure()
    plt.imshow(image)
    plt.imshow(mask_all, alpha=0.5)
    return all_grains, labels, mask_all