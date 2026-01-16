Getting started
---------------

.. toctree::
   :caption: Getting started

The best way to use the `segmenteverygrain` package is to run the `Segment_every_grain.ipynb <https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain.ipynb>`_ notebook.

The notebook goes through the steps of loading the models, running the segmentation, interactively updating the result, and saving the grain data and the mask. The text below summarizes the steps that you need to take to run the segmentation.

Loading the models
~~~~~~~~~~~~~~~~~~

To load the U-Net model, you can use the 'load_model' function from Keras. The U-Net model is saved in the 'seg_model.keras' file.

.. code-block:: python

   import segmenteverygrain as seg
   from keras.saving import load_model
   model = load_model("seg_model.keras", custom_objects={'weighted_crossentropy': seg.weighted_crossentropy})

This assumes that you are using Keras 3 and 'seg_model.keras' was saved using Keras 3. Older models created with a ``segmenteverygrain`` version that was based on Keras 2 do not work with with the latest version of the package.

The Segment Anything model can be downloaded from this `link <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth>`_. You can also download it programmatically:

.. code-block:: python

   import os
   import urllib.request
   
   if not os.path.exists("./models/sam_vit_h_4b8939.pth"):
       url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
       urllib.request.urlretrieve(url, "./models/sam_vit_h_4b8939.pth")

Running the segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

To run the U-Net segmentation on an image and label the grains in the U-Net output:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from keras.utils import load_img
   
   # Load your image
   fname = "path/to/your/image.jpg"
   image = np.array(load_img(fname))
   
   # Run U-Net prediction
   image_pred = seg.predict_image(image, model, I=256)
   labels, coords = seg.label_grains(image, image_pred, dbs_max_dist=20.0)
    
The input image should not be much larger than ~2000x3000 pixels, in part to avoid long running times; it is supposed to be a numpy array with 3 channels (RGB).
Grains should be well defined in the image and not too small (e.g., only a few pixels in size).

Quality control of U-Net prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The U-Net prediction should be QC-d before running the SAM segmentation:

.. code-block:: python

   plt.figure(figsize=(15,10))
   plt.imshow(image_pred)
   plt.scatter(np.array(coords)[:,0], np.array(coords)[:,1], c='k')
   plt.xticks([])
   plt.yticks([])

The black dots in the figure represent the SAM prompts that will be used for grain segmentation. If the U-Net segmentation is of low quality, the base model can be (and should be) finetuned using the steps outlined :ref:`below<Finetuning the U-Net model>`.

SAM segmentation
~~~~~~~~~~~~~~~~

Here is an example showing how to run the SAM segmentation on an image, using the outputs from the U-Net model:

.. code-block:: python

   from segment_anything import sam_model_registry
   sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth") # load the SAM model
   all_grains, labels, mask_all, grain_data, fig, ax = seg.sam_segmentation(sam, image, image_pred, coords, labels, min_area=400.0, plot_image=True, remove_edge_grains=False, remove_large_objects=False)

The ``all_grains`` list contains shapely polygons of the grains detected in the image. ``labels`` is an image that contains the labels of the grains. 
``grain_data`` is a pandas dataframe with a number of grain parameters.

Interactive editing of results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the initial segmentation, you can interactively edit the results using the ``GrainPlot`` class from the ``interactions`` module. This provides a modern, interactive interface for deleting, merging, and adding grains.

First, convert the polygons to ``Grain`` objects and set up the SAM predictor:

.. code-block:: python

   import segmenteverygrain.interactions as si
   from segment_anything import SamPredictor
   from tqdm import tqdm

   # Convert polygons to Grain objects
   grains = si.polygons_to_grains(all_grains, image=image)
   for g in tqdm(grains, desc='Measuring detected grains'):
       g.measure()

   # Set up SAM predictor for adding new grains
   predictor = SamPredictor(sam)
   predictor.set_image(image)

Then create the interactive plot:

.. code-block:: python

   plot = si.GrainPlot(
       grains,
       image=image,
       predictor=predictor,
       blit=True,                      # Use blitting for faster rendering
       color_palette='tab20b',         # Matplotlib colormap for grain colors
       figsize=(12, 8),                # Figure size in inches
       scale_m=500*1e-6,               # Length of scale bar in meters (for unit conversion)
       px_per_m=1.0,                   # Pixels per meter (will be updated if scale bar is drawn)
   )
   plot.activate()

Interactive controls (also shown in the figure title bar):

**Mouse controls:**

* **Left-click on existing grain**: Select/unselect the grain
* **Left-click in grain-free area**: Place foreground prompt for instant grain creation (auto-create)
* **Alt + Left-click**: Place foreground prompt for multi-prompt grain creation (hold Alt, click multiple times, release Alt to create)
* **Alt + Right-click**: Place background prompt for multi-prompt creation
* **Shift + Left-drag**: Draw a scale bar line (red line) for unit conversion
* **Middle-click** or **Shift + Left-click on grain**: Show grain measurement info

**Keyboard controls:**

* **d** or **Delete**: Delete selected (highlighted) grains
* **m**: Merge selected grains (must be touching)
* **z**: Undo (delete the most recently created grain)
* **Ctrl** (hold): Temporarily hide all grain masks
* **Esc**: Remove all prompts and unselect all grains
* **c**: Create grain from existing prompts (alternative to auto-create)

After editing, retrieve the updated grains and deactivate the interactive features:

.. code-block:: python

   grains = plot.get_grains()  # Get the edited list of grains
   plot.deactivate()           # Turn off interactive features

   # Optionally draw the major and minor axes on each grain
   plot.draw_axes()

Scale bar and unit conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert measurements from pixels to real-world units, you can either:

1. **Specify the scale when creating the plot** using the ``px_per_m`` parameter (pixels per meter).

2. **Draw a scale bar interactively**: Hold ``Shift`` and drag the mouse to draw a line on a known reference object. The ``scale_m`` parameter specifies the real-world length of this reference object in meters.

After drawing a scale bar, the pixel-to-meter conversion is automatically updated:

.. code-block:: python

   # Retrieve the scale after drawing a scale bar
   px_per_m = plot.px_per_m

Grain size analysis
~~~~~~~~~~~~~~~~~~~

Generate a summary dataframe and histogram of grain measurements:

.. code-block:: python

   # Get summary dataframe with all grain measurements
   summary = si.get_summary(grains, px_per_m=plot.px_per_m)

   # Create histogram of major and minor axis lengths
   hist = seg.plot_histogram_of_axis_lengths(
       summary['major_axis_length'] * 1000,  # Convert to mm
       summary['minor_axis_length'] * 1000,
       binsize=0.25
   )

Saving results
~~~~~~~~~~~~~~

The ``interactions`` module provides convenient functions for saving all results:

.. code-block:: python

   out_fn = "./examples/output/my_image"  # Base filename (without extension)

   # Save grain shapes as GeoJSON
   si.save_grains(out_fn + '_grains.geojson', grains)

   # Save the plot with grain overlays
   plot.savefig(out_fn + '_grains.jpg')

   # Save grain measurements as CSV
   summary = si.save_summary(out_fn + '_summary.csv', grains, px_per_m=plot.px_per_m)

   # Save histogram as image
   si.save_histogram(out_fn + '_summary.jpg', summary=summary)

   # Save binary mask for training (0-1 values)
   si.save_mask(out_fn + '_mask.png', grains, image, scale=False)

   # Save human-readable mask (0-255 values)
   si.save_mask(out_fn + '_mask2.jpg', grains, image, scale=True)

Large image processing
~~~~~~~~~~~~~~~~~~~~~~

If you want to detect grains in large images, you should use the ``predict_large_image`` function, which will split the image into patches and run the Unet and SAM segmentations on each patch:

.. code-block:: python

   from PIL import Image
   import segmenteverygrain.interactions as si

   Image.MAX_IMAGE_PIXELS = None  # needed for very large images

   fname = "./examples/my_large_image.jpg"
   image = si.load_image(fname)

   all_grains, image_pred, all_coords = seg.predict_large_image(
       fname, unet, sam,
       min_area=400.0,
       patch_size=2000,
       overlap=200,
       remove_edge_grains=False  # Keep grains on outer edges of the full image
   )

Just like before, the ``all_grains`` list contains shapely polygons of the grains detected in the image. The image containing the grain labels can be generated like this:

.. code-block:: python

   labels = seg.rasterize_grains(all_grains, image)

To interactively edit the results, convert to Grain objects and use GrainPlot:

.. code-block:: python

   grains = si.polygons_to_grains(all_grains, image=image)
   predictor.set_image(image)

   plot = si.GrainPlot(
       grains,
       image=image,
       predictor=predictor,
       color_palette='tab20b',
   )
   plot.activate()

See the `Segment_every_grain.ipynb <https://github.com/zsylvester/segmenteverygrain/blob/main/Segment_every_grain.ipynb>`_ notebook for a complete example
of how the models can be loaded and used for segmenting an image and QC-ing the result. The notebook goes through all the steps described above in an interactive format.


Grain extraction and clustering
-------------------------------

The ``grain_utils`` module provides functions for extracting individual grain images and clustering them for classification tasks.

Extracting individual grains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract standardized, square images of individual grains suitable for machine learning:

.. code-block:: python

   from segmenteverygrain import extract_all_grains, extract_grain_image

   # Extract all grains as standardized 224x224 images
   grain_images, grain_masks, grain_preds = extract_all_grains(
       all_grains, image, image_pred, target_size=224
   )

   # Or extract a single grain
   grain_img, grain_mask, grain_pred, orig_size = extract_grain_image(
       all_grains[0], image, image_pred, target_size=224, pad=10
   )

Feature extraction
~~~~~~~~~~~~~~~~~~

Extract deep learning features using pre-trained CNNs for clustering or classification:

.. code-block:: python

   from segmenteverygrain import extract_vgg16_features, extract_color_features

   # Extract VGG16 features (4096-dimensional)
   features, model = extract_vgg16_features(grain_images, model_name='VGG16')

   # Or extract color-based features
   color_features = extract_color_features(grain_images, color_space='hsv')

Clustering grains
~~~~~~~~~~~~~~~~~

Cluster grains based on their features:

.. code-block:: python

   from segmenteverygrain import cluster_grains, create_clustered_grain_montage

   # Cluster using K-means with PCA dimensionality reduction
   labels, reduced_features, pca, clusterer = cluster_grains(
       features, n_clusters=10, n_components=25
   )

   # Create a visual montage of clustered grains
   montage, cluster_info = create_clustered_grain_montage(
       labels, grain_images, grid_cols=20, draw_boundaries=True
   )

Interactive grain selection and labeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``ClusterMontageSelector`` for quality control (removing bad grains or clusters):

.. code-block:: python

   from segmenteverygrain import ClusterMontageSelector

   selector = ClusterMontageSelector(labels, grain_images, all_grains)
   selector.activate()  # Interactive mode

   # After selection, get filtered results
   filtered_grains = selector.get_filtered_grains()

Use the ``ClusterMontageLabeler`` for labeling grains with custom categories:

.. code-block:: python

   from segmenteverygrain import ClusterMontageLabeler

   labeler = ClusterMontageLabeler(
       labels, grain_images, all_grains,
       label_names=['quartz', 'feldspar', 'lithic', 'other']
   )
   labeler.activate()  # Interactive mode

   # Export labels
   labeler.export_labels('grain_labels.csv')
   labeler.save_labeled_images('labeled_grains/')

Visualizing classified grains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot grains colored by their classification:

.. code-block:: python

   from segmenteverygrain import plot_classified_grains

   fig, ax = plot_classified_grains(
       image, all_grains, classifications,
       class_colors={'quartz': 'blue', 'feldspar': 'red', 'other': 'green'}
   )

Hardware requirements
~~~~~~~~~~~~~~~~~~~~~

For training a new U-Net model or fine tuning the existing one, GPU access is necessary. The easiest way of getting access to a powerful GPU is Google Colab. In inference mode, a moderately powerful computer with at least 16 GB of memory should be enough. That said, larger CPU speeds and more memory will significantly reduce inference time.


Finetuning the U-Net model
--------------------------

The last section of the `Segment_every_grain.ipynb <https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain.ipynb>`_ notebook shows how to finetune the U-Net model. The first step is to create patches (usually 256x256 pixels in size) from the images and the corresponding masks that you want to use for training.

.. code-block:: python

   image_dir, mask_dir = seg.patchify_training_data(input_dir, patch_dir)

The ``input_dir`` should contain the images and masks that you want to use for training. These files should have 'image' and 'mask' in their filenames, for example, 'sample1_image.png' and 'sample1_mask.png'. An example image can be found `here <https://github.com/zsylvester/segmenteverygrain/blob/main/torrey_pines_beach_image.jpeg>`_; and the corresponding mask is `here <https://github.com/zsylvester/segmenteverygrain/blob/main/torrey_pines_beach_mask.png>`_.

The mask is an 8-bit image and should contain only three numbers: 0, 1, and 2. 0 is the background, 1 is the grain, and 2 is the grain boundary. Usually the mask is generated using the ``segmenteverygrain`` workflow, that is, by running the U-Net segmentation first, the SAM segmentation second, and then cleaning up the result. That said, when the U-Net ouputs are of low quality, it might be a good idea to generate the masks directly with SAM. Once you have a good mask, you can save it using ``cv2.imwrite`` (see also the example notebook):

.. code-block:: python

   cv2.imwrite('sample1_mask.png', mask)

The ``patch_dir`` is the directory where the patches will be saved. A folder named 'Patches' will be created in this directory, and the patches will be saved in subfolders named 'images' and 'labels'.

Next, training, validation, and test datasets are created from the patches:

.. code-block:: python

   train_dataset, val_dataset, test_dataset = seg.create_train_val_test_data(image_dir, mask_dir, augmentation=True)

Now we are ready to load the existing model weights and to train the model:

.. code-block:: python

   model = seg.create_and_train_model(train_dataset, val_dataset, test_dataset, model_file='seg_model.keras', epochs=100)

If you are happy with the finetuned model, you will want to save it:

.. code-block:: python

   model.save('seg_model_finetuned.keras')

If you want to use this new model to make predictions, you will need to load it with the custom loss function:

.. code-block:: python

   model = load_model("seg_model_finetuned.keras", custom_objects={'weighted_crossentropy': seg.weighted_crossentropy})


Training the U-Net model from scratch
-------------------------------------

If you want to train a U-Net model from scratch, you can use the `Train_Unet_model.ipynb <https://github.com/zsylvester/segmenteverygrain/blob/main/Train_Unet_model.ipynb>`_ notebook, which mostly consists of the code snippets below.

.. code-block:: python

   import segmenteverygrain as seg
   model = seg.Unet() # create model
   model.compile(optimizer=Adam(), loss=seg.weighted_crossentropy, metrics=["accuracy"])

If you place the training images and masks in the 'images' directory (the filenames are supposed to terminate with '_image.png' and '_mask.png'), you can create the training dataset like this:

.. code-block:: python

   input_dir = "../images/"
   patch_dir = "../patches/"
   image_dir, mask_dir = seg.patchify_training_data(input_dir, patch_dir)
   image_dir = '../patches/Patches/images'
   mask_dir = '../patches/Patches/labels'
   train_dataset, val_dataset, test_dataset = seg.create_train_val_test_data(image_dir, mask_dir, augmentation=True)

Then you can train and test the model:

.. code-block:: python

   model = seg.create_and_train_model(train_dataset, val_dataset, test_dataset, epochs=200)

The model can be saved using the Keras save method:

.. code-block:: python

   model.save('seg_model.keras')

The U-Net model in the GitHub repository was trained using 66 images and the corresponding masks of a variety of grains, split into 44,533 patches of 256x256 pixels. 48 of these image-mask pairs are available at this Zenodo repository: https://zenodo.org/records/15786086. The model was trained for 200 epochs with a batch size of 32, using the Adam optimizer and a weighted cross-entropy loss function. The training accuracy was 0.937, the validation accuracy was 0.922, and the testing accuracy was 0.922 at the end of training. The model is available in the repository as 'seg_model.keras'.