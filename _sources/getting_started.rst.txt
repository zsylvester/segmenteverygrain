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

After the initial segmentation, you can interactively edit the results to delete unwanted grains or merge grains that should be combined:

.. code-block:: python

   grain_inds = []
   cid1 = fig.canvas.mpl_connect(
       "button_press_event",
       lambda event: seg.onclick2(event, all_grains, grain_inds, ax=ax),
   )
   cid2 = fig.canvas.mpl_connect(
       "key_press_event",
       lambda event: seg.onpress2(event, all_grains, grain_inds, fig=fig, ax=ax),
   )

Interactive controls:

* Click on a grain that you want to remove and press the 'x' key
* Click on two grains that you want to merge and press the 'm' key (they have to be the last two grains you clicked on)
* Press the 'g' key to hide the grain masks; press the 'g' key again to show the grain masks

After editing, disconnect the event handlers and update the grain data:

.. code-block:: python

   fig.canvas.mpl_disconnect(cid1)
   fig.canvas.mpl_disconnect(cid2)
   all_grains, labels, mask_all = seg.get_grains_from_patches(ax, image)

Adding new grains
~~~~~~~~~~~~~~~~~

You can also add new grains using the Segment Anything Model:

.. code-block:: python

   from segment_anything import SamPredictor
   
   predictor = SamPredictor(sam)
   predictor.set_image(image)  # this can take a while
   coords = []
   cid3 = fig.canvas.mpl_connect(
       "button_press_event", lambda event: seg.onclick(event, ax, coords, image, predictor)
   )
   cid4 = fig.canvas.mpl_connect(
       "key_press_event", lambda event: seg.onpress(event, ax, fig)
   )

Interactive controls for adding grains:

* Click on an unsegmented grain that you want to add
* Press the 'x' key to delete the last grain you added
* Press the 'm' key to merge the last two grains that you added
* Right click outside the grain (but inside the most recent mask) to restrict the grain to a smaller mask

Grain size analysis
~~~~~~~~~~~~~~~~~~~

To perform grain size analysis, you first need to establish the scale of your image by clicking on a scale bar:

.. code-block:: python

   cid5 = fig.canvas.mpl_connect(
       "button_press_event", lambda event: seg.click_for_scale(event, ax)
   )

Click on one end of the scale bar with the left mouse button and on the other end with the right mouse button. Then calculate the scale:

.. code-block:: python

   n_of_units = 10.0  # length of scale bar in real units (e.g., centimeters)
   units_per_pixel = n_of_units / scale_bar_length_pixels  # from the output above

Calculate grain properties and create a dataframe:

.. code-block:: python

   from skimage.measure import regionprops_table
   import pandas as pd
   
   props = regionprops_table(
       labels.astype("int"),
       intensity_image=image,
       properties=(
           "label", "area", "centroid", "major_axis_length", 
           "minor_axis_length", "orientation", "perimeter",
           "max_intensity", "mean_intensity", "min_intensity",
       ),
   )
   grain_data = pd.DataFrame(props)
   
   # Convert pixel measurements to real units
   grain_data["major_axis_length"] = grain_data["major_axis_length"].values * units_per_pixel
   grain_data["minor_axis_length"] = grain_data["minor_axis_length"].values * units_per_pixel
   grain_data["perimeter"] = grain_data["perimeter"].values * units_per_pixel
   grain_data["area"] = grain_data["area"].values * units_per_pixel**2

Saving results
~~~~~~~~~~~~~~

Save the grain data and masks:

.. code-block:: python

   import cv2
   
   # Save grain data to CSV
   grain_data.to_csv(fname[:-4] + ".csv")
   
   # Save mask as PNG
   cv2.imwrite(fname[:-4] + "_mask.png", mask_all)
   
   # Save processed image as PNG
   cv2.imwrite(fname[:-4] + "_image.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

Large image processing
~~~~~~~~~~~~~~~~~~~~~~

If you want to detect grains in large images, you should use the ``predict_large_image`` function, which will split the image into patches and run the Unet and SAM segmentations on each patch:

.. code-block:: python

   from PIL import Image
   Image.MAX_IMAGE_PIXELS = None  # needed for very large images
   
   all_grains, image_pred, all_coords = seg.predict_large_image(
       fname, model, sam, min_area=400.0, patch_size=2000, overlap=200
   )

Just like before, the ``all_grains`` list contains shapely polygons of the grains detected in the image. The image containing the grain labels can be generated like this:

.. code-block:: python

   labels = seg.rasterize_grains(all_grains, large_image)

See the `Segment_every_grain.ipynb <https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain.ipynb>`_ notebook for a complete example 
of how the models can be loaded and used for segmenting an image and QC-ing the result. The notebook goes through all the steps described above in an interactive format.

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

If you want to train a U-Net model from scratch, you can use the `Train_Unet_model.ipynb <https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Train_Unet_model.ipynb>`_ notebook, which mostly consists of the following code:

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

The U-Net model in the GitHub repository was trained using 66 images and the corresponding masks of a variety of grains, split into 44,533 patches of 256x256 pixels. 48 of these image-mask pairs are available at this Zenodo repository: https://zenodo.org/record/10058049. The model was trained for 200 epochs with a batch size of 32, using the Adam optimizer and a weighted cross-entropy loss function. The training accuracy was 0.937, the validation accuracy was 0.922, and the testing accuracy was 0.922 at the end of training. The model is available in the repository as 'seg_model.keras'.