Getting started
---------------

.. toctree::
   :caption: Getting started

To load the Unet model:

.. code-block:: python

   import segmenteverygrain as seg
   model = seg.Unet()
   model.compile(optimizer=Adam(), loss=seg.weighted_crossentropy, metrics=["accuracy"])
   model.load_weights('./checkpoints/seg_model')

To run the Unet segmentation on an image and label the grains in the Unet output:

.. code-block:: python

   image_pred = seg.predict_image(image, model, I=256)
   labels, coords = seg.label_grains(image, image_pred, dbs_max_dist=20.0)
    
The input image should not be much larger than ~2000x3000 pixels, in part to avoid long running times; it is supposed to be a numpy array with 3 channels (RGB).
Grains should be well defined in the image and not too small (e.g., only a few pixels in size).
The Unet prediction should be QC-d before running the SAM segmentation:

.. code-block:: python

   plt.figure(figsize=(15,10))
   plt.imshow(big_im_pred)
   plt.scatter(np.array(coords)[:,0], np.array(coords)[:,1], c='k')
   plt.xticks([])
   plt.yticks([]);

If the Unet segmentation is of low quality, the base model can be (and should be) fine tuned using the ``Train_seg_unet_model.ipynb`` notebook.

To run the SAM segmentation on an image, using the outputs from the Unet model:

.. code-block:: python

   all_grains, labels, mask_all, grain_data, fig, ax = seg.sam_segmentation(sam, image, image_pred, coords, labels, min_area=400.0, plot_image=True, remove_edge_grains=False, remove_large_objects=False)

The ``all_grains`` list contains shapely polygons of the grains detected in the image. ``labels`` is an image that contains the labels of the grains. 
``grain_data`` is a pandas dataframe with a number of grain parameters.

If you want to detect grains in large images, you should use the ``predict_large_image`` function, which will split the image into patches and run the Unet and SAM segmentations on each patch:

.. code-block:: python

   all_grains = seg.predict_large_image(fname, model, sam, min_area=400.0, patch_size=2000, overlap=200)

Just like before, the ``all_grains`` list contains shapely polygons of the grains detected in the image. The image containing the grain labels can be generated like this:

.. code-block:: python

   labels = seg.rasterize_grains(all_grains, large_image)

See the `Segment_every_grain.ipynb <https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain.ipynb>`_ notebook for an example 
of how the models can be loaded and used for segmenting an image and QC-ing the result. The notebook goes through the steps of loading the models, running the 
segmentation, interactively updating the result, and saving the grain data and the mask.

The last section of the `Segment_every_grain.ipynb` notebook shows how to finetune the U-Net model. The first step is to create patches (usually 256x256 pixels in size) from the images and the corresponding masks that you want to use for training.

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

Now we are ready to load the existing model weights and to train the model::

.. code-block:: python

   model = seg.create_and_train_model(train_dataset, val_dataset, test_dataset, model_file='seg_model.keras', epochs=100)