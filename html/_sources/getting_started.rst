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

The `Train_seg_unet_model.ipynb <https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Train_seg_unet_model.ipynb>`_ notebook goes through the 
steps needed to create, train, and test the Unet model. If the base Unet model does not work well on a specific type of image, it is a good idea to generate some 
new training data (a few small images are usually enough) and to fine tune the base model so that it works better on the new image type. The workflow in the 
'Train_seg_unet_model' notebook can be used to do this finetuning -- you just need to load the weights of the base model before starting the training.