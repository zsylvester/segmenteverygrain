API Reference
=============
.. toctree::
   :caption: API Reference

Interactive Editing (interactions module)
-----------------------------------------

The ``interactions`` module provides an interactive interface for editing grain segmentations.

Classes
~~~~~~~

.. autoclass:: segmenteverygrain.GrainPlot
   :members:
   :undoc-members:

.. autoclass:: segmenteverygrain.Grain
   :members:
   :undoc-members:

Functions
~~~~~~~~~

.. autofunction:: segmenteverygrain.load_image
.. autofunction:: segmenteverygrain.polygons_to_grains
.. autofunction:: segmenteverygrain.save_grains
.. autofunction:: segmenteverygrain.get_summary
.. autofunction:: segmenteverygrain.save_summary
.. autofunction:: segmenteverygrain.save_histogram
.. autofunction:: segmenteverygrain.save_mask


Grain Extraction and Clustering (grain_utils module)
----------------------------------------------------

The ``grain_utils`` module provides functions for extracting individual grain images, extracting features, and clustering grains.

.. autofunction:: segmenteverygrain.extract_grain_image
.. autofunction:: segmenteverygrain.extract_all_grains
.. autofunction:: segmenteverygrain.make_square
.. autofunction:: segmenteverygrain.extract_vgg16_features
.. autofunction:: segmenteverygrain.extract_color_features
.. autofunction:: segmenteverygrain.cluster_grains
.. autofunction:: segmenteverygrain.create_grain_panel
.. autofunction:: segmenteverygrain.create_grain_panel_cluster
.. autofunction:: segmenteverygrain.create_clustered_grain_montage
.. autofunction:: segmenteverygrain.plot_classified_grains

.. autoclass:: segmenteverygrain.ClusterMontageSelector
   :members:
   :undoc-members:

.. autoclass:: segmenteverygrain.ClusterMontageLabeler
   :members:
   :undoc-members:


Core Segmentation Functions
---------------------------

Model Prediction
~~~~~~~~~~~~~~~~

.. autofunction:: segmenteverygrain.predict_image_tile
.. autofunction:: segmenteverygrain.predict_image
.. autofunction:: segmenteverygrain.predict_large_image
.. autofunction:: segmenteverygrain.label_grains
.. autofunction:: segmenteverygrain.sam_segmentation
.. autofunction:: segmenteverygrain.one_point_prompt
.. autofunction:: segmenteverygrain.two_point_prompt

Polygon Operations
~~~~~~~~~~~~~~~~~~

.. autofunction:: segmenteverygrain.find_overlapping_polygons
.. autofunction:: segmenteverygrain.find_connected_components
.. autofunction:: segmenteverygrain.merge_overlapping_polygons
.. autofunction:: segmenteverygrain.collect_polygon_from_mask
.. autofunction:: segmenteverygrain.pick_most_similar_polygon
.. autofunction:: segmenteverygrain.calculate_iou
.. autofunction:: segmenteverygrain.rasterize_grains
.. autofunction:: segmenteverygrain.create_labeled_image
.. autofunction:: segmenteverygrain.read_polygons
.. autofunction:: segmenteverygrain.save_polygons

Visualization
~~~~~~~~~~~~~

.. autofunction:: segmenteverygrain.plot_image_w_colorful_grains
.. autofunction:: segmenteverygrain.plot_grain_axes_and_centroids
.. autofunction:: segmenteverygrain.plot_histogram_of_axis_lengths
.. autofunction:: segmenteverygrain.plot_images_and_labels

Model Training
~~~~~~~~~~~~~~

.. autofunction:: segmenteverygrain.Unet
.. autofunction:: segmenteverygrain.weighted_crossentropy
.. autofunction:: segmenteverygrain.patchify_training_data
.. autofunction:: segmenteverygrain.create_train_val_test_data
.. autofunction:: segmenteverygrain.create_and_train_model
.. autofunction:: segmenteverygrain.load_and_preprocess

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: segmenteverygrain.extract_patch
.. autofunction:: segmenteverygrain.convert_to_large_image_coords
.. autofunction:: segmenteverygrain.classify_points
.. autofunction:: segmenteverygrain.compute_curvature
.. autofunction:: segmenteverygrain.find_grain_size_classes
.. autofunction:: segmenteverygrain.get_area_weighted_distribution
.. autofunction:: segmenteverygrain.click_for_scale
.. autofunction:: segmenteverygrain.get_grains_from_patches

Legacy Interactive Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions are from the original interactive editing interface. The new ``GrainPlot`` class from the ``interactions`` module is now the recommended approach.

.. autofunction:: segmenteverygrain.onclick
.. autofunction:: segmenteverygrain.onclick2
.. autofunction:: segmenteverygrain.onclick_large_image
.. autofunction:: segmenteverygrain.onpress
.. autofunction:: segmenteverygrain.onpress2