API Reference
=============
.. toctree::
   :caption: API Reference

.. function:: segmenteverygrain.predict_image(big_im, model, I)

   Segmantic segmentation of the entire image using a Unet model.

   :param big_im: The image that is being segmented. Can have one or more channels.
   :type big_im: 2D or 3D array
   :param model: Tensorflow model used for semantic segmentation.
   :param I: Size of the square-shaped image tiles in pixels.
   :type I: int
   :returns: Semantic segmentation result for the input image.
   :rtype: 3D array

.. autofunction:: segmenteverygrain.predict_large_image
.. autofunction:: segmenteverygrain.predict_image_tile
.. autofunction:: segmenteverygrain.label_grains
.. autofunction:: segmenteverygrain.one_point_prompt
.. autofunction:: segmenteverygrain.two_point_prompt
.. autofunction:: segmenteverygrain.find_overlapping_polygons
.. autofunction:: segmenteverygrain.weighted_crossentropy
.. autofunction:: segmenteverygrain.plot_images_and_labels
.. autofunction:: segmenteverygrain.calculate_iou
.. autofunction:: segmenteverygrain.pick_most_similar_polygon
.. autofunction:: segmenteverygrain.sam_segmentation
.. autofunction:: segmenteverygrain.find_connected_components
.. autofunction:: segmenteverygrain.merge_overlapping_polygons
.. autofunction:: segmenteverygrain.rasterize_grains
.. autofunction:: segmenteverygrain.create_labeled_image
.. autofunction:: segmenteverygrain.load_and_preprocess
.. autofunction:: segmenteverygrain.onclick
.. autofunction:: segmenteverygrain.onpress
.. autofunction:: segmenteverygrain.onclick2
.. autofunction:: segmenteverygrain.onpress2
.. autofunction:: segmenteverygrain.click_for_scale
.. autofunction:: segmenteverygrain.get_grains_from_patches
.. autofunction:: segmenteverygrain.plot_image_w_colorful_grains
.. autofunction:: segmenteverygrain.plot_grain_axes_and_centroids
.. autofunction:: segmenteverygrain.classify_points
.. autofunction:: segmenteverygrain.compute_curvature