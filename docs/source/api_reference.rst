API Reference
=============
.. toctree::
   :caption: API Reference

.. function:: segmenteverygrain.predict_image(big_im, model, I)

   Semantic segmentation of the entire image using a Unet model.

   :param big_im: The image that is being segmented. Can have one or more channels.
   :type big_im: 2D or 3D array
   :param model: Tensorflow model used for semantic segmentation.
   :param I: Size of the square-shaped image tiles in pixels.
   :type I: int
   :returns: Semantic segmentation result for the input image.
   :rtype: 3D array