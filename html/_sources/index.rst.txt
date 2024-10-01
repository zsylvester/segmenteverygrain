.. figure:: ../../gravel_example_mask.png
   :alt: grains detected in an image
   :align: left

segmenteverygrain
=================
`GitHub Repository <https://github.com/zsylvester/segmenteverygrain>`_

``segmenteverygrain`` is a Python package that aims to detect grains (or grain-like objects) in images. 
The goal is to develop a machine learning model that does a reasonably good job at detecting most of the grains in a photo, so that it is 
useful for determining grain size and grain shape, a common task in geomorphology and sedimentary geology. ``segmenteverygrain`` 
relies on the `Segment Anything Model (SAM) <https://github.com/facebookresearch/segment-anything>`_, developed by Meta, 
for getting high-quality outlines of the grains. However, SAM requires prompts for every object detected and, when used in 
'everything' mode, it tends to be slow and results in many overlapping masks and non-grain (background) objects. 
To deal with these issues, 'segmenteverygrain' relies on a Unet-style, patch-based convolutional neural network to create a 
first-pass segmentation which is then used to generate prompts for the SAM-based segmentation. Some of the grains will be missed 
with this approach, but the segmentations that are created tend to be of high quality.

``segmenteverygrain`` also includes a set of functions that make it possible to clean up the segmentation results: delete and 
merge objects by clicking on them, and adding grains that were not segmented automatically. The QC-d masks can be saved and 
added to a dataset of grain images. These images then can be used to improve the Unet model.

Installation
------------
.. toctree::
   :caption: Installation

To install ``segmenteverygrain`` you can use ``pip``:

.. code-block:: shell

   pip install segmenteverygrain

Or you can install it from the source code:

.. code-block:: shell

   git clone https://github.com/zsylvester/segmenteverygrain.git
   cd segmenteverygrain
   pip install .

The easiest way of creating a Python environment in which 'segmenteverygrain' works well is to use 
the `environment.yml <https://github.com/zsylvester/segmenteverygrain/blob/main/environment.yml>`_ file with conda (or mamba).

Contents
--------
.. toctree::
   :maxdepth: 2

   getting_started
   api_reference
