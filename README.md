# segmenteverygrain

<img src="https://github.com/zsylvester/segmenteverygrain/blob/main/gravel_example_mask.png" width="600">

## Description

'segmenteverygrain' is a Python package that aims to detect grains (or grain-like objects) in images. The goal is to develop an ML model that does a reasonably good job at detecting most of the grains in a photo, so that it will be useful for determining grain size and grain shape, a common task in geomorphology and sedimentary geology. 'segmenteverygrain' relies on the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), developed by Meta, for getting high-quality outlines of the grains. However, SAM requires prompts for every object detected and, when used in 'everything' mode, it tends to be slow and results in many overlapping masks and non-grain (background) objects. To deal with these issues, 'segmenteverygrain' relies on a Unet-style, patch-based convolutional neural network to create a first-pass segmentation which is then used to generate prompts for the SAM-based segmentation. Some of the grains will be missed with this approach, but the segmentations that are created tend to be of high quality.

'segmenteverygrain' also includes a set of functions that make it possible to clean up the segmentation results: delete and merge objects by clicking on them, and adding grains that were not segmented automatically. The QC-d masks can be saved and added to a dataset of grain images. These images then can be used to improve the Unet model. Many of the images used in the dataset are from the [sedinet](https://github.com/DigitalGrainSize/SediNet) project.


## Requirements

- numpy
- matplotlib
- scipy
- pandas
- pillow
- scikit-image
- opencv-python
- networkx
- rasterio
- shapely
- tensorflow
- pytorch
- segment-anything
- tqdm

## Installation

'segmenteverygrain' is available through pypi.org, so you can install by running
<code>pip install segmenteverygrain</code>

The easiest way of creating a Python environment in which 'segmenteverygrain' works well is to use the ['environment.yml'](https://github.com/zsylvester/segmenteverygrain/blob/main/environment.yml) file with conda (or mamba).

## Getting started

See the [Segment_every_grain.ipynb](https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain.ipynb) notebook for an example of how the models can be loaded and used for segmenting an image and QC-ing the result. The notebook goes through the steps of loading the models, running the segmentation, interactively updating the result, and saving the grain data and the mask. The last section of the notebook illustrates the use of the 'predict_large_image' function that is recommended for large images (e.g., larger than 2000x3000 pixels). The images below illustrate how a relatively large thin-section image of a sandstone can be segmented using this approach.

<img src="https://github.com/zsylvester/segmenteverygrain/blob/main/miocene_sst_large_1.jpeg" width="600">

<img src="https://github.com/zsylvester/segmenteverygrain/blob/main/miocene_sst_large_2.jpeg" width="600">

The [Train_seg_unet_model.ipynb](https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Train_seg_unet_model.ipynb) notebook goes through the steps needed to create, train, and test the Unet model. If the base Unet model does not work well on a specific type of image, it is a good idea to generate some new training data (a few small images are usually enough) and to fine tune the base model so that it works better on the new image type. The workflow in the 'Train_seg_unet_model' notebook can be used to do this finetuning -- you just need to load the weights of the base model before starting the training.

The [Segment_every_grain_colab.ipynb](https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain_colab.ipynb) has been adjusted so that the segmentation can be tested in Google Colab. That said, the interactivity in Colab is not as smooth as in a local notebook.

## Acknowledgements

Thanks to Danny Stockli, Nick Howes, Kalinda Roberts, Jake Covault, Matt Malkowski, Raymond Luong, Wilson Bai, Rowan Martindale, and Sergey Fomel for discussions and/or helping with generating training data. Funding for this work came from the [Quantitative Clastics Laboratory industrial consortium](http://www.beg.utexas.edu/qcl) at the Bureau of Economic Geology, The University of Texas at Austin.

## License

segmenteverygrain is licensed under the [Apache License 2.0](https://github.com/zsylvester/segmenteverygrain/blob/master/LICENSE.txt).