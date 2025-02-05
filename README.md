# segmenteverygrain

<img src="https://github.com/zsylvester/segmenteverygrain/blob/main/gravel_example_mask.png" width="600">

[![Tests](https://github.com/zsylvester/segmenteverygrain/actions/workflows/ci.yaml/badge.svg)](https://github.com/zsylvester/segmenteverygrain/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


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
- rtree
- tqdm

## Documentation

More documentation is available at [https://zsylvester.github.io/segmenteverygrain/index.html](https://zsylvester.github.io/segmenteverygrain/index.html).

## Installation

'segmenteverygrain' is available through pypi.org, so you can install it by running:
```
pip install segmenteverygrain
```

The easiest way of creating a Python environment in which 'segmenteverygrain' works well is to use the ['environment.yml'](https://github.com/zsylvester/segmenteverygrain/blob/main/environment.yml) file with conda (or mamba).

If you are starting from scratch (no software on your compute for Python package management and no git installed), here are some more detailed instructions to follow:

First, download Anaconda at [https:www.anaconda.com/download](https:www.anaconda.com/download). \[This will install the Anaconda Distribution. Alternatively, if you are comfortable with the command line, you can [rely only on conda or mamba](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). Currently [miniforge](https://conda-forge.org/download) might be the best option as it allows you to rely on the `mamba` package solver, which is faster than `conda`.\]

In Anaconda Prompt (Windows), or Terminal (Mac), enter the following to install pip and git packages:

```
conda install pip git
```

Download the segmenteverygrain files.

Windows:
```
git clone --depth 1 https://github.com/zsylvester/segmenteverygrain.git
```
Linux/Mac:
```
git clone --depth 1 git@github.com:zsylvester/segmenteverygrain.git
```

Set up the `segmenteverygrain` environment with conda (Windows):
```
conda env create -f segmenteverygrain/environment.yml
```

Set up the `segmenteverygrain` environment with conda (Mac):
```
conda env create -f segmenteverygrain/environment_macos.yml
```

Activate environment:
```
conda activate segmenteverygrain
```

## Getting started

See the [Segment_every_grain.ipynb](https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain.ipynb) notebook for an example of how the models can be loaded and used for segmenting an image and QC-ing the result. The notebook goes through the steps of loading the models, running the segmentation, interactively updating the result, and saving the grain data and the mask. The last section of the notebook illustrates the use of the 'predict_large_image' function that is recommended for large images (e.g., larger than 2000x3000 pixels). The images below illustrate how a relatively large thin-section image of a sandstone can be segmented using this approach. Image from [Digital Rocks Portal](https://www.digitalrocksportal.org/projects/244).

<img src="https://github.com/zsylvester/segmenteverygrain/blob/main/miocene_sst_large_1.jpeg" width="600">

<img src="https://github.com/zsylvester/segmenteverygrain/blob/main/miocene_sst_large_2.jpeg" width="600">

 If the base Unet model does not work well on a specific type of image, it is a good idea to generate some new training data (a few small images are usually enough) and to fine tune the base model so that it works better on the new image type. This can be done by running the cells in the last section ('Finetuning the base model') of the [Segment_every_grain.ipynb](https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain.ipynb) notebook.

The [Segment_every_grain_colab.ipynb](https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain_colab.ipynb) has been adjusted so that the segmentation can be tested in Google Colab. That said, the interactivity in Colab is not as smooth as in a local notebook.

## Acknowledgements

Thanks to Danny Stockli, Nick Howes, Kalinda Roberts, Jake Covault, Matt Malkowski, Raymond Luong, Wilson Bai, Rowan Martindale, and Sergey Fomel for discussions and/or helping with generating training data. Funding for this work came from the [Quantitative Clastics Laboratory industrial consortium](http://www.beg.utexas.edu/qcl) at the Bureau of Economic Geology, The University of Texas at Austin.

## License

`segmenteverygrain` is licensed under the [Apache License 2.0](https://github.com/zsylvester/segmenteverygrain/blob/master/LICENSE.txt).
