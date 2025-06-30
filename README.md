# segmenteverygrain

<img src="docs/gravel_example_mask.png" width="600">

[![Tests](https://github.com/zsylvester/segmenteverygrain/actions/workflows/ci.yaml/badge.svg)](https://github.com/zsylvester/segmenteverygrain/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![status](https://joss.theoj.org/papers/3adb1bac50434eb701915a59d65eba40/status.svg)](https://joss.theoj.org/papers/3adb1bac50434eb701915a59d65eba40)


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
If you are using 'pip', you need to make sure that the  Python version is 3.9 (and not higher), so that all dependencies work correctly.

Note that you need to clone the repository to get the model files in one go; otherwise you need to download the SAM and U-Net models manually and place them in the right folders.

The easiest way of creating a Python environment in which 'segmenteverygrain' works well is to use the ['environment.yml'](environment.yml) file with conda (or mamba).

We recommend that you install [conda or mamba](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) to manage your Python environments. Due to licensing restrictions on Aanconda, [miniforge](https://conda-forge.org/download) might be the best option as there are no strings attached and it allows you to rely on the `mamba` package solver, which is faster than `conda`. If you are using `mamba`, you can simply replace `conda` with `mamba` in the commands below.

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

See the [Segment_every_grain.ipynb](Segment_every_grain.ipynb) notebook for an example of how the models can be loaded and used for segmenting an image and QC-ing the result. The notebook goes through the steps of loading the models, running the segmentation, interactively updating the result, and saving the grain data and the mask. The last section of the notebook illustrates the use of the 'predict_large_image' function that is recommended for large images (e.g., larger than 2000x3000 pixels). The images below illustrate how a relatively large thin-section image of a sandstone can be segmented using this approach. Image from [Digital Rocks Portal](https://www.digitalrocksportal.org/projects/244).

<img src="docs/miocene_sst_large_1.jpeg" width="600">

<img src="docs/miocene_sst_large_2.jpeg" width="600">

 If the base Unet model does not work well on a specific type of image, it is a good idea to generate some new training data (a few small images are usually enough) and to fine tune the base model so that it works better on the new image type. This can be done by running the cells in the last section ('Finetuning the base model') of the [Segment_every_grain.ipynb](Segment_every_grain.ipynb) notebook.

The [Segment_every_grain_colab.ipynb](Segment_every_grain_colab.ipynb) has been adjusted so that the segmentation can be tested in Google Colab. That said, the interactivity in Colab is not as smooth as in a local notebook.

## Running times

It takes 2 minutes and 40 seconds to run the full segmentation on a 3 megapixel (e.g., 1500x2000 pixels) image, on an Apple M2 Max laptop with 96 GB RAM. The same image takes the same amount of time to segment using Google Colab with a Nvidia A100 GPU.

Obviously, large images take longer to process. The segmentation of the ~20 megapixel example image that is provided in the repository ('mair_et_al_L2_DJI_0382_image.jpg') takes ~20 minutes with both hardware configurations mentioned before. As the processing of large images is done in patches, the increase in computational time is roughly linear.

## Contributing

We welcome contributions from anyone interested in improving the project. To contribute to the model use the following steps:

1. Fork the repository.
2. Create a new branch for your changes:

```bash
git checkout -b feature/my-feature
```

3. Make your changes and commit them:

```bash
git add .
git commit -m "Add my feature"
```

4. Push your changes to your forked repository:

```bash
git push origin feature/my-feature
```

5. Create a pull request from your forked repository back to the original repository.

## Reporting Issues

If you encounter any issues or problems while using segmentanygrain, we encourage you to report them to us. This helps us identify and address any bugs or areas for improvement.

To report an issue, please follow these steps:

1. **Check the Existing Issues:** Before submitting a new issue, search our issue tracker to see if the problem you're experiencing has already been reported. If you find a similar issue, you can add any additional information or comments to that existing issue.
2. **Create a New Issue:** If you don't find an existing issue that matches your problem, create a new issue by clicking the "New issue" button on the issues page. Provide a clear and descriptive title for your issue, and include the following information in the description:
    - A detailed description of the problem you're experiencing, including any error messages or unexpected behavior.
    - The steps to reproduce the issue, if possible.
    - Your operating system and the version of the software you're using.
    - Any relevant logs or screenshots that could help us understand the problem.
3. **Submit the Issue**: Once you've provided all the necessary information, click the "Submit new issue" button to create the issue. Our team will review the issue and respond as soon as possible.

We appreciate you taking the time to report any issues you encounter. Your feedback helps us improve.

## Acknowledgements

Thanks to Danny Stockli, Nick Howes, Kalinda Roberts, Jake Covault, Matt Malkowski, Raymond Luong, Wilson Bai, Rowan Martindale, and Sergey Fomel for discussions and/or helping with generating training data. Funding for this work came from the [Quantitative Clastics Laboratory industrial consortium](http://www.beg.utexas.edu/qcl) at the Bureau of Economic Geology, The University of Texas at Austin.

## License

`segmenteverygrain` is licensed under the [Apache License 2.0](https://github.com/zsylvester/segmenteverygrain/blob/master/LICENSE.txt).
