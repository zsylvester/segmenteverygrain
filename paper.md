---
title: 'Segmenteverygrain: A Python module for segmentation of grains in a variety of image types'
tags:
  - Python
  - geology
  - geomorphology
  - sedimentology
  - petrography
authors:
  - name: Zoltan Sylvester
    orcid: 0000-0002-4890-4063
    corresponding: true 
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Bureau of Economic Geology, The University of Texas at Austin, USA
   index: 1
 - name: Department of Earth and Planetary Sciences, The University of Texas at Austin, USA 
   index: 2
date: 18 December 2024
bibliography: paper.bib
---

# Summary

`Segmenteverygrain` is a Python module that addresses the need for quantifying grain size in granular materials. It combines the Segment Anything Model (SAM) with a U-Net-style convolutional neural network to detect and measure grains or clasts in diverse image types, from microphotographs of sand and thin sections to images of gravel and boulder fields. Compared to similar tools such as GrainSight and ImageGrains, `Segmenteverygrain` offers greater flexibility and robustness, though it requires basic familiarity with Python and Jupyter notebooks. `Segmenteverygrain` supports segmentation of large images, georeferencing, and interactive editing of the results. The combined U-Net-SAM approach provides high-quality segmentation results, allows fine-tuning with minimal training data, and can be extended to images with other object types.

# Statement of need

Grain size and shape are key factors that influence the physical and chemical properties of granular materials. Quantitative estimates of grain size and shape are important in a broad range of fields, including: 

* geomorphology, sedimentology, and stratigraphy (sediment transport and morphodynamics);
* subsurface reservoir engineering (hydrocarbon production, aquifer management, CO2 storage);
* civil and geotechnical engineering (foundation design, erosion control, sedimentation);
* environmental science (pollutant transport);
* materials science (gravel and sand for construction; manufacturing of glass, concrete, and asphalt).

While the grain size of loose sand and finer-than-sand sediment can be measured accurately using laser particle size analyzers [@Blott:2004], quantifying the grain size of cemented rocks or of sediment that is coarser than medium sand commonly requires manual measurement of a large number of grain lengths. This is increasingly done in digital images, often in a grain-by-grain fashion. In recent years, several studies illustrated the promise of machine learning (ML) approaches [@Buscombe:2020, @Tang:2020, @Mair:2022, @Prieur:2023, @Mair:2024, @Azzam:2024]. While these studies clearly show that ML techniques are superior to both manual data collection and conventional image processing techniques [e.g., @Purinton:2021], they tend to focus on a relatively narrow range of image types, e.g., gravel on fluvial bars [@Mair:2022, @Mair:2024], boulder fields on planetary surfaces [@Prieur:2023, @Robin:2024], or petrographic images [Tang:2020, @Azzam:2024].

With the emergence of large image segmentation models that have been trained on millions of images [e.g., @Kirillov:2023, @Ravi:2024], the opportunity arises to use these pre-trained models to detect a wide variety of grains and grain-like objects in a broad range of image types. `Segmenteverygrain` is a Python package that takes advantage of the Segment Anything Model (SAM) [@Kirillov:2023] to generate accurate segmentation masks for grains in any image, as long as the grains are relatively well defined in the image. To fully automate the process, `Segmenteverygrain` uses a U-Net-style convolutional neural network to create prompts (pixel coordinates of grain centers) for SAM; and ensures that the resulting masks contain no duplicates and do not overlap each other. In general, SAM masks are more robust and more accurate than the U-Net output (Figure 1). The U-Net model has been trained on 66 images of a variety of grains that were split into 44,533 patches of 256x256 pixels. Some of these images were labeled by manually outlining every grain; for others, we used SAM in interactive mode: clicking on each object to provide a prompt. Segmenting every grain eliminates the problem of selecting a representative sample of grains - as long as the image itself is representative.

![Photo of fluvial gravel (A), output of the U-Net segmentation (B), and the result of the SAM segmentation (C). Grains 1-5 are all detected correctly, despite the fact that the grain boundary class in the U-Net segmentation does not fully delineate them. Patch #6 in the final segmentation result is a false positive that can be deleted manually. \label{fig:U-Net_sam_comparison}](joss_paper_fig_1.jpg)

So far, `Segmenteverygrain` has been successfully used on:
* microphotographs (taken in reflected light) of sand and detrital zircon grains 
* microphotographs (taken in transmitted light) of thin sections of sandstones and oolitic limestones (Figure 2)
* photographs of gravel and cobbles on beaches [@Roberts:2024] and on fluvial bars (Figure 3)
* images of boulder fields on asteroids [@Robin:2024].

![Photo of fluvial gravel (A), output of the `Segmenteverygrain` segmentation (B), size distributions of the major and minor grain axes (C), and the area-weighted size distributions (D). \label{fig:gravel_segmentation}](joss_paper_fig_2.jpg)

![Photo of a sandstone in thin section (A), output of the `Segmenteverygrain` segmentation (B), size distributions of the major and minor grain axes (C), and the area-weighted size distributions (D). Photo from [@Prodanovic:2019] \label{fig:sandstone_segmentation}](joss_paper_fig_3.jpg)

In addition to grain segmentation, the combined U-Net-SAM approach used in `Segmenteverygrain` has the potential to provide high-quality segmentation results when working with other types of objects as well.

Although the approach developed by [@Azzam:2024] for petrographic thin sections has the potential to be applied to other types as images as well, the philosophy of `GrainSight` is different from that of `Segmenteverygrain`: it provides a simple no-code user interface that requires no coding experience. In contrast, `Segmenteverygrain` assumes that the user is familiar with basic Python and ML concepts and is comfortable with running Jupyter notebooks. Although this approach reduces the potential user base, it makes it relatively easy to modify the code and train the model. Even when the base model does not work well on a new image type, it is straightforward to generate some training data and fine tune the U-Net model to improve the SAM output.

`ImageGrains` [@Mair:2022, @Mair:2024] is another alternative to the approach described here. `ImageGrains` is based on `Cellpose`, a U-Net-style convolutional neural network [@Ronneberger:2015] trained for segmenting cells and cell nuclei in biomedical images [@Stringer:2021]. This CNN model has been further trained on images of coarse-grained fluvial sediment [@Mair:2024], thus it is specific to this type of imagery. `segmenteverygrain` has been trained a wide variety of images of grains and clasts and thus is more generic; using SAM on the ouput of the `Segmenteverygrain` U-Net model also results in additional robustness and generality. `Segmenteverygrain` also adds functionality for segmenting large images, working with georeferenced data, and interactively deleting, merging, and adding grains to the segmentation result.

# Key functionality

The base U-Net model that is available in the `segmenteverygrain` repository works relatively well on variety of image types (e.g., thin sections and reflected-light microphotographs of sand, gravel and coarser clasts). However, it is recommended that it is first tested on a small image (e.g., 2000 x 300 pixels), to see how well the U-Net model captures the difference between the grains and the background: 

```
image_pred = seg.predict_image(image, model, I=256) # U-Net prediction
labels, coords = seg.label_grains(image, image_pred, dbs_max_dist=20.0) # create grain labels and SAM prompts
```

The second line of the code snippet above creates the grain labels and the SAM prompts (pixel coordinates of points that are clustered near the grain centers). The SAM segmentation is done as follows:

```
all_grains, labels, mask_all, grain_data, fig, ax = seg.sam_segmentation(sam, image, image_pred, 
        coords, labels, min_area=400.0, plot_image=True, remove_edge_grains=False, remove_large_objects=False)
```

Looking et the U-Net output `image_pred`, if it is obvious that there is room for improvement, a few small images should be used to derive training data for fine tuning the U-Net model. This can be done either by first running the segmentation workflow and then cleaning up the result, or, if the output is of low quality, by using SAM to do the segmentation. `segmenteverygrain` makes this process straightforward as it has tools for interactively deleting, merging, and adding grains.

Next, the training data (pairs of images and segmentation masks) can be used to fine tune the U-Net model. The `patchify_training_data` function creates patches of 256 x 256 pixels, which are split into training, validation, and test sets using the `create_train_val_test_data` function. Data augmentation is applied during training (`load_and_preprocess` function). The `create_and_train_model` function loads the weights of the base model, runs the training, and evaluates the new model.

Once the `segmenteverygrain` workflow has been tested on smaller images, the `predict_large_image` function can be used to run the segmentation of large images that contain thousands or tens of thousands of grains:

```
all_grains, image_pred = seg.predict_large_image(fname, model, sam, min_area=400.0, patch_size=2000, overlap=200)
```

This is done by running the U-Net + SAM predictions on smaller tiles of the input image (default size of 2000 x 2000 pixels), and collecting the grains into a list without duplicates.

After all the grains have been outlined in the image, a pandas dataframe can be created with grain area, centroid coordinates, major and minor axis lengths, and a number of other grain features. True lengths and areas can be somputed using a `units_per_pixel` variable that, if not availbale, can be determined using a scale bar in the image. Running the `plot_histogram_of_axis_lengths` function creates a plot with the distributions of major- and minor grain axis lengths plotted, both as histograms and as empirical cumulative distribution functions. If the grain areas are provided as well as input to the function, the distributions will be weighted by grain areas, so that they are more consistent with grain size distributions that come from sieving, point counting, or Wolman counts [@Taylor:2020].

The `Segment_every_grain_w_georeferencing.ipynb` notebook demonstrates how one can run `Segmenteverygrain` on a georeferenced image and write out the results as a set of grain polygons in shapefile format.

# Dependencies and availability

The `Segmenteverygrain` package is available from PyPI at <https://pypi.org/project/segmenteverygrain/>. Because it relies on the manipulation of both raster and vector data, the dependencies include a number of image processing and shape manipulation tools, such as `Pillow` [@Clark:2015], `scikit-image` [@Van_der_Walt:2014], `rasterio` [@Gillies:2019], and `shapely` [@Gillies:2007]. To identify and manipulate overlapping polygons that result from the SAM segmentation, we rely on the `networkx` package [@Hagberg:2008]. The U-Net model is built and trained using `tensorflow` [@Abadi:2015] and `keras` [@Chollet:2015]; some parts of the machine learning workflow are handled with functions from the `scikit-learn` library [@Pedregosa:2011].

# Acknowledgements

Funding for this work came from the Quantitative Clastics Laboratory industrial consortium at the Bureau of Economic Geology, The University of Texas at Austin.

# References