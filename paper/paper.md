---
title: 'Segmenteverygrain: A Python module for segmentation of grains in a variety of image types'
tags:
  - Python
  - geology
  - geomorphology
  - sedimentology
  - petrography
authors:
  - name: Zoltán Sylvester
    orcid: 0000-0002-4890-4063
    corresponding: true 
    affiliation: "1, 2"
  - name: Daniel F. Stockli
    orcid: 0000-0001-7652-2129
    affiliation: 2
  - name: Nick Howes
    affiliation: 3
  - name: Kalinda Roberts
    affiliation: 4
  - name: Matthew A. Malkowski
    orcid: 0000-0002-7781-6249
    affiliation: 2
  - name: Zsófia Poros
    affiliation: 5
  - name: Rowan C. Martindale
    orcid: 0000-0003-2681-083X
    affiliation: 2
  - name: Wilson Bai
    affiliation: 2
affiliations:
  - name: Bureau of Economic Geology, Jackson School of Geosciences, The University of Texas at Austin, Austin, TX, USA
    index: 1
  - name: Department of Earth and Planetary Sciences, Jackson School of Geosciences, The University of Texas at Austin, Austin, TX, USA 
    index: 2
  - name: The Water Institute, New Orleans, LA, USA
    index: 3
  - name: BSC Group, Boston, MA, USA
    index: 4
  - name: ConocoPhillips, Houston, TX, USA
    index: 5

date: 5 February 2025
bibliography: paper.bib
---

# Summary

`Segmenteverygrain` is a Python module that addresses the need for quantifying grain size in granular materials. It combines the `Segment Anything Model` (SAM) with a U-Net-style convolutional neural network to detect and measure grains or clasts in diverse image types, ranging from photomicrographs of sand, mineral grains, and thin sections to photographic images of gravel and boulder fields. Compared to similar tools such as `GrainSight` [@Azzam:2024] and `ImageGrains` [@Mair:2022; @Mair:2024], `Segmenteverygrain` offers greater flexibility and robustness, though it requires basic familiarity with Python and Jupyter notebooks. `Segmenteverygrain` supports segmentation of large images, georeferencing, and interactive editing of the results. The combined U-Net-SAM approach provides high-quality segmentation results, allows for fine-tuning with minimal training data, and can be expanded to images with other object types.

# Statement of need

Grain morphology, in particular size and shape, are key parameters that influence the physical and chemical properties of granular materials. Quantitative estimates of these parameters  are important in a broad range of fields, including:

* geomorphology, sedimentology, stratigraphy, and paleontology (sediment transport and morphodynamics);
* subsurface reservoir quality and engineering (compaction and cementation, hydrocarbon production, aquifer management, CO2 storage);
* structural geology, petrology and geochemistry (grain-size sensitive rheology, nucleation and cooling rates of melts, chemical reactiveness and alteration);
* civil and geotechnical engineering (foundation design, erosion control, sedimentation);
* environmental science (pollutant transport, absorption potential, biochemical rock-water interaction);
* materials science (gravel and sand for construction; manufacturing of glass, concrete, and asphalt).

While the grain size of loose sand and finer-than-sand sediment can be measured accurately on bulk sample using laser particle size analyzers [@Blott:2004], quantifying the grain size of small samples, cemented rock or of sediment that is coarser than medium sand commonly requires manual measurement of hundreds of individual grain lengths and widths. This is increasingly done via the analysis of digital images, often in a grain-by-grain fashion. In recent years, however, numerous studies have illustrated the promise of automated image analysis and/or machine learning (ML) approaches [@Buscombe:2020; @Tang:2020; @Mair:2022; @Chen:2023; @Prieur:2023; @Mair:2024; @Azzam:2024]. While these studies clearly show that ML techniques are superior to both painstaking manual data collection and conventional image processing techniques [e.g., @Purinton:2021], they tend to focus on a relatively narrow range of image types, e.g., gravel on fluvial bars [@Mair:2022; @Mair:2024], boulder fields on planetary surfaces [@Prieur:2023; @Robin:2024], or petrographic images [Tang:2020; @Azzam:2024].

With the emergence of large image segmentation models that have been trained on millions of images [e.g., @Kirillov:2023; @Ravi:2024], the opportunity arises for the utilization of these pre-trained models to detect a wide variety of grains and grain-like objects in a broad range of image types. `Segmenteverygrain` is a Python package that takes advantage of the Segment Anything Model (SAM) [@Kirillov:2023] to generate accurate segmentation masks for grains in any image, as long as the grains are relatively well defined in the image. To fully automate this process, `Segmenteverygrain` uses a U-Net-style convolutional neural network to create prompts (pixel coordinates of grain centers) for SAM; and ensures that the resulting masks contain no duplicates and  do not overlap. In general, SAM masks are more robust and accurate than the U-Net output (\autoref{fig:1}). The U-Net model uses patches as input and output; to reduce edge effects, Hann-window-based weighting is used on overlapping patches [@Pielawski:2020]. The U-Net model has been trained on 66 images of a variety of grains that were split into 44,533 patches of 256x256 pixels. Some of these images were labeled by manually outlining individual grains; for others, we used SAM in interactive mode by clicking on each object to provide a prompt. Segmenting every grain eliminates the problem of selecting a representative sample of grains - as long as the image itself is representative.

![Photo of fluvial gravel (A), output of the U-Net segmentation (B), and the result of the SAM segmentation (C). Grains 1-5 are all detected correctly, despite the fact that the grain boundary class in the U-Net segmentation does not fully delineate them. Patch #6 in the final segmentation result is a false positive that can be deleted manually. \label{fig:U-Net_sam_comparison}\label{fig:1}](joss_paper_fig_1.jpg)

So far, `Segmenteverygrain` has been successfully used on:

* images of boulder fields on asteroids [@Robin:2024];
* photographs of gravel and cobbles on beaches [@Roberts:2024] and on fluvial bars (\autoref{fig:2});  
* photomicrographs (taken in transmitted light) of thin sections of sandstones (\autoref{fig:3}) and oolitic limestones;  
* photomicrographs (taken in reflected light) of sand and detrital zircon grains.

![Photo of fluvial gravel (A), output of the \`Segmenteverygrain\` segmentation (B), size distributions of the major and minor grain axes (C), and the area-weighted size distributions (D). Major grain axis lengths are shown in blue, minor grain axis lengths in orange.\label{fig:2}](joss_paper_fig_2.jpg)

In addition to grain segmentation, the combined U-Net-SAM approach used in `Segmenteverygrain` has the potential to provide high-quality segmentation results when working with other types of objects as well.

Although the approach developed by [@Azzam:2024] for petrographic thin sections has the potential to be applied to other types as images as well, the philosophy of `GrainSight` is different from that of `Segmenteverygrain` as it provides a simple no-code user interface that requires no coding experience. In contrast, `Segmenteverygrain` assumes that the user is familiar with basic Python and ML concepts and is comfortable with running Jupyter notebooks. Although this approach might reduce the potential user base, it allows its users to relatively easily modify the code and train the model. Even when the base model does not work well on a new image type, it is straightforward to generate new training data and fine tune the U-Net model to improve the SAM output.  
 
![Photomicrograph (plane polarized light) of a sandstone in thin section (A), output of the \`Segmenteverygrain\` segmentation (B), size distributions of the major and minor grain axes (C), and the area-weighted size distributions (D). Photomicrograph from \[@Prodanovic:2019\]. Major grain axis lengths are shown in blue, minor grain axis lengths in orange.\label{fig:3}](joss_paper_fig_3.jpg)

`ImageGrains` [@Mair:2022; @Mair:2024] is another alternative to the approach described here. `ImageGrains` is based on `Cellpose`, a U-Net-style convolutional neural network [@Ronneberger:2015] trained for segmenting cells and cell nuclei in biomedical images [@Stringer:2021]. This CNN model has been further trained on images of coarse-grained fluvial sediment [@Mair:2024] and is thus specific to this type of clast imagery. In contrast, `Segmenteverygrain` has been trained on a wide variety of images of grains and clasts and thus is more generic; using SAM on the output of the `Segmenteverygrain` U-Net model also results in additional robustness and generality. `Segmenteverygrain` also adds functionality for segmenting large images (with a size limited by computer memory), working with georeferenced data, and interactively deleting, merging, and adding grains to the segmentation result.

# Key functionality

The base U-Net model that is available in the `Segmenteverygrain` repository works relatively well on a variety of image types (e.g., thin sections and reflected-light photomicrographs of mineral grains, sand, gravel and coarser clasts). However, it is recommended that it is first tested on a small image (e.g., 2000 x 300 pixels), to see how well the U-Net model captures the difference between the grains and the background:

```
image_pred = seg.predict_image(image, model, I=256) # U-Net prediction
labels, coords = seg.label_grains(image, image_pred, dbs_max_dist=20.0)
```

The second line of the code snippet above creates the grain labels and the SAM prompts (pixel coordinates of points that are clustered near the grain centers). The SAM segmentation is done as follows:

```
all_grains, labels, mask_all, grain_data, fig, ax = seg.sam_segmentation(sam, 
    image, image_pred, coords, labels, min_area=400.0, plot_image=True, 
    remove_edge_grains=False, remove_large_objects=False)
```

Looking at the U-Net output `image_pred`, if it is obvious that there is room for improvement, a few small images should be used to derive training data for fine tuning the U-Net model. This can be done either by first running the segmentation workflow and then cleaning up the result, or, if the output is of low quality, by using SAM to do the segmentation. `Segmenteverygrain` makes this process straightforward as it has tools for interactively deleting, merging, and adding grains.

Next, the training data (pairs of images and segmentation masks) can be used to fine tune the U-Net model. The `patchify_training_data` function creates patches of 256 x 256 pixels, which are split into training, validation, and test sets using the `create_train_val_test_data` function. Data augmentation is applied during training (`load_and_preprocess` function). The `create_and_train_model` function loads the weights of the base model, runs the training, and evaluates the new model.

Once the `Segmenteverygrain` workflow has been tested on smaller images, the `predict_large_image` function can be used to run the segmentation of larger images that contain thousands or tens of thousands of grains:

```
all_grains, image_pred = seg.predict_large_image(fname, model, sam, 
    min_area=400.0, patch_size=2000, overlap=200)
```

This is done by running the U-Net + SAM predictions on smaller tiles of the input image (default size of 2000 x 2000 pixels), and collecting the grains into a list without duplicates.

After all the grains (or clasts) have been outlined in the image, a pandas dataframe can be created with grain area, centroid coordinates, major and minor axis lengths, and a number of other grain features. True lengths and widths and areas can be computed using a `units_per_pixel` variable that, if not available, can be determined using a scale bar in the image. Running the `plot_histogram_of_axis_lengths` function creates a plot with the distributions of major- and minor grain axis lengths plotted, both as histograms and as empirical cumulative distribution functions. If the grain areas are provided as well as input to the function, the distributions will be weighted by grain areas, so that they are more consistent with grain size distributions that come from sieving, point counting, or Wolman counts [@Taylor:2022].

The `Segment_every_grain_w_georeferencing.ipynb` notebook demonstrates how one can run `Segmenteverygrain` on a georeferenced image and write out the results as a set of grain polygons in shapefile format. This feature enables detailed geospatial analyses of the coarse material distributions, capturing variations in grain size across surfaces (\autoref{fig:4}). When applied to georeferenced orthomosaics, such as those generated from photogrammetric techniques like structure-from-motion (SfM), the model enables precise assessments of granulometric change over time. By integrating these results with elevation data from digital surface models or other topographic datasets, the model enables examinations of relationships between sediment dynamics and broader geomorphic changes.
 
![A clip of orthoimagery from ground-based structure-from motion survey of mixed sand and gravel beach (A). Clip of orthoimagery overlain with segmented grains output from processing the imagery through the `Segmenteverygrain` model, with grains colored by Wentworth size classes (B).\label{fig:4}](joss_paper_fig_4.jpg)

# Dependencies and availability

The `Segmenteverygrain` package is available from PyPI at [https://pypi.org/project/segmenteverygrain/](https://pypi.org/project/segmenteverygrain/). Because it relies on the manipulation of both raster and vector data, the dependencies include a number of image processing and shape manipulation tools, such as `Pillow` [@Clark:2015], `scikit-image` [@Van_der_Walt:2014], `rasterio` [@Gillies:2019], and `shapely` [@Gillies:2007]. To identify and manipulate overlapping polygons that result from the SAM segmentation, we rely on the `networkx` package [@Hagberg:2008]. The U-Net model is built and trained using `tensorflow` [@Abadi:2015] and `keras` [@Chollet:2015]; some parts of the machine learning workflow are handled with functions from the `scikit-learn` library [@Pedregosa:2011].

# Acknowledgements

Funding for this work came from the Quantitative Clastics Laboratory industrial consortium at the Bureau of Economic Geology, The University of Texas at Austin. We are thankful to Jake Covault, Sergey Fomel, and Tim Lawton for discussions.

# References