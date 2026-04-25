Put your manually annotated training pairs in this folder.

Expected naming:
- `sample1_image.jpg` or `sample1_image.png`
- `sample1_mask.png`
- `sample2_image.jpg`
- `sample2_mask.png`

Rules:
- Each training image must have a matching mask.
- Filenames must contain `image` for the photo and `mask` for the label image.
- Matching pairs should sort together, so keep the same prefix before `_image` and `_mask`.
- Masks should be single-channel grayscale label images.

Mask pixel values:
- `0` = background
- `1` = grain interior
- `2` = grain boundary

Good examples:
- `blurred_run_01_image.png`
- `blurred_run_01_mask.png`

If your labels came from ImageJ-style colored masks, use the remapping notes in `Label_remapping.ipynb` to convert them into `0/1/2` masks before training.
