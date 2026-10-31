import os
import tempfile
import unittest

import numpy as np

import segmenteverygrain as seg  # noqa: F401  (import before torch; see test_unet.py)
import torch
from PIL import Image

from segmenteverygrain.training import (
    GrainPatchDataset,
    augment_pair,
    create_and_train_model,
    create_train_val_test_data,
    evaluate_model,
    patchify_training_data,
    train_model,
    weighted_crossentropy,
)
from segmenteverygrain.unet import UNet


def _write_synthetic_patches(folder, n, size=64):
    """Write n image/mask PNG pairs: a bright disk (grain) with a boundary ring."""
    images_dir = os.path.join(folder, "images")
    masks_dir = os.path.join(folder, "labels")
    os.makedirs(images_dir)
    os.makedirs(masks_dir)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:size, :size]
    for i in range(n):
        cy, cx = rng.integers(size // 4, 3 * size // 4, size=2)
        r = rng.integers(size // 8, size // 4)
        dist = np.hypot(yy - cy, xx - cx)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[dist < r] = 1
        mask[(dist >= r) & (dist < r + 2)] = 2
        image = np.full((size, size, 3), 40, dtype=np.uint8)
        image[mask == 1] = 200
        image[mask == 2] = 120
        Image.fromarray(image).save(os.path.join(images_dir, f"im{i:03d}.png"))
        Image.fromarray(mask).save(os.path.join(masks_dir, f"im{i:03d}.png"))
    return images_dir, masks_dir


class TestAugmentPair(unittest.TestCase):

    def test_shapes_dtypes_and_mask_values(self):
        rng = np.random.default_rng(1)
        image = rng.random((64, 64, 3), dtype=np.float32)
        mask = rng.integers(0, 3, size=(64, 64))
        for seed in range(10):
            img_a, mask_a = augment_pair(image, mask, rng=np.random.default_rng(seed))
            self.assertEqual(img_a.shape, (64, 64, 3))
            self.assertEqual(img_a.dtype, np.float32)
            self.assertEqual(mask_a.shape, (64, 64))
            self.assertTrue(img_a.min() >= 0.0 and img_a.max() <= 1.0)
            self.assertTrue(set(np.unique(mask_a)).issubset({0, 1, 2}))

    def test_geometry_applied_to_image_and_mask_together(self):
        # A mask that marks where the image is bright; after augmentation the
        # bright pixels must still coincide with mask==1.
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[10:30, 5:40] = 1.0
        mask = (image[:, :, 0] > 0.5).astype(np.int64)
        for seed in range(10):
            img_a, mask_a = augment_pair(image, mask, rng=np.random.default_rng(seed))
            bright = img_a.mean(axis=-1) > 0.5
            self.assertGreater((bright == (mask_a == 1)).mean(), 0.97)


class TestDatasetAndPatchify(unittest.TestCase):

    def test_dataset_getitem(self):
        with tempfile.TemporaryDirectory() as tmp:
            images_dir, masks_dir = _write_synthetic_patches(tmp, 3)
            files = sorted(os.listdir(images_dir))
            ds = GrainPatchDataset(
                [os.path.join(images_dir, f) for f in files],
                [os.path.join(masks_dir, f) for f in files],
                augment=True,
            )
            self.assertEqual(len(ds), 3)
            image, mask = ds[0]
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(tuple(mask.shape), (64, 64))
        self.assertEqual(mask.dtype, torch.int64)

    def test_dataset_length_mismatch(self):
        with self.assertRaises(ValueError):
            GrainPatchDataset(["a.png", "b.png"], ["a.png"])

    def test_patchify(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "input")
            os.makedirs(input_dir)
            rng = np.random.default_rng(0)
            image = rng.integers(0, 255, (400, 520, 3)).astype(np.uint8)
            mask = rng.integers(0, 3, (400, 520)).astype(np.uint8)
            Image.fromarray(image).save(os.path.join(input_dir, "s1_image.png"))
            Image.fromarray(mask).save(os.path.join(input_dir, "s1_mask.png"))
            images_dir, labels_dir = patchify_training_data(input_dir, os.path.join(tmp, "out"))
            n_images = len(os.listdir(images_dir))
            n_labels = len(os.listdir(labels_dir))
            # rows: (400-256)//128+1 = 2, cols: (520-256)//128+1 = 3
            self.assertEqual(n_images, 6)
            self.assertEqual(n_labels, 6)
            first = np.array(Image.open(os.path.join(images_dir, "im000.png")))
            self.assertEqual(first.shape, (256, 256, 3))
            np.testing.assert_array_equal(first, image[:256, :256])
            first_mask = np.array(Image.open(os.path.join(labels_dir, "im000.png")))
            np.testing.assert_array_equal(first_mask, mask[:256, :256])

    def test_patchify_missing_dir(self):
        self.assertIsNone(patchify_training_data("/nonexistent/dir", "/tmp/unused"))


class TestTrainingLoop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        images_dir, masks_dir = _write_synthetic_patches(cls.tmp.name, 12)
        cls.train, cls.val, cls.test = create_train_val_test_data(
            images_dir, masks_dir, augmentation=True, batch_size=4, num_workers=0
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_split_sizes(self):
        # 12 patches: 15% test -> 2, then 25% of 10 -> 3 val, 7 train
        self.assertEqual(len(self.train.dataset), 7)
        self.assertEqual(len(self.val.dataset), 3)
        self.assertEqual(len(self.test.dataset), 2)
        images, masks = next(iter(self.train))
        self.assertEqual(tuple(images.shape), (4, 3, 64, 64))
        self.assertEqual(tuple(masks.shape), (4, 64, 64))

    def test_train_model_reduces_loss(self):
        torch.manual_seed(0)
        model = UNet(base_filters=4)
        history = train_model(model, self.train, self.val, epochs=3, device="cpu")
        self.assertEqual(len(history["loss"]), 3)
        self.assertEqual(len(history["val_accuracy"]), 3)
        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertFalse(model.training)

    def test_evaluate_model(self):
        model = UNet(base_filters=4)
        results = evaluate_model(model, self.test, device="cpu")
        self.assertEqual(set(results), {"loss", "accuracy", "mean_iou", "per_class_iou"})
        self.assertEqual(set(results["per_class_iou"]), {"background", "grain", "boundary"})
        self.assertTrue(0.0 <= results["accuracy"] <= 1.0)

    def test_create_and_train_model_from_file(self):
        import matplotlib

        matplotlib.use("Agg")
        from segmenteverygrain.unet import save_unet

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "start.pt")
            save_unet(UNet(base_filters=4), path)
            model = create_and_train_model(
                self.train, self.val, self.test, model_file=path, epochs=1, device="cpu"
            )
        self.assertIsInstance(model, UNet)
        self.assertEqual(model.base_filters, 4)


class TestLossDevice(unittest.TestCase):

    def test_loss_accepts_numpy(self):
        y_true = np.array([[[0, 1, 2]]])
        y_pred = np.random.rand(1, 3, 1, 3).astype(np.float32)
        loss = weighted_crossentropy(y_true, y_pred)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
