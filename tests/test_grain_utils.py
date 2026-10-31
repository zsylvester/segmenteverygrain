import unittest

import numpy as np

import segmenteverygrain as seg  # noqa: F401  (import before torch; see test_unet.py)
import torch

from segmenteverygrain.grain_utils import (
    _build_feature_extractor,
    _resize,
    extract_vgg16_features,
)


class TestResize(unittest.TestCase):

    def test_rank_and_dtype(self):
        rgb = np.random.randint(0, 255, (50, 70, 3)).astype(np.uint8)
        out = _resize(rgb, 32)
        self.assertEqual(out.shape, (32, 32, 3))
        self.assertEqual(out.dtype, np.float32)
        gray = np.random.rand(50, 70).astype(np.float32)
        self.assertEqual(_resize(gray, 32, antialias=True).shape, (32, 32))

    def test_constant_image_is_preserved(self):
        img = np.full((40, 40, 3), 77, dtype=np.uint8)
        np.testing.assert_allclose(_resize(img, 24), 77.0, atol=1e-4)


class TestFeatureExtractor(unittest.TestCase):
    """Shapes of the torchvision-based extractors (random weights, CPU)."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.images = np.random.randint(0, 255, (3, 64, 64, 3)).astype(np.uint8)

    def _features(self, model_name, layer_name=None):
        feats, model = extract_vgg16_features(
            self.images, model_name=model_name, layer_name=layer_name,
            pretrained=False, device="cpu",
        )
        self.assertEqual(feats.dtype, np.float32)
        self.assertEqual(feats.shape[0], 3)
        self.assertIsInstance(model, torch.nn.Module)
        return feats, model

    def test_vgg16_layers(self):
        self.assertEqual(self._features("VGG16")[0].shape, (3, 4096))
        self.assertEqual(self._features("VGG16", "fc1")[0].shape, (3, 4096))
        self.assertEqual(self._features("VGG16", -3)[0].shape, (3, 4096))
        self.assertEqual(self._features("VGG16", "flatten")[0].shape, (3, 25088))
        self.assertEqual(self._features("vgg16", "predictions")[0].shape, (3, 1000))

    def test_resnet50(self):
        self.assertEqual(self._features("ResNet50")[0].shape, (3, 2048))
        self.assertEqual(self._features("ResNet50", "predictions")[0].shape, (3, 1000))

    def test_inceptionv3(self):
        self.assertEqual(self._features("InceptionV3")[0].shape, (3, 2048))

    def test_model_reuse_and_batching(self):
        feats_a, model = self._features("VGG16")
        feats_b, _ = extract_vgg16_features(
            self.images, model=model, batch_size=1, device="cpu"
        )
        np.testing.assert_allclose(feats_a, feats_b, atol=1e-4)

    def test_unknown_model_and_layer(self):
        with self.assertRaises(ValueError):
            _build_feature_extractor("AlexNet", "fc2", pretrained=False)
        with self.assertRaises(ValueError):
            _build_feature_extractor("VGG16", "block5_pool", pretrained=False)
        with self.assertRaises(ValueError):
            extract_vgg16_features(np.zeros((2, 64, 64)), pretrained=False, device="cpu")


if __name__ == "__main__":
    unittest.main()
