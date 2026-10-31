import os
import tempfile
import unittest

import numpy as np

# Import the package (and with it TensorFlow/SciPy) before torch. On macOS,
# importing torch first and TensorFlow afterwards loads two copies of the
# OpenMP runtime and aborts the process; this ordering avoids that.
import segmenteverygrain  # noqa: F401
import torch

from segmenteverygrain.unet import (
    UNet,
    convert_keras_model,
    get_device,
    load_unet,
    predict_batch,
    save_unet,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERAS_MODEL = os.path.join(REPO_ROOT, "models", "seg_model_smooth_labels.keras")
PT_MODEL = os.path.join(REPO_ROOT, "models", "seg_model_smooth_labels.pt")

try:
    import keras  # noqa: F401

    HAS_KERAS = True
except ImportError:  # pragma: no cover
    HAS_KERAS = False


class TestUNetArchitecture(unittest.TestCase):

    def test_output_shape(self):
        model = UNet().eval()
        x = torch.zeros(2, 3, 256, 256)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, 3, 256, 256))

    def test_parameter_count_matches_keras_unet(self):
        # Keras reports 2,161,683 params including BatchNorm running statistics
        # (1,472 non-trainable); the trainable count is 2,160,211.
        model = UNet()
        n_trainable = sum(p.numel() for p in model.parameters())
        self.assertEqual(n_trainable, 2160211)

    def test_non_square_input(self):
        model = UNet().eval()
        with torch.no_grad():
            y = model(torch.zeros(1, 3, 128, 256))
        self.assertEqual(tuple(y.shape), (1, 3, 128, 256))

    def test_config(self):
        model = UNet(in_channels=1, num_classes=2, base_filters=8)
        self.assertEqual(
            model.config, {"in_channels": 1, "num_classes": 2, "base_filters": 8}
        )


class TestSaveLoad(unittest.TestCase):

    def test_roundtrip(self):
        model = UNet(base_filters=4).eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y_before = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            save_unet(model, path)
            loaded = load_unet(path, device="cpu")
        self.assertEqual(loaded.config, model.config)
        self.assertFalse(loaded.training)
        with torch.no_grad():
            y_after = loaded(x)
        self.assertTrue(torch.allclose(y_before, y_after))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_unet("/nonexistent/model.pt")

    def test_rejects_non_unet_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.pt")
            torch.save({"foo": torch.zeros(1)}, path)
            with self.assertRaises(ValueError):
                load_unet(path)

    def test_get_device_explicit(self):
        self.assertEqual(get_device("cpu"), torch.device("cpu"))
        self.assertIsInstance(get_device(), torch.device)


class TestPredictBatch(unittest.TestCase):

    def setUp(self):
        self.model = UNet(base_filters=4).eval()

    def test_batch_output_is_probabilities(self):
        tiles = np.random.rand(3, 64, 64, 3).astype(np.float32)
        probs = predict_batch(self.model, tiles, batch_size=2)
        self.assertEqual(probs.shape, (3, 64, 64, 3))
        self.assertEqual(probs.dtype, np.float32)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
        self.assertTrue((probs >= 0).all())

    def test_single_tile(self):
        tile = np.random.rand(64, 64, 3)
        probs = predict_batch(self.model, tile)
        self.assertEqual(probs.shape, (64, 64, 3))

    def test_batching_is_consistent(self):
        tiles = np.random.rand(5, 64, 64, 3).astype(np.float32)
        a = predict_batch(self.model, tiles, batch_size=1)
        b = predict_batch(self.model, tiles, batch_size=5)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_wrong_channels(self):
        with self.assertRaises(ValueError):
            predict_batch(self.model, np.zeros((1, 64, 64, 1)))


@unittest.skipUnless(os.path.exists(PT_MODEL), "shipped .pt model not found")
class TestShippedModel(unittest.TestCase):

    def test_loads_and_predicts(self):
        model = load_unet(PT_MODEL, device="cpu")
        self.assertEqual(model.config, {"in_channels": 3, "num_classes": 3, "base_filters": 16})
        probs = predict_batch(model, np.random.rand(256, 256, 3))
        self.assertEqual(probs.shape, (256, 256, 3))


@unittest.skipUnless(
    HAS_KERAS and os.path.exists(KERAS_MODEL), "keras or shipped .keras model not available"
)
class TestKerasConversion(unittest.TestCase):

    def test_converted_model_matches_keras(self):
        import keras

        keras_model = keras.saving.load_model(KERAS_MODEL, compile=False)
        torch_model = convert_keras_model(KERAS_MODEL, verify=False)

        rng = np.random.default_rng(42)
        x = rng.random((2, 256, 256, 3), dtype=np.float32)
        keras_out = np.asarray(keras_model.predict(x, verbose=0))
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(x).permute(0, 3, 1, 2))
        torch_out = torch_out.permute(0, 2, 3, 1).numpy()
        self.assertLess(float(np.abs(keras_out - torch_out).max()), 1e-3)

    def test_shipped_pt_matches_shipped_keras(self):
        if not os.path.exists(PT_MODEL):
            self.skipTest("shipped .pt model not found")
        import keras
        from scipy.special import softmax

        keras_model = keras.saving.load_model(KERAS_MODEL, compile=False)
        torch_model = load_unet(PT_MODEL, device="cpu")
        rng = np.random.default_rng(7)
        x = rng.random((1, 256, 256, 3), dtype=np.float32)
        keras_probs = softmax(np.asarray(keras_model.predict(x, verbose=0)), axis=-1)
        torch_probs = predict_batch(torch_model, x)
        self.assertLess(float(np.abs(keras_probs - torch_probs).max()), 1e-4)

    def test_convert_and_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "converted.pt")
            convert_keras_model(KERAS_MODEL, path, verify=True)
            self.assertTrue(os.path.exists(path))
            model = load_unet(path, device="cpu")
            self.assertEqual(model.num_classes, 3)



class TestPredictImageWithUNet(unittest.TestCase):
    """predict_image / predict_image_tile running on the PyTorch model."""

    def setUp(self):
        self.model = UNet(base_filters=4).eval()

    def test_predict_image_tile(self):
        pred = segmenteverygrain.predict_image_tile(np.random.rand(256, 256, 3), self.model)
        self.assertEqual(pred.shape, (256, 256, 3))
        np.testing.assert_allclose(pred.sum(axis=-1), 1.0, atol=1e-5)

    def test_predict_image_shapes_and_batching(self):
        image = (np.random.rand(300, 420, 3) * 255).astype(np.uint8)
        pred_a = segmenteverygrain.predict_image(image, self.model, I=256, batch_size=4)
        pred_b = segmenteverygrain.predict_image(image, self.model, I=256, batch_size=64)
        self.assertEqual(pred_a.shape, (300, 420, 3))
        self.assertEqual(pred_a.dtype, np.float32)
        np.testing.assert_allclose(pred_a, pred_b, atol=1e-4)

    def test_predict_image_grayscale(self):
        image = np.random.rand(256, 256)
        pred = segmenteverygrain.predict_image(image, self.model, I=256)
        self.assertEqual(pred.shape, (256, 256, 3))

    def test_rejects_unknown_model(self):
        with self.assertRaises(TypeError):
            segmenteverygrain.predict_image_tile(np.zeros((256, 256, 3)), object())

    def test_keras_style_model_warns(self):
        class MockKeras:
            def predict(self, x, verbose=0):
                return np.ones_like(x)

        with self.assertWarns(FutureWarning):
            pred = segmenteverygrain.predict_image_tile(np.zeros((256, 256, 3)), MockKeras())
        self.assertTrue(np.allclose(pred, 1.0 / 3.0))


@unittest.skipUnless(
    HAS_KERAS and os.path.exists(KERAS_MODEL) and os.path.exists(PT_MODEL),
    "keras or shipped models not available",
)
class TestPredictImageParity(unittest.TestCase):
    """The full tiled prediction gives the same result with the .pt and .keras models."""

    def test_predict_image_matches_keras(self):
        import keras
        import warnings

        keras_model = keras.saving.load_model(KERAS_MODEL, compile=False)
        torch_model = load_unet(PT_MODEL, device="cpu")
        rng = np.random.default_rng(3)
        image = (rng.random((300, 400, 3)) * 255).astype(np.uint8)
        pred_torch = segmenteverygrain.predict_image(image, torch_model, I=256)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pred_keras = segmenteverygrain.predict_image(image, keras_model, I=256)
        self.assertLess(float(np.abs(pred_torch - pred_keras).max()), 1e-3)

if __name__ == "__main__":
    unittest.main()
