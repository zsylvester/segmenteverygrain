import os
import tempfile
import unittest
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from shapely.geometry import Polygon
import segmenteverygrain as seg
import tensorflow as tf


class TestGetGrainsFromPatches(unittest.TestCase):

    def setUp(self):
        # Create a mock image
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a mock Axes object with patches
        self.fig, self.ax = plt.subplots()
        self.patches = [
            MplPolygon(np.array([[10, 10], [20, 10], [20, 20], [10, 20]]), closed=True),
            MplPolygon(np.array([[30, 30], [40, 30], [40, 40], [30, 40]]), closed=True),
        ]
        for patch in self.patches:
            self.ax.add_patch(patch)
        plt.axis("equal")

    def test_get_grains_from_patches(self):
        all_grains, rasterized, mask_all = seg.get_grains_from_patches(
            self.ax, self.image
        )

        # Check the number of grains
        self.assertEqual(len(all_grains), len(self.patches))

        # Check the type of grains
        for grain in all_grains:
            self.assertIsInstance(grain, Polygon)

        # Check the rasterized image
        self.assertEqual(rasterized.shape, self.image.shape[:2])
        self.assertTrue(np.any(rasterized > 0))

        # Check the mask_all image
        self.assertEqual(mask_all.shape, self.image.shape[:2])
        self.assertTrue(np.any(mask_all == 1))
        self.assertTrue(np.any(mask_all == 2))


class TestRasterizeGrains(unittest.TestCase):

    def setUp(self):
        # Create a mock image
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create mock grains
        self.grains = [
            Polygon([(10, 10), (20, 10), (20, 20), (10, 20)]),
            Polygon([(30, 30), (40, 30), (40, 40), (30, 40)]),
        ]

    def test_rasterize_grains_shape(self):
        rasterized = seg.rasterize_grains(self.grains, self.image)
        self.assertEqual(rasterized.shape, self.image.shape[:2])

    def test_rasterize_grains_labels(self):
        rasterized = seg.rasterize_grains(self.grains, self.image)
        unique_labels = np.unique(rasterized)
        self.assertTrue(set(unique_labels).issubset({0, 1, 2}))

    def test_rasterize_grains_nonzero(self):
        rasterized = seg.rasterize_grains(self.grains, self.image)
        self.assertTrue(np.any(rasterized > 0))

    def test_rasterize_grains_label_positions(self):
        rasterized = seg.rasterize_grains(self.grains, self.image)
        self.assertEqual(rasterized[15, 15], 1)
        self.assertEqual(rasterized[35, 35], 2)


class TestPredictImageTile(unittest.TestCase):
    def setUp(self):
        # Create a mock model with a predict method
        class MockModel:
            def predict(self, x, verbose=0):
                return np.ones_like(x)

        self.model = MockModel()

    def test_predict_image_tile_3d(self):
        im_tile = np.zeros((256, 256, 3))
        im_tile_pred = seg.predict_image_tile(im_tile, self.model)
        self.assertEqual(im_tile_pred.shape, (256, 256, 3))
        # Mock returns all-ones logits; softmax converts to uniform 1/3 probabilities
        self.assertTrue(np.allclose(im_tile_pred, 1.0 / 3.0))

    def test_predict_image_tile_invalid_input(self):
        im_tile = np.zeros((256,))
        with self.assertRaises(ValueError):
            seg.predict_image_tile(im_tile, self.model)


class TestPredictImage(unittest.TestCase):
    def setUp(self):
        # Create a mock model with a predict method
        class MockModel:
            def predict(self, x, verbose=0):
                return np.ones_like(x)

        self.model = MockModel()

    def test_predict_image_2d(self):
        big_im = np.zeros((512, 512))
        big_im_pred = seg.predict_image(big_im, self.model, 256)
        self.assertEqual(big_im_pred.shape, (512, 512, 3))
        self.assertTrue(np.all(big_im_pred > 0))

    def test_predict_image_3d(self):
        big_im = np.zeros((512, 512, 3))
        big_im_pred = seg.predict_image(big_im, self.model, 256)
        self.assertEqual(big_im_pred.shape, (512, 512, 3))
        self.assertTrue(np.all(big_im_pred > 0))

    def test_predict_image_invalid_input(self):
        big_im = np.zeros((512,))
        with self.assertRaises(ValueError):
            seg.predict_image(big_im, self.model, 256)


class TestLabelGrains(unittest.TestCase):

    def setUp(self):
        # Create a mock image
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a mock prediction
        self.prediction = np.zeros((100, 100, 3), dtype=np.float32)
        self.prediction[10:20, 10:20, 1] = 1  # grain
        self.prediction[30:40, 30:40, 1] = 1  # grain
        self.prediction[10:20, 10:20, 2] = 1  # boundary
        self.prediction[30:40, 30:40, 2] = 1  # boundary

    def test_label_grains_output_shapes(self):
        labels_simple, all_coords = seg.label_grains(self.image, self.prediction)
        self.assertEqual(labels_simple.shape, self.image.shape[:2])
        self.assertEqual(all_coords.shape[1], 2)

    def test_label_grains_nonzero_labels(self):
        labels_simple, all_coords = seg.label_grains(self.image, self.prediction)
        self.assertTrue(np.any(labels_simple > 0))

    def test_label_grains_coords_within_image(self):
        labels_simple, all_coords = seg.label_grains(self.image, self.prediction)
        self.assertTrue(np.all(all_coords[:, 0] < self.image.shape[1]))
        self.assertTrue(np.all(all_coords[:, 1] < self.image.shape[0]))

    def test_label_grains_no_background_coords(self):
        labels_simple, all_coords = seg.label_grains(self.image, self.prediction)
        background_probs = self.prediction[:, :, 0][all_coords[:, 1], all_coords[:, 0]]
        self.assertTrue(np.all(background_probs < 0.3))


class TestSamSegmentationEmptyPrompts(unittest.TestCase):

    def test_empty_coords_raises_informative_error(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_pred = np.zeros((100, 100, 3), dtype=np.float32)
        labels = np.zeros((100, 100), dtype=int)
        coords = np.empty((0, 2), dtype=np.int32)
        # The guard fires before the SAM model is used, so sam can be None
        with self.assertRaisesRegex(ValueError, "No SAM prompts"):
            seg.sam_segmentation(
                None, image, image_pred, coords, labels, min_area=400.0
            )


class TestPredictLargeImageNoGrains(unittest.TestCase):

    def test_no_grains_detected_warns(self):
        # Mock model that predicts background everywhere (as logits, since
        # predict_image_tile applies softmax to the model output)
        class BackgroundModel:
            def predict(self, x, verbose=0):
                pred = np.zeros(x.shape[:-1] + (3,), dtype=np.float32)
                pred[..., 0] = 10.0  # large background logit
                return pred

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "blank.png")
            Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(fname)
            with self.assertWarnsRegex(UserWarning, "No grains were detected"):
                all_grains, image_pred, all_coords = seg.predict_large_image(
                    fname,
                    BackgroundModel(),
                    use_sam=False,
                    min_area=400.0,
                    patch_size=2000,
                    overlap=300,
                )
        self.assertEqual(len(all_grains), 0)
        self.assertEqual(len(all_coords), 0)


class TestFindOverlappingPolygons(unittest.TestCase):

    def setUp(self):
        # Create some mock polygons
        self.polygons = [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),  # Polygon 1
            Polygon(
                [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]
            ),  # Polygon 2 (overlaps with Polygon 1)
            Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]),  # Polygon 3 (no overlap)
            Polygon(
                [(4.5, 4.5), (6.5, 4.5), (6.5, 6.5), (4.5, 6.5)]
            ),  # Polygon 4 (overlaps with Polygon 3)
        ]

    def test_find_overlapping_polygons(self):
        overlapping_polygons = seg.find_overlapping_polygons(self.polygons)

        # Check the number of overlapping pairs
        self.assertEqual(len(overlapping_polygons), 2)

        # Check the overlapping pairs
        self.assertIn((0, 1), overlapping_polygons)
        self.assertIn((2, 3), overlapping_polygons)

    def test_no_overlapping_polygons(self):
        non_overlapping_polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]
        overlapping_polygons = seg.find_overlapping_polygons(non_overlapping_polygons)

        # Check that there are no overlapping pairs
        self.assertEqual(len(overlapping_polygons), 0)

    def test_all_overlapping_polygons(self):
        all_overlapping_polygons = [
            Polygon([(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)]),
            Polygon([(1, 1), (4, 1), (4, 4), (1, 4)]),
            Polygon([(1.5, 1.5), (4.5, 1.5), (4.5, 4.5), (1.5, 4.5)]),
        ]
        overlapping_polygons = seg.find_overlapping_polygons(all_overlapping_polygons)

        # Check the number of overlapping pairs
        self.assertEqual(len(overlapping_polygons), 3)

        # Check the overlapping pairs
        self.assertIn((0, 1), overlapping_polygons)
        self.assertIn((0, 2), overlapping_polygons)
        self.assertIn((1, 2), overlapping_polygons)


class TestUnet(unittest.TestCase):

    def test_unet_output_shape(self):
        model = seg.Unet()
        input_shape = (1, 256, 256, 3)
        dummy_input = np.zeros(input_shape, dtype=np.float32)
        output = model.predict(dummy_input)
        self.assertEqual(output.shape, (1, 256, 256, 3))

    def test_unet_layer_names(self):
        model = seg.Unet()
        expected_layer_names = [
            "input",
            "conv2d",
            "conv2d_1",
            "batch_normalization",
            "max_pooling2d",
            "conv2d_2",
            "conv2d_3",
            "batch_normalization_1",
            "max_pooling2d_1",
            "conv2d_4",
            "conv2d_5",
            "batch_normalization_2",
            "max_pooling2d_2",
            "conv2d_6",
            "conv2d_7",
            "batch_normalization_3",
            "max_pooling2d_3",
            "conv2d_8",
            "conv2d_9",
            "batch_normalization_4",
            "conv2d_transpose",
            "concatenate",
            "conv2d_10",
            "conv2d_11",
            "batch_normalization_5",
            "conv2d_transpose_1",
            "concatenate_1",
            "conv2d_12",
            "conv2d_13",
            "batch_normalization_6",
            "conv2d_transpose_2",
            "concatenate_2",
            "conv2d_14",
            "conv2d_15",
            "batch_normalization_7",
            "conv2d_transpose_3",
            "concatenate_3",
            "conv2d_16",
            "conv2d_17",
            "batch_normalization_8",
            "conv2d_18",
        ]
        actual_layer_names = [layer.name for layer in model.layers]
        self.assertEqual(actual_layer_names, expected_layer_names)

    def test_unet_compile(self):
        model = seg.Unet()
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        self.assertIsNotNone(model.metrics)


class TestWeightedCrossentropy(unittest.TestCase):

    def test_weighted_crossentropy_shape(self):
        y_true = tf.constant([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]], dtype=tf.float32)
        y_pred = tf.constant(
            [[[[2.0, 1.0, 0.1], [0.5, 2.0, 0.5], [0.1, 0.5, 2.0]]]], dtype=tf.float32
        )
        loss = seg.weighted_crossentropy(y_true, y_pred)
        self.assertEqual(loss.shape, ())

    def test_weighted_crossentropy_value(self):
        y_true = tf.constant([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]], dtype=tf.float32)
        y_pred = tf.constant(
            [[[[2.0, 1.0, 0.1], [0.5, 2.0, 0.5], [0.1, 0.5, 2.0]]]], dtype=tf.float32
        )
        loss = seg.weighted_crossentropy(y_true, y_pred)
        expected_loss = 0.9759197  # Precomputed expected loss value (with label smoothing=0.1)
        self.assertAlmostEqual(loss.numpy(), expected_loss, places=5)


class TestCalculateIoU(unittest.TestCase):

    def setUp(self):
        # Create some mock polygons
        self.poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # Square polygon
        self.poly2 = Polygon(
            [(1, 1), (3, 1), (3, 3), (1, 3)]
        )  # Overlapping square polygon
        self.poly3 = Polygon(
            [(3, 3), (5, 3), (5, 5), (3, 5)]
        )  # Non-overlapping square polygon
        self.poly4 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])  # Larger square polygon

    def test_calculate_iou_overlapping(self):
        iou = seg.calculate_iou(self.poly1, self.poly2)
        expected_iou = 1 / 7  # Precomputed expected IoU value
        self.assertAlmostEqual(iou, expected_iou, places=5)

    def test_calculate_iou_non_overlapping(self):
        iou = seg.calculate_iou(self.poly1, self.poly3)
        self.assertEqual(iou, 0.0)

    def test_calculate_iou_contained(self):
        iou = seg.calculate_iou(self.poly1, self.poly4)
        expected_iou = 1 / 4  # Precomputed expected IoU value
        self.assertAlmostEqual(iou, expected_iou, places=5)

    def test_calculate_iou_identical(self):
        iou = seg.calculate_iou(self.poly1, self.poly1)
        self.assertEqual(iou, 1.0)


class TestPickMostSimilarPolygon(unittest.TestCase):

    def setUp(self):
        # Create some mock polygons
        self.polygons = [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),  # Polygon 1
            Polygon(
                [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]
            ),  # Polygon 2 (overlaps with Polygon 1)
            Polygon([(4, 4), (6, 4), (6, 6), (4, 6)]),  # Polygon 3 (no overlap)
            Polygon(
                [(4.5, 4.5), (6.5, 4.5), (6.5, 6.5), (4.5, 6.5)]
            ),  # Polygon 4 (overlaps with Polygon 3)
        ]

    def test_pick_most_similar_polygon(self):
        most_similar_polygon = seg.pick_most_similar_polygon(self.polygons)

        # Check that the most similar polygon is one of the input polygons
        self.assertIn(most_similar_polygon, self.polygons)

        # Check that the most similar polygon is the one with the highest average IoU
        expected_polygon = self.polygons[0]  # Precomputed expected most similar polygon
        self.assertEqual(most_similar_polygon, expected_polygon)

    def test_pick_most_similar_polygon_no_overlap(self):
        non_overlapping_polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]
        most_similar_polygon = seg.pick_most_similar_polygon(non_overlapping_polygons)

        # Check that the most similar polygon is one of the input polygons
        self.assertIn(most_similar_polygon, non_overlapping_polygons)

        # Check that the most similar polygon is the one with the highest average IoU
        expected_polygon = non_overlapping_polygons[
            0
        ]  # Precomputed expected most similar polygon
        self.assertEqual(most_similar_polygon, expected_polygon)

    def test_pick_most_similar_polygon_identical(self):
        identical_polygons = [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        ]
        most_similar_polygon = seg.pick_most_similar_polygon(identical_polygons)

        # Check that the most similar polygon is one of the input polygons
        self.assertIn(most_similar_polygon, identical_polygons)

        # Check that the most similar polygon is the one with the highest average IoU
        expected_polygon = identical_polygons[
            0
        ]  # Precomputed expected most similar polygon
        self.assertEqual(most_similar_polygon, expected_polygon)


class TestFindConnectedComponents(unittest.TestCase):

    def setUp(self):
        # Create some mock polygons
        self.polygons = [
            Polygon(
                [(0, 0), (2, 0), (2, 2), (0, 2)]
            ),  # Polygon 1 (overlaps with Polygon 2)
            Polygon(
                [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]
            ),  # Polygon 2 (overlaps with Polygon 1)
            Polygon(
                [(4, 4), (6, 4), (6, 6), (4, 6)]
            ),  # Polygon 3 (overlaps with Polygon 4)
            Polygon(
                [(4.5, 4.5), (6.5, 4.5), (6.5, 6.5), (4.5, 6.5)]
            ),  # Polygon 4 (overlaps with Polygon 3)
            Polygon([(4, 0), (6, 0), (6, 2), (4, 2)]),  # Polygon 5 (no overlap)
        ]

    def test_find_connected_components(self):
        new_grains, comps, g = seg.find_connected_components(
            self.polygons, min_area=1.0
        )

        # Check the number of new grains
        self.assertEqual(len(new_grains), 1)

        # Check the number of connected components
        self.assertEqual(len(comps), 2)

        # Check the nodes in the graph
        self.assertEqual(len(g.nodes), 4)

        # Check the edges in the graph
        self.assertEqual(len(g.edges), 2)

    def test_find_connected_components_with_min_area(self):
        new_grains, comps, g = seg.find_connected_components(
            self.polygons, min_area=2.0
        )

        # Check the number of new grains
        self.assertEqual(len(new_grains), 1)

        # Check the number of connected components
        self.assertEqual(len(comps), 2)

        # Check the nodes in the graph
        self.assertEqual(len(g.nodes), 4)

        # Check the edges in the graph
        self.assertEqual(len(g.edges), 2)

    def test_find_connected_components_no_overlap(self):
        non_overlapping_polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]
        new_grains, comps, g = seg.find_connected_components(
            non_overlapping_polygons, min_area=0.5
        )

        # Check the number of new grains
        self.assertEqual(len(new_grains), 2)

        # Check the number of connected components
        self.assertEqual(len(comps), 0)

        # Check the nodes in the graph
        self.assertEqual(len(g.nodes), 0)

        # Check the edges in the graph
        self.assertEqual(len(g.edges), 0)


class TestMergeOverlappingPolygons(unittest.TestCase):

    def setUp(self):
        # Create some mock polygons
        self.poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # Square polygon
        self.poly2 = Polygon(
            [(1, 1), (3, 1), (3, 3), (1, 3)]
        )  # Overlapping square polygon
        self.poly3 = Polygon(
            [(3, 3), (5, 3), (5, 5), (3, 5)]
        )  # Non-overlapping square polygon
        self.poly4 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])  # Larger square polygon
        self.all_grains = [self.poly1, self.poly2, self.poly3, self.poly4]
        self.new_grains = [self.poly1, self.poly3]
        self.comps = [{0, 1}, {2, 3}]
        self.min_area = 1.0
        self.image_pred = np.zeros((10, 10, 3))

    def test_merge_overlapping_polygons(self):
        merged_grains = seg.merge_overlapping_polygons(
            self.all_grains, self.new_grains, self.comps, self.min_area, self.image_pred
        )
        self.assertEqual(len(merged_grains), 2)
        self.assertTrue(any(poly.equals(self.poly1) for poly in merged_grains))
        self.assertTrue(any(poly.equals(self.poly3) for poly in merged_grains))
        self.assertFalse(any(poly.equals(self.poly4) for poly in merged_grains))

    def test_merge_overlapping_polygons_min_area(self):
        min_area = 5.0
        merged_grains = seg.merge_overlapping_polygons(
            self.all_grains, self.new_grains, self.comps, min_area, self.image_pred
        )
        self.assertEqual(len(merged_grains), 2)
        self.assertTrue(any(poly.equals(self.poly1) for poly in merged_grains))
        self.assertTrue(any(poly.equals(self.poly3) for poly in merged_grains))

    def test_merge_overlapping_polygons_no_overlap(self):
        polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            Polygon([(2.1, 2.1), (3.1, 2.1), (3.1, 3.1), (2.1, 3.1)]),
        ]
        comps = [{2, 3}]
        new_grains = polygons[:2]
        merged_grains = seg.merge_overlapping_polygons(
            polygons, new_grains, comps, 0.5, self.image_pred
        )
        self.assertEqual(len(merged_grains), 3)
        self.assertTrue(any(poly.equals(polygons[0]) for poly in merged_grains))
        self.assertTrue(any(poly.equals(polygons[1]) for poly in merged_grains))
        self.assertTrue(any(poly.equals(polygons[2]) for poly in merged_grains))
        self.assertFalse(any(poly.equals(polygons[3]) for poly in merged_grains))

    def test_merge_overlapping_polygons_large_overlap(self):
        poly5 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])  # Very large polygon
        all_grains = [self.poly1, self.poly2, self.poly3, self.poly4, poly5]
        new_grains = []
        comps = [{0, 1, 2, 3, 4}]
        merged_grains = seg.merge_overlapping_polygons(
            all_grains, new_grains, comps, self.min_area, self.image_pred
        )
        self.assertEqual(len(merged_grains), 1)
        self.assertTrue(any(poly.equals(self.poly4) for poly in merged_grains))


class TestClassifyPoints(unittest.TestCase):

    def test_classify_points_on_line(self):
        feature1 = [1, 2, 3]
        feature2 = [1, 2, 3]
        x1, y1 = 0, 0
        x2, y2 = 4, 4
        classifications = seg.classify_points(feature1, feature2, x1, y1, x2, y2)
        self.assertEqual(classifications, [0, 0, 0])

    def test_classify_points_one_side(self):
        feature1 = [1, 2, 3]
        feature2 = [2, 3, 4]
        x1, y1 = 0, 0
        x2, y2 = 4, 4
        classifications = seg.classify_points(feature1, feature2, x1, y1, x2, y2)
        self.assertEqual(classifications, [0, 0, 0])

    def test_classify_points_other_side(self):
        feature1 = [2, 3, 4]
        feature2 = [1, 2, 3]
        x1, y1 = 0, 0
        x2, y2 = 4, 4
        classifications = seg.classify_points(feature1, feature2, x1, y1, x2, y2)
        self.assertEqual(classifications, [1, 1, 1])

    def test_classify_points_mixed(self):
        feature1 = [1, 2, 3, 4]
        feature2 = [2, 3, 2, 1]
        x1, y1 = 0, 0
        x2, y2 = 4, 4
        classifications = seg.classify_points(feature1, feature2, x1, y1, x2, y2)
        self.assertEqual(classifications, [0, 0, 1, 1])


class TestWeightedEcdf(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.values = rng.normal(size=200)
        self.weights = rng.uniform(0.1, 10.0, 200)

    def test_unweighted_matches_legacy_curve(self):
        # The pre-weighting code plotted np.arange(1, n + 1) / n reversed;
        # the unweighted ECDF must reproduce it exactly
        values_sorted = np.sort(self.values)
        n = len(values_sorted)
        legacy = (np.arange(1, n + 1) / n)[::-1]
        v, cdf, exceedance = seg.weighted_ecdf(self.values)
        np.testing.assert_array_equal(v, values_sorted)
        np.testing.assert_allclose(exceedance, legacy)

    def test_values_are_sorted(self):
        v, _, _ = seg.weighted_ecdf(self.values, self.weights)
        np.testing.assert_array_equal(v, np.sort(self.values))

    def test_cdf_is_monotonic_and_ends_at_one(self):
        _, cdf, _ = seg.weighted_ecdf(self.values, self.weights)
        self.assertTrue(np.all(np.diff(cdf) >= 0))
        self.assertAlmostEqual(cdf[-1], 1.0)
        self.assertGreater(cdf[0], 0.0)

    def test_exceedance_is_monotonic_and_starts_at_one(self):
        _, _, exceedance = seg.weighted_ecdf(self.values, self.weights)
        self.assertTrue(np.all(np.diff(exceedance) <= 0))
        self.assertAlmostEqual(exceedance[0], 1.0)
        self.assertGreater(exceedance[-1], 0.0)

    def test_integer_weights_match_replication_exactly(self):
        # Weighting by w must be identical to repeating each value w times,
        # evaluated at the top (CDF) and bottom (exceedance) of each step
        rng = np.random.default_rng(7)
        values = rng.uniform(0.01, 2.0, 100)
        weights = rng.integers(1, 12, 100).astype(float)
        replicated = np.repeat(values, weights.astype(int))

        v_w, cdf_w, exc_w = seg.weighted_ecdf(values, weights)
        v_r, cdf_r, exc_r = seg.weighted_ecdf(replicated)

        top = np.searchsorted(v_r, v_w, side="right") - 1
        bottom = np.searchsorted(v_r, v_w, side="left")
        np.testing.assert_allclose(cdf_r[top], cdf_w, atol=1e-12)
        np.testing.assert_allclose(exc_r[bottom], exc_w, atol=1e-12)

    def test_invariant_to_weight_scaling(self):
        # Only the relative weights matter, so the units of the areas do not
        _, cdf_a, _ = seg.weighted_ecdf(self.values, self.weights)
        _, cdf_b, _ = seg.weighted_ecdf(self.values, self.weights * 1e6)
        np.testing.assert_allclose(cdf_a, cdf_b)

    def test_uniform_weights_match_no_weights(self):
        _, cdf_a, _ = seg.weighted_ecdf(self.values)
        _, cdf_b, _ = seg.weighted_ecdf(self.values, np.full(200, 3.0))
        np.testing.assert_allclose(cdf_a, cdf_b)

    def test_mismatched_weights_raise(self):
        with self.assertRaises(ValueError):
            seg.weighted_ecdf(self.values, self.weights[:10])

    def test_single_value(self):
        v, cdf, exceedance = seg.weighted_ecdf([2.5], [7.0])
        np.testing.assert_array_equal(v, [2.5])
        np.testing.assert_allclose(cdf, [1.0])
        np.testing.assert_allclose(exceedance, [1.0])


class TestWeightedPercentile(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.values = rng.normal(size=500)
        self.percentiles = np.array([5, 16, 25, 50, 75, 84, 95])

    def test_equal_weights_approximate_numpy(self):
        # The midpoint convention differs slightly from numpy's, but not much
        result = seg.weighted_percentile(self.values, self.percentiles)
        expected = np.percentile(self.values, self.percentiles)
        np.testing.assert_allclose(result, expected, atol=0.02)

    def test_integer_weights_match_replication(self):
        rng = np.random.default_rng(7)
        values = rng.uniform(0.01, 2.0, 200)
        weights = rng.integers(1, 12, 200).astype(float)
        replicated = np.repeat(values, weights.astype(int))
        result = seg.weighted_percentile(values, self.percentiles, weights=weights)
        expected = np.percentile(replicated, self.percentiles)
        np.testing.assert_allclose(result, expected, atol=0.02)

    def test_monotonic_in_percentile(self):
        weights = np.abs(self.values) + 0.1
        result = seg.weighted_percentile(self.values, self.percentiles, weights)
        self.assertTrue(np.all(np.diff(result) >= 0))

    def test_endpoints_are_min_and_max(self):
        weights = np.abs(self.values) + 0.1
        result = seg.weighted_percentile(self.values, [0, 100], weights)
        self.assertAlmostEqual(result[0], np.min(self.values))
        self.assertAlmostEqual(result[1], np.max(self.values))

    def test_invariant_to_weight_scaling(self):
        weights = np.abs(self.values) + 0.1
        a = seg.weighted_percentile(self.values, self.percentiles, weights)
        b = seg.weighted_percentile(self.values, self.percentiles, weights * 1e6)
        np.testing.assert_allclose(a, b)

    def test_weighting_pulls_median_toward_heavy_values(self):
        # Weighting grain sizes by area must shift the median toward the
        # large grains, which is the whole point of area weighting
        values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        areas = values ** 2
        unweighted = seg.weighted_percentile(values, 50)
        weighted = seg.weighted_percentile(values, 50, weights=areas)
        self.assertAlmostEqual(float(unweighted), 3.0)
        self.assertGreater(float(weighted), 50.0)

    def test_scalar_percentile(self):
        result = seg.weighted_percentile(self.values, 50)
        self.assertEqual(np.asarray(result).shape, ())

    def test_mismatched_weights_raise(self):
        with self.assertRaises(ValueError):
            seg.weighted_percentile(self.values, self.percentiles, weights=[1.0, 2.0])


class TestPlotHistogramOfAxisLengths(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.major = rng.uniform(0.2, 4.0, 120)
        self.minor = self.major * rng.uniform(0.4, 0.95, 120)
        self.area = np.pi * 0.25 * self.major * self.minor

    def tearDown(self):
        plt.close("all")

    def bar_heights(self, ax):
        return np.array([patch.get_height() for patch in ax.containers[0]])

    def test_unweighted_labels_count(self):
        fig, ax = seg.plot_histogram_of_axis_lengths(self.major, self.minor)
        self.assertEqual(ax.get_ylabel(), "count")
        # Without weights the bars are plain counts, so they sum to the
        # number of grains inside the binning range
        self.assertLessEqual(self.bar_heights(ax).sum(), len(self.major))

    def test_weighted_labels_area_weighted_count(self):
        fig, ax = seg.plot_histogram_of_axis_lengths(
            self.major, self.minor, area=self.area
        )
        self.assertEqual(ax.get_ylabel(), "area-weighted count")

    def test_weights_are_normalized_to_grain_count(self):
        # Weights sum to the number of grains, so the y axis stays on a
        # count-like scale no matter what units the areas are in
        fig, ax = seg.plot_histogram_of_axis_lengths(
            self.major, self.minor, area=self.area
        )
        self.assertAlmostEqual(self.bar_heights(ax).sum(), len(self.major), places=6)

    def test_invariant_to_area_units(self):
        fig_a, ax_a = seg.plot_histogram_of_axis_lengths(
            self.major, self.minor, area=self.area
        )
        fig_b, ax_b = seg.plot_histogram_of_axis_lengths(
            self.major, self.minor, area=self.area * 1e6
        )
        np.testing.assert_allclose(self.bar_heights(ax_a), self.bar_heights(ax_b))

    def test_weighting_changes_the_distribution(self):
        fig_a, ax_a = seg.plot_histogram_of_axis_lengths(self.major, self.minor)
        fig_b, ax_b = seg.plot_histogram_of_axis_lengths(
            self.major, self.minor, area=self.area
        )
        self.assertFalse(
            np.allclose(self.bar_heights(ax_a), self.bar_heights(ax_b))
        )

    def test_nan_does_not_misalign_weights(self):
        # A NaN axis length must drop that grain's area too; previously the
        # arrays were filtered separately and zip() paired sizes with the
        # wrong areas
        major = self.major.copy()
        major[17] = np.nan
        fig_a, ax_a = seg.plot_histogram_of_axis_lengths(
            major, self.minor, area=self.area
        )
        keep = np.ones(len(self.major), dtype=bool)
        keep[17] = False
        fig_b, ax_b = seg.plot_histogram_of_axis_lengths(
            self.major[keep], self.minor[keep], area=self.area[keep]
        )
        np.testing.assert_allclose(self.bar_heights(ax_a), self.bar_heights(ax_b))

    def test_nan_in_area_is_dropped(self):
        area = self.area.copy()
        area[3] = np.nan
        fig, ax = seg.plot_histogram_of_axis_lengths(
            self.major, self.minor, area=area
        )
        heights = self.bar_heights(ax)
        self.assertTrue(np.all(np.isfinite(heights)))
        self.assertAlmostEqual(heights.sum(), len(self.major) - 1, places=6)

    def test_mismatched_area_raises(self):
        with self.assertRaises(ValueError):
            seg.plot_histogram_of_axis_lengths(
                self.major, self.minor, area=self.area[:10]
            )

    def test_all_nan_returns_placeholder(self):
        nans = np.full(5, np.nan)
        fig, ax = seg.plot_histogram_of_axis_lengths(nans, nans)
        self.assertEqual(len(ax.containers), 0)
        self.assertEqual(ax.get_ylabel(), "count")


class TestGetAreaWeightedDistribution(unittest.TestCase):

    def test_deprecation_warning(self):
        with self.assertWarns(DeprecationWarning):
            seg.get_area_weighted_distribution([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    def test_still_replicates(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = seg.get_area_weighted_distribution([1.0, 2.0], [1.0, 3.0])
        # mean area is 2.0, so the copy counts are int(1/1) and int(3/1)
        self.assertEqual(result, [1.0, 2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
