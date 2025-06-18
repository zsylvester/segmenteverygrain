# Standard imports
import math
import unittest
# Pip imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
# Local imports
import segmenteverygrain as seg
import segmenteverygrain.interactions as si


# Geometry calculations
def _azimuth(point1: tuple, point2: tuple) -> float:
    ''' Azimuth between 2 points (interval 0 - 180) '''
    return np.arctan2(point2[0] - point1[0], point2[1] - point1[1]) / 2


def dist(a: tuple, b: tuple) -> float:
    ''' Distance between points '''
    return math.hypot(b[0] - a[0], b[1] - a[1])


def azimuth(mrr: shapely.Polygon) -> float:
    ''' Azimuth of minimum_rotated_rectangle '''
    bbox = list(mrr.exterior.coords)
    axis1 = dist(bbox[0], bbox[3])
    axis2 = dist(bbox[0], bbox[1])
    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])
    return az


# Comparison methods
def absolute(a, b):
    return abs(a - b)


def relative(a, b):
    return 100 * abs((a - b) / (a + b)) / 2


def absolute_angle(a, b):
    diff = absolute(a, b)
    return min(diff, np.pi - diff) * 180. / np.pi


class Comparison():
    def __init__(self, f: object, tolerance: float):
        # Comparison function
        self.f = f
        # Tolerance for comparison
        self.tolerance = tolerance


metrics = {
    'centroid-0': Comparison(absolute, 2.),
    'centroid-1': Comparison(absolute, 2.),
    'area': Comparison(relative, 2.),
    'perimeter': Comparison(relative, 2.),
    'orientation': Comparison(absolute_angle, 10.),
    'major_axis_length': Comparison(relative, 1.),
    'minor_axis_length': Comparison(relative, 2.),
    # 'max_intensity-0': Comparison(absolute, 5.),
    # 'min_intensity-0': Comparison(absolute, 10.),
    'mean_intensity-0': Comparison(relative, 2.),
    # 'max_intensity-1': Comparison(absolute, 5.),
    # 'min_intensity-1': Comparison(absolute, 10.),
    'mean_intensity-1': Comparison(relative, 2.),
    # 'max_intensity-2': Comparison(absolute, 5.),
    # 'min_intensity-2': Comparison(absolute, 10.),
    'mean_intensity-2': Comparison(relative, 2.)
}


class TestPolygonMeasurement(unittest.TestCase):
    ''' Test polygon measurement functions. '''

    @classmethod
    def setUpClass(cls):
        # Create image with random color
        cls.rgb = np.random.randint(0, 255, size=3, dtype=np.uint8)
        cls.image = np.zeros((100, 100, 3), dtype=np.uint8)
        cls.image[:, :] = cls.rgb
        # Create polygons
        cls.polygons = [shapely.Polygon(coords) for coords in [
            [[10, 10], [25, 15], [25, 25], [15, 25]],
            [[30, 30], [40, 30], [40, 40], [30, 40]]
        ]]

    def test_measure_polygon(self):
        for poly in self.polygons:
            data = si.measure_polygon(poly)
            # Check the centroid
            self.assertEqual(data['centroid'], poly.centroid.xy[::-1])
            # Check the area
            self.assertEqual(data['area'], poly.area)

    def test_measure_ellipse(self):
        for poly in self.polygons:
            # Get ellipse metrics
            data = si.measure_ellipse(si.measure_polygon(poly))
            # Check that the orientation matches the minimum rotated rectangle
            angle = azimuth(poly.oriented_envelope)
            self.assertEqual(data['orientation'], angle)

    def test_measure_color(self):
        for poly in self.polygons:
            # Get color metrics
            data = si.measure_color(self.image, poly)
            # Check the mean color intensities
            r, g, b = self.rgb
            self.assertEqual(data['mean_intensity-0'], r)
            self.assertEqual(data['mean_intensity-1'], g)
            self.assertEqual(data['mean_intensity-2'], b)


class TestGrainObject(unittest.TestCase):
    ''' Test Grain object methods. '''

    @classmethod
    def setUpClass(cls):
        # Create image with random solid color
        cls.rgb = np.random.randint(0, 255, size=3, dtype=np.uint8)
        cls.image = np.zeros((100, 100, 3), dtype=np.uint8)
        cls.image[:, :] = cls.rgb

        # Create a mock Axes object with patches
        cls.fig, cls.ax = plt.subplots()
        cls.ax.imshow(cls.image)
        cls.patches = [
            mpl.patches.Polygon(
                np.array([[10, 10], [25, 15], [25, 25], [15, 25]]), closed=True),
            mpl.patches.Polygon(
                np.array([[30, 30], [40, 30], [40, 40], [30, 40]]), closed=True)
        ]
        for patch in cls.patches:
            cls.ax.add_patch(patch)

        # Create Shapely polygons from patches
        cls.polygons = [shapely.Polygon(p.get_xy()) for p in cls.patches]

        # Create and draw Grain objects from polygons
        cls.grains = si.polygons_to_grains(cls.polygons, image=cls.image)
        for grain in cls.grains:
            grain.measure()
            grain.draw_patch(cls.ax)

    def test_init(self):
        # Check the number of grains
        self.assertEqual(len(self.grains), len(self.patches))
        # Check the type of grains
        for grain in self.grains:
            self.assertIsInstance(grain, si.Grain)

    def test_drawing(self):
        for grain, patch in zip(self.grains, self.patches):
            # Verify coordinates
            for grain_xy, patch_xy in zip(grain.patch.get_xy(), patch.get_xy()):
                self.assertTrue((grain_xy == patch_xy).all())

    def test_measure(self):
        ''' Verify grain measurements. '''
        for grain, poly in zip(self.grains, self.polygons):
            data = grain.data
            # Centroid
            centroid = poly.centroid
            self.assertEqual(data['centroid-1'], centroid.x)
            self.assertEqual(data['centroid-0'], centroid.y)
            # Area
            self.assertEqual(data['area'], poly.area)
            # Perimeter
            self.assertEqual(data['perimeter'], poly.length)
            # Orientation
            angle = azimuth(poly.oriented_envelope)
            self.assertEqual(data['orientation'], angle)
            # Color
            r, g, b = self.rgb
            self.assertEqual(data['mean_intensity-0'], r)
            self.assertEqual(data['mean_intensity-1'], g)
            self.assertEqual(data['mean_intensity-2'], b)

    def test_rescale(self):
        pass


class TestGrainPlot(unittest.TestCase):
    ''' Test GrainPlot object methods. '''

    @classmethod
    def setUpClass(cls):
        cls.image = si.load_image('examples/torrey_pines.jpg')
        cls.grains = si.load_grains(
            'examples/interactive_edit/torrey_pines_grains.geojson',
            image=cls.image)
        cls.plot = si.GrainPlot(cls.grains, cls.image)

    def test_create_plot(self):
        # Check if the plot is created
        plot = self.plot
        self.assertIsInstance(plot.fig, plt.Figure)
        self.assertIsInstance(plot.ax, plt.Axes)
        # Check if the image is displayed
        self.assertIsInstance(plot.image, np.ndarray)
        self.assertIsInstance(plot.display_image, np.ndarray)
        # Verify that image isn't downscaled
        self.assertEqual(plot.image.shape, plot.display_image.shape)

    def test_measure_methods(self):
        # Compare old/new measurement methods to ensure they reasonably agree
        for grain in self.grains:
            a = grain.measure(raster=False)
            b = grain.measure(raster=True)
            for metric, comparison in metrics.items():
                error = comparison.f(a[metric], b[metric])
                self.assertTrue(error < comparison.tolerance)


class TestDownscaledGrainPlot(unittest.TestCase):
    ''' Test GrainPlot object methods with downscaled image. '''

    @classmethod
    def setUpClass(cls):
        cls.image = si.load_image('examples/torrey_pines.jpg')
        cls.grains = si.load_grains(
            'examples/interactive_edit/torrey_pines_grains.geojson',
            image=cls.image)
        cls.testdata = cls.grains[0].measure()
        cls.testxy = cls.grains[0].xy
        cls.max_dim = 320
        cls.plot = si.GrainPlot(
            cls.grains, cls.image, image_max_size=(cls.max_dim, cls.max_dim))

    def test_create_plot(self):
        # Check if the plot is created
        plot = self.plot
        self.assertIsInstance(plot.fig, plt.Figure)
        self.assertIsInstance(plot.ax, plt.Axes)
        # Check that display image exists and is downscaled
        self.assertIsInstance(plot.display_image, np.ndarray)
        self.assertNotEqual(plot.image.shape, plot.display_image.shape)
        self.assertEqual(max(plot.display_image.shape), self.max_dim)
        # Check that grain sizes are downscaled
        pass

    def test_measurement(self):
        # Ensure measurements are correctly calculated
        for grain in self.grains:
            data = grain.data
            self.assertIn('area', data)
            self.assertIn('perimeter', data)
            self.assertIn('centroid-0', data)
            self.assertIn('centroid-1', data)

    def test_grain_scale(self):
        # Verify grain coordinates
        testdata = self.testdata
        graindata = self.grains[0].data
        self.assertEqual(testdata['area'], graindata['area'])
        plotdata = self.plot.grains[0].data
        self.assertEqual(testdata['area'], plotdata['area'])
        # Verify grain data
        testxy = self.testxy
        grainxy = self.grains[0].xy
        self.assertTrue((testxy == grainxy).all())
        plotxy = self.plot.grains[0].xy
        self.assertTrue((testxy == plotxy).all())


if __name__ == '__main__':
    unittest.main()
