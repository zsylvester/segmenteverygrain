# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from unittest import mock

# Mock open3d because it fails to build in readthedocs
MOCK_MODULES = ["numpy", "matplotlib", "matplotlib.pyplot", "pandas", "scipy", "skimage", "scikit-learn", "tqdm", 
    "rtree", "itertools", "networkx", "rasterio", "rasterio.features", "shapely", "tensorflow", "segment-anything",
    "skimage.measure", "skimage.morphology", "skimage.segmentation", "skimage.feature", "shapely.geometry", "shapely.affinity",
    "scipy.ndimage", "sklearn.cluster", "tensorflow.keras.models", "tensorflow.keras.layers", 
    "tensorflow.keras.preprocessing.image", "segment_anything"]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# Example of setting return values for specific methods that need to be iterable
mock_numpy = sys.modules['numpy']
mock_numpy.array.return_value = mock.Mock()
mock_numpy.array.return_value.__iter__ = lambda x: iter([])

mock_pandas = sys.modules['pandas']
mock_pandas.DataFrame.return_value = mock.Mock()
mock_pandas.DataFrame.return_value.__iter__ = lambda x: iter([])

# Mock common iterable methods for other modules
for mod_name in MOCK_MODULES:
    mock_module = sys.modules[mod_name]
    mock_module.items.return_value = iter([])
    mock_module.keys.return_value = iter([])
    mock_module.values.return_value = iter([])


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'segmenteverygrain'
copyright = '2024, Zoltan Sylvester'
author = 'Zoltan Sylvester'
release = '0.1.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
