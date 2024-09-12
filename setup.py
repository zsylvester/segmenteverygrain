import setuptools

long_description = """\
'segmenteverygrain' is a Python package that aims to detect grains (or grain-like objects) in images. 
The goal is to develop an ML model that does a reasonably good job at detecting most of the grains in a photo, 
so that it will be useful for determining grain size and grain shape, a common task in geomorphology and sedimentary geology.
"""

setuptools.setup(
    name="segmenteverygrain",
    version="0.1.8",
    author="Zoltan Sylvester",
    author_email="zoltan.sylvester@beg.utexas.edu",
    description="a SAM-based model for segmenting grains in images of grains",
    keywords = 'sedimentology, geomorphology, grain size, segment anything model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zsylvester/segmenteverygrain",
    packages=['segmenteverygrain'],
    install_requires=['numpy','matplotlib',
        'scipy','pillow','scikit-image','tqdm','opencv-python',
        'networkx','rasterio','shapely','tensorflow','segment-anything'],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
