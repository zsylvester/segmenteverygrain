import setuptools

long_description = """\

"""

setuptools.setup(
    name="segmenteverygrain",
    version="0.0.2",
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
