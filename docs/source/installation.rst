Installation
------------

.. toctree::
   :caption: Installation

There are three ways to use ``segmenteverygrain``:

1. **Google Colab** — no installation required; good for a quick test drive.
2. **Local conda environment (recommended)** — clone the repository and create a
   conda environment from the provided environment file. This gives you the
   notebooks, the example images, and the trained U-Net model in one go.
3. **pip only** — if you just need the library as a dependency in an existing
   environment.

Trying it in Google Colab
~~~~~~~~~~~~~~~~~~~~~~~~~

The `Segment_every_grain_colab.ipynb <https://github.com/zsylvester/segmenteverygrain/blob/main/notebooks/Segment_every_grain_colab.ipynb>`_
notebook has been adjusted so that the segmentation can be run in
`Google Colab <https://colab.research.google.com>`_, without installing anything
on your machine. Note that the interactive editing of the results is not as
smooth in Colab as it is in a local Jupyter session, so for serious work a local
installation is recommended.

Step-by-step: local installation with conda (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These instructions assume no prior setup — only a terminal (macOS/Linux) or the
Miniforge/Anaconda Prompt (Windows).

**1. Install conda (if you do not have it already).**
We recommend `miniforge <https://conda-forge.org/download>`_: unlike the full
Anaconda distribution, it has no licensing restrictions, and it includes
``mamba``, a faster drop-in replacement for the ``conda`` command. If you use
``mamba``, simply replace ``conda`` with ``mamba`` in the commands below.

**2. Install git (if you do not have it already).**
In the terminal (macOS/Linux) or Miniforge Prompt (Windows):

.. code-block:: shell

   conda install git

**3. Clone the repository.**
Cloning (rather than only pip-installing the package) is important because the
repository contains the trained U-Net model and the example images used by the
notebooks. Use the HTTPS address on all platforms:

.. code-block:: shell

   git clone --depth 1 https://github.com/zsylvester/segmenteverygrain.git

.. note::

   If you have seen ``git@github.com:...`` addresses elsewhere: those use SSH
   and fail with a ``Permission denied (publickey)`` error unless you have set
   up SSH keys with GitHub. The HTTPS address above works without any setup.

**4. Create the conda environment.**
The environment files install all dependencies *and* the ``segmenteverygrain``
package itself (from PyPI), plus JupyterLab for running the notebooks. Pick the
file by hardware, not by operating system.

On Apple Silicon Macs, or on any machine without an NVIDIA GPU:

.. code-block:: shell

   conda env create -f segmenteverygrain/environment.yml

On Linux or Windows machines with an NVIDIA GPU (the NVIDIA driver must be
installed; CUDA itself comes with the environment):

.. code-block:: shell

   conda env create -f segmenteverygrain/environment_cuda.yml

This step downloads a few gigabytes of packages (PyTorch, SAM 2, JupyterLab),
so it can take a while.

**5. Activate the environment.**

.. code-block:: shell

   conda activate segmenteverygrain

**6. Launch JupyterLab and open the main notebook.**
Start JupyterLab from the repository folder, so that the relative paths in the
notebooks point to the right places:

.. code-block:: shell

   cd segmenteverygrain
   jupyter lab

Then open ``notebooks/Segment_every_grain.ipynb`` in the JupyterLab file
browser and run the cells from the top.

Model files
~~~~~~~~~~~

Two trained models are needed to run the segmentation workflow:

- **U-Net model** (``models/seg_model_smooth_labels.pt``, ~9 MB): included
  in the repository, so nothing needs to be downloaded if you cloned it.
- **SAM 2.1 checkpoint** (``models/sam2.1_hiera_large.pt``, ~860 MB): *not*
  included in the repository. The first cells of the
  ``Segment_every_grain.ipynb`` notebook download it automatically into the
  ``models`` folder. You can also download it manually from
  `this link <https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt>`_
  and place it in the ``models`` folder.

Verifying the installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

With the ``segmenteverygrain`` environment activated, run:

.. code-block:: shell

   python -c "import segmenteverygrain as seg; import torch; print('segmenteverygrain', seg.__version__, 'OK'); print('PyTorch:', torch.__version__); print('device:', seg.get_device())"

If this prints the version numbers without errors, the installation works.
``device: mps`` (Apple Silicon) or ``device: cuda`` (NVIDIA) means that both
the U-Net and SAM 2.1 will run on the GPU; ``device: cpu`` means everything
runs on the CPU, which works but is slower.

Installing with pip only
~~~~~~~~~~~~~~~~~~~~~~~~

If you only need ``segmenteverygrain`` as a library in an existing environment:

.. code-block:: shell

   pip install segmenteverygrain

Python 3.10 or higher is required. Note that a pip install does *not* include
the notebooks, the example images, or the U-Net model file — clone the
repository (see above) to get those.

Development install
~~~~~~~~~~~~~~~~~~~

To work on the package itself, install the cloned repository in editable mode
inside the ``segmenteverygrain`` environment:

.. code-block:: shell

   cd segmenteverygrain
   pip install -e .

Platform notes
~~~~~~~~~~~~~~

- **All platforms:** the only deep-learning framework is PyTorch, which the
  environment files install from conda-forge. All code in the package
  auto-detects the available device (CUDA, MPS, or CPU).
- **macOS (Apple Silicon):** use ``environment.yml``; both the U-Net and
  SAM 2.1 run on the GPU through PyTorch's MPS backend, and nothing extra needs
  to be installed.
- **Linux and Windows with an NVIDIA GPU:** use ``environment_cuda.yml``,
  which requests the CUDA build of PyTorch (``pytorch-gpu``). If you already
  created the environment from ``environment.yml``, switch to the GPU build
  with ``conda install -c conda-forge pytorch-gpu``. The official PyTorch
  ``pip`` wheels (https://pytorch.org/get-started/locally/) work as well.
- **Windows:** run all commands in the Miniforge Prompt (or Anaconda Prompt).

Troubleshooting
~~~~~~~~~~~~~~~

- ``Permission denied (publickey)`` **when cloning:** you used the SSH address;
  use the HTTPS address given in step 3 above.
- **Environment creation is very slow:** use ``mamba env create -f ...``
  instead of ``conda env create -f ...`` (mamba is included with miniforge).
- ``FileNotFoundError`` **for model files when running the notebook:** make
  sure you started JupyterLab from the repository folder and opened the
  notebook in place — the notebook refers to the models as
  ``../models/seg_model_smooth_labels.pt``, relative to the ``notebooks``
  folder.
- ``OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
  initialized`` **(macOS):** two copies of the OpenMP runtime are loaded,
  typically because PyTorch was installed with ``pip`` into a conda environment
  whose numpy/scipy come from conda-forge. Install PyTorch from conda-forge
  instead (``conda install -c conda-forge pytorch torchvision``, as
  ``environment.yml`` does), or import ``segmenteverygrain`` before ``torch``
  in your scripts. Setting ``KMP_DUPLICATE_LIB_OK=TRUE`` silences the error but
  is not recommended.
- **The interactive editing window does not respond:** the ``GrainPlot``
  interface requires an interactive matplotlib backend. The notebook sets this
  up with the ``%matplotlib qt`` magic (PyQt is included in the conda
  environments); make sure that cell has been run.
- **Old U-Net models fail to load:** the U-Net is now a PyTorch model
  (``.pt`` file). A custom or fine-tuned Keras model (``.keras`` file) can be
  converted once with
  ``seg.convert_keras_model("my_model.keras", "my_model.pt")`` while Keras and
  TensorFlow are installed. This works for models from any previous version,
  including those trained with v0.3.0 or earlier that have the softmax built
  into the model.
