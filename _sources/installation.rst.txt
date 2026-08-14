Installation
------------

.. toctree::
   :caption: Installation

There are three ways to use ``segmenteverygrain``:

1. **Google Colab** — no installation required; good for a quick test drive.
2. **Local conda environment (recommended)** — clone the repository and create a
   conda environment from the provided environment files. This gives you the
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
package itself (from PyPI), plus JupyterLab for running the notebooks.

On Linux or Windows:

.. code-block:: shell

   conda env create -f segmenteverygrain/environment.yml

On macOS (Apple Silicon):

.. code-block:: shell

   conda env create -f segmenteverygrain/environment_macos.yml

This step downloads several gigabytes of packages (TensorFlow, PyTorch, SAM 2),
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

- **U-Net model** (``models/seg_model_smooth_labels.keras``, ~25 MB): included
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

   python -c "import segmenteverygrain; import tensorflow as tf; import torch; print('segmenteverygrain OK'); print('TensorFlow:', tf.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('MPS available:', torch.backends.mps.is_available())"

If this prints the version numbers without errors, the installation works. On
Apple Silicon Macs, ``MPS available: True`` means SAM 2.1 will run on the GPU;
on machines with an NVIDIA GPU, look for ``CUDA available: True``.

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

- **macOS (Apple Silicon):** use ``environment_macos.yml``. It pins
  ``tensorflow<=2.18.1`` and adds ``tensorflow-metal`` so that the U-Net model
  can use the GPU; SAM 2.1 uses the GPU through PyTorch's MPS backend. All code
  in the package auto-detects the available device (CUDA, MPS, or CPU).
- **Windows:** run all commands in the Miniforge Prompt (or Anaconda Prompt).
- **Linux:** ``environment.yml`` installs the standard pip builds of TensorFlow
  and PyTorch; on machines with an NVIDIA GPU these use CUDA automatically.

Troubleshooting
~~~~~~~~~~~~~~~

- ``Permission denied (publickey)`` **when cloning:** you used the SSH address;
  use the HTTPS address given in step 3 above.
- **Environment creation is very slow:** use ``mamba env create -f ...``
  instead of ``conda env create -f ...`` (mamba is included with miniforge).
- ``FileNotFoundError`` **for model files when running the notebook:** make
  sure you started JupyterLab from the repository folder and opened the
  notebook in place — the notebook refers to the models as
  ``../models/seg_model_smooth_labels.keras``, relative to the ``notebooks``
  folder.
- **The interactive editing window does not respond:** the ``GrainPlot``
  interface requires an interactive matplotlib backend. The notebook sets this
  up with the ``%matplotlib qt`` magic (PyQt is included in the conda
  environments); make sure that cell has been run.
- **Old U-Net models fail to load:** as of v0.4.0, the U-Net model outputs raw
  logits, and models trained with v0.3.0 or earlier (e.g., ``seg_model.keras``)
  are incompatible. Use ``seg_model_smooth_labels.keras`` or retrain your
  custom models. Similarly, models saved with Keras 2 do not load under the
  current Keras 3-based package.
