=======
Install
=======


1. Create environment with necessary dependencies

::

    (base)$ conda create -n laminocupy -c conda-forge python=3.9 dxchange cupy scikit-build swig pywavelets numexpr astropy olefile opencv
    (base)$ conda activate laminocupy
    (laminocupy)$ pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

2. Install the pytorch pywavelets package for ring removal

::

    (laminocupy)$ git clone https://github.com/fbcotter/pytorch_wavelets
    (laminocupy)$ cd pytorch_wavelets
    (laminocupy)$ pip install .
    (laminocupy)$ cd -

3. Set path to the nvcc profiler (e.g. /local/cuda-11.4/bin/nvcc ) and install laminocupy

::

    (laminocupy)$ export CUDACXX=/local/cuda-11.4/bin/nvcc 
    (laminocupy)$ git clone https://github.com/nikitinvv/laminocupy-cli
    (laminocupy)$ cd laminocupy-cli
    (laminocupy)$ python setup.py install 


Update
======

**laminocupy-cli** is constantly updated to include new features. To update your locally installed version::

    (laminocupy)$ cd laminocupy-cli
    (laminocupy)$ git pull
    (laminocupy)$ python setup.py install
