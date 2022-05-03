============
Laminocupy-cli
============

**Laminocupy-cli** is a command-line interface for GPU reconstruction of Laminographic data in 32-bit precision. All preprocessing operations are implemented on GPU with using cupy library. The back-projection method is implemented using direct linear interpoaltion with CUDA C++ and python wrappers. Ring removal is implemented with using the pytorch wavelet library on GPU.

The package supports two types of reconstructions: manual center search (option '--reconstruction-type try') and whole volume reconstruction (option '--reconstruction-type full'). 


=============
Documentation
=============

**Laminocupy-cli**  documentation is available here available `here <https://laminocupy.readthedocs.io/en/latest/>`_
