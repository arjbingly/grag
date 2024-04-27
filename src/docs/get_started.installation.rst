Installation
===============

To install the package

* ``pip install grag``


Note that since this package is still under development, to check out the latest features.

* ``git clone`` the repository
* ``pip install .`` from the repository
* *For Developers*: ``pip install -e .``


GPU and Hardware acceleration support
-----------------------------------------

GRAG uses ``llama.cpp`` to inference LLMs locally. It supports a number of hardware acceleration backends to speed up
inference as well as backend specific options. See the
`llama.cpp README <https://github.com/ggerganov/llama.cpp#build>`_ for a full list.

Below are some of the supported backends.

* Note that the below instructions are tailored for Linux and MACOS users, Windows users should add ``$env:`` before
  defining environment variables.

.. code-block:: console
    $env:CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
    pip install llama-cpp-python

#. OpenBLAS (CPU)

To install with OpenBLAS, set the `LLAMA_BLAS` and `LLAMA_BLAS_VENDOR` environment variables before installing:

.. code-block:: console

    CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
    pip install llama-cpp-python


#. CUDA (Nvidia-GPU)

To install with CUDA support, set the `LLAMA_CUDA=on` environment variable before installing:

.. code-block:: console

    CMAKE_ARGS="-DLLAMA_CUDA=on"
    pip install grag


#. Metal (MacOS)

To install with Metal (MPS), set the `LLAMA_METAL=on` environment variable before installing:

.. code-block:: console

    CMAKE_ARGS="-DLLAMA_METAL=on"
    pip install grag


#. CLBlast (OpenCL)

To install with CLBlast, set the `LLAMA_CLBLAST=on` environment variable before installing:

.. code-block:: console

    CMAKE_ARGS="-DLLAMA_CLBLAST=on"
    pip install grag


#. hipBLAS (AMD ROCm)</summary>

To install with hipBLAS / ROCm support for AMD cards, set the `LLAMA_HIPBLAS=on` environment variable before installing:

.. code-block:: console

    CMAKE_ARGS="-DLLAMA_HIPBLAS=on"
    pip install grag


#. Vulkan

To install with Vulkan support, set the `LLAMA_VULKAN=on` environment variable before installing:

.. code-block:: console

    CMAKE_ARGS="-DLLAMA_VULKAN=on"
    pip install grag


#. Kompute

To install with Kompute support, set the `LLAMA_KOMPUTE=on` environment variable before installing:

.. code-block:: console

    CMAKE_ARGS="-DLLAMA_KOMPUTE=on"
    pip install grag


#. SYCL

To install with SYCL support, set the `LLAMA_SYCL=on` environment variable before installing:

.. code-block:: console

    CMAKE_ARGS="-DLLAMA_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx"
    pip install grag


For more details and troubleshooting please refer  `llama-cpp-python <https://github.com/abetlen/llama-cpp-python>`_

Upgrading and Reinstalling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In case you want to upgrade to change hardware acceleration support, or did not install with hardware acceleration
support, simply rebuilt ``llama-cpp-python`` using the instructions below.

To upgrade and rebuild ``llama-cpp-python`` add ``--upgrade --force-reinstall --no-cache-dir``
flags to the pip install command along with the necessary environment variables listed above
to ensure the package is rebuilt from source.

Example usage for reinstalling with CUDA support:

.. code-block:: console

    CMAKE_ARGS="-DLLAMA_CUDA=on"
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir



`Note that one does not have to reinstall the grag package`
