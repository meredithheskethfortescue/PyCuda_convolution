FROM tensorflow/tensorflow:latest-gpu-py3
#FROM alpine:3.7

# Personal development setup for GPU accellerated computing purposes
# Usage with nvidia-docker
# |-- OS: Ubuntu 18.04 --> Alpine-Linux?
# |-- Python: 3.5.2
# |-- PyCUDA
# |   |-- CUDA 10.0
# |   |-- Driver Version: 410.104
# |   |-- CUDNN: 7.4.1.5-1
# |-- Tensorflow-GPU: 1.13.1 --> compile from source?

LABEL maintainer="Raphael Stascheit"

# Install prerequisites
RUN apt update && apt install -y \
    wget \
    nano \
    cmake \
    libboost-all-dev \
    libboost-python-dev \
    libboost-thread-dev \
    python3-tk

# Additional Python Packages
RUN pip3 install --upgrade pip
RUN pip3 install \
    matplotlib \
    scipy \
    scikit-image \
    numba \
    Pillow \
    pandas \
    tqdm


# CUDA-Toolkit 10.0
RUN apt install -y cuda-toolkit-10-0

# PyCUDA (with pip)
RUN pip3 install pycuda

# TODO: PyCUDA (compile from source)
# RUN wget https://files.pythonhosted.org/packages/c8/35/130ac8867b30f9c6ae699b689633a803a73787533f41e52afcf28b75bd81/pycuda-2019.1.1.tar.gz
# # COPY pycuda-2019.1.1.tar.gz /root/pycuda-2019.1.1.tar.gz
# RUN tar xzf /root/pycuda-2019.1.1.tar.gz -C /root && \
#     cd /root/pycuda-2019.1.1 && \
#     ./configure.py \
#         --cuda-root=/usr/local/cuda \
#         --cudadrv-lib-dir=/usr/lib/x86_64-linux-gnu \
#         --boost-inc-dir=/usr/include \
#         --boost-lib-dir=/usr/lib \
#         --boost-python-libname=boost_python \
#         --boost-thread-libname=boost_thread \
#         # not existing option:
#         # --no-use-shipped-boost && \
#     make -j8 /root/pycuda-2019.1.1 && \
#     python3 setup.py install && \
#     pip3 install . && \
#     rm /root/pycuda* -rf
