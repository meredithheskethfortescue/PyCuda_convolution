#!/usr/bin/env python3
"""Host - PyCuda wrapper for a simple convolution on the GPU"""

import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
import pycuda.autoinit

'''Global CUDA setup'''
# read and compile cuda file
with open('./kernel.cu', 'r') as f:
    mod = SourceModule(f.read())

# bind cuda textures to python
TEX_IMG = mod.get_texref('tex_img')
TEX_KERNEL = mod.get_texref('tex_kernel')

# bind cuda kernel function to python
CUDA_CONVOLUTION = mod.get_function('convolve')

del mod  # module not needed anymore in global space


def wrap_cuda_convolution(img, kernel) -> np.ndarray:
    # check kernel and get radius
    kernel_width, kernel_height = np.int32(np.shape(kernel))
    assert kernel_height % 2 or kernel_width % 2, "Kernel shape does not consist of odd numbers!"
    assert kernel_height == kernel_width, "Kernel is not a square!"

    # cast input to float32
    img_in = img.astype(np.float32)
    kernel_cpu = kernel.astype(np.float32)

    # pass data to cuda texture
    cuda.matrix_to_texref(img_in, TEX_IMG, order='C')
    cuda.matrix_to_texref(kernel_cpu, TEX_KERNEL, order='C')

    # setup output
    img_out = np.zeros_like(img, dtype=np.float32)

    # setup grid
    img_height, img_width = np.shape(img_in)
    blocksize = 32
    grid = (int(np.ceil(img_width / blocksize)),
            int(np.ceil(img_height / blocksize)),
            1)

    kernel_radius = kernel_width // 2
    CUDA_CONVOLUTION(np.int32(img_width),
                     np.int32(img_height),
                     np.int32(kernel_radius),
                     cuda.Out(img_out),
                     texrefs=[TEX_IMG, TEX_KERNEL],
                     block=(blocksize, blocksize, 1),
                     grid=grid)

    return img_out


if __name__ == '__main__':
    # setup test image matrix
    # shape = (1000, 1000)
    # img = np.random.rand(*shape).astype(np.float32)
    # img = np.ones(shape)
    img = np.array([[1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1]],
                   dtype=np.float32)

    # define kernel matrix
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    # normalize kernel
    kernel = kernel / np.sum(kernel)

    out = wrap_cuda_convolution(img, kernel)
    print(img, "\n")
    print(out)
