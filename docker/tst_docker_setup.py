#!/usr/bin/env python3
"""Test System Setup
Check for correct versions and dependencies:
CUDA Driver: 10.0
CuDNN: 7.4

@author: Raphael Stascheit
"""
import unittest


class SystemSetup(unittest.TestCase):
    def test_os_version(self):
        import platform
        print(platform.platform())
        print()

    def test_python_version(self):
        import sys
        print("Python Interpreter Version:", sys.version)
        print()
        # self.assertEqual(sys.version_info, (3, 5, 2, 'final', 0), msg="Using Python " + '.'.join(str(e) for e in sys.version_info[0:3]) + " instead of 3.5.2")
        self.assertEqual(sys.version_info, (3, 6, 8, 'final', 0), msg="Using Python " + '.'.join(str(e) for e in sys.version_info[0:3]) + " instead of 3.6.8")

    def test_pycuda(self):
        import numpy as np
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        a = np.random.randn(4, 4)
        a = a.astype(np.float32)
        a_gpu = cuda.mem_alloc(a.nbytes)

        # As a last step, we need to transfer the data to the GPU:
        cuda.memcpy_htod(a_gpu, a)  # memory copy from host to device

        # load and apply CUDA kernel
        mod = SourceModule(
            """
            __global__ void doublify(float *a)
            {
                int idx = threadIdx.x + threadIdx.y*4;
                a[idx] *= 2;
            }
            """
        )

        func = mod.get_function("doublify")
        func(a_gpu, block=(4, 4, 1))

        # fetch data from GPU
        a_doubled = np.empty_like(a)
        cuda.memcpy_dtoh(a_doubled, a_gpu)  # memory copy from device to host

        self.assertEqual((2 * a).all(), a_doubled.all())

    def test_cuda_version(self):
        import pycuda.driver as drv
        self.assertEqual(drv.get_version(), (10, 0, 0))

    def test_tensorflow_available(self):
        import tensorflow as tf

    def test_tensorflow_has_access_to_gpu(self):
        import tensorflow as tf
        self.assertTrue(tf.test.is_gpu_available(), msg="No GPU detected!")


if __name__ == '__main__':
    print("Hello, user!\n")
    unittest.main()
