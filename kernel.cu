/// Device - CUDA Kernel: Convolution
/// Every pixel in the image gets assigned to one thread.
/// Padding is handled by using the `tex2D` function in CUDA.
/*
Index terminology:
z = channel = axis 0 = depth = normal
x = col = axis 1 = width = horizontal
y = row = axis 2 = height = vertical
*/

#include <pycuda-helpers.hpp>


texture<float, cudaTextureType2D> tex_img;
texture<float, cudaTextureType2D> tex_kernel;


__global__ void convolve(const int input_width,
                         const int input_height,
                         const int kernel_radius,
                         float *output) {
    // :input_width, input_height: Size of the input matrix is required to calculate the tread indices.
    // :kernel_radius: By requesting the radius instead of the diameter of the kernel matrix, errors due to an even kernel width/height can't appear.
    // :output: Pointer to the output matrix.

    // get coordinates by thread index
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // check if in range
    if ((col < input_height) && (row < input_width)) {
        float value = 0.0f;

        // iterate over the kernel matrix
        for (int y = -kernel_radius; y <= kernel_radius; ++y) {
            for (int x = -kernel_radius; x <= kernel_radius; ++x) {
                // img value * kernel value
                value += tex2D(tex_img, row + x, col + y) * tex2D(tex_kernel, x + kernel_radius, y + kernel_radius);
            }
        }

        // write to output
        int idx = col * input_width + row;
        output[idx] = value;
    }
}
