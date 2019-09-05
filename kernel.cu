/// Device - CUDA Kernel: Convolution
/*
Index terminology:
x = row = axis 1 = width = horizontal
y = col = axis 0 = height = vertical
z = channel = axis 2 = depth
*/

#include <pycuda-helpers.hpp>


texture<float, cudaTextureType2D> tex_img;
texture<float, cudaTextureType2D> tex_kernel;


__global__ void convolve(const int input_width,
                         const int input_height,
                         const int kernel_radius,
                         float *output)
{
    // get coordinates by thread index
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // check if in range
    if((col < input_height) && (row < input_width))
    {
        float value = 0.0f;

        // iterate over the kernel matrix
        for(int y = -kernel_radius; y <= kernel_radius; ++y)
        {
            for(int x = -kernel_radius; x <= kernel_radius; ++x)
            {
                // img value * kernel value
                value += tex2D(tex_img, row + x, col + y) * tex2D(tex_kernel, x + kernel_radius, y + kernel_radius);
            }
        }

        // todo: output to texture?
        int idx = col * input_width + row;
        output[idx] = value;
    }
}
