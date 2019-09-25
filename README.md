# Convolution using Python 3 and CUDA for GPU-Acceleration

Minimal example of a CUDA kernel that applies a convolution to a 2D matrix using tex2D. The kernel is written in pure CUDA-C but launched from Python 3 by using the PyCuda API.

`kernel.cu` contains the CUDA kernel which is read by the `main.py` file.

A possible setup with docker is described in the `docker` directory.
`get_docker.sh` installs docker and it's dependencies for usage with GPU.
After building a docker image with 
```bash
docker build -t stascheit/gpgpu .
```
the `launch_tst.sh` can be executed to run unittests (in `tst_docker_setup`) to the built docker image in order to validate the installation.
