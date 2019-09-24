#!/bin/bash
# Launch tests on an instance of the docker image

# If not done yet, build docker image with
# $ docker build -t stascheit/gpgpu .

# run test
docker run --gpus all -v "$(pwd):/workspace" -it stascheit/gpgpu:latest /workspace/tst_docker_setup.py
