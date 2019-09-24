#!/bin/bash
# Docker Installation Instructions as depicted on https://docs.docker.com/install/linux/docker-ce/ubuntu/#prerequisites

# For installation of NVIDIA-Drivers uncomment following lines:
# sudo apt purge nvidia*
# sudo add-apt-repository ppa:graphics-drivers
# sudo apt install nvidia-drivers-410
# sudo apt-mark hold nvidia-drivers-410


# install prerequisites
sudo apt remove docker docker-engine docker.io containerd runc
sudo apt update
sudo apt install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common


# install docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io


# usage as non-root user
sudo groupadd docker
sudo usermod -aG docker $USER


# install gpu support with nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker


# NEXT STEPS: 
# Build Docker image
# $ docker build -t stascheit/gpgpu .
