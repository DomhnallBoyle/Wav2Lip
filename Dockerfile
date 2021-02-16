# arguments from the command line
ARG CUDA_BASE_VERSION=10.0
ARG CUDNN_BASE_VERSION=7

# use smaller nvidia cuda runtime base image
# need to use the devel image for building from source
FROM nvidia/cuda:${CUDA_BASE_VERSION}-cudnn${CUDNN_BASE_VERSION}-devel

# arguments from the command line
ARG CUDA_BASE_VERSION=10.0
ARG CUDNN_BASE_VERSION=7

ENV CUDA_BASE_VERSION=${CUDA_BASE_VERSION} \
    CUDNN_BASE_VERSION=${CUDNN_BASE_VERSION}

RUN apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3-pip \
    libsndfile1-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg

COPY . /opt
WORKDIR /opt

RUN python3.7 -m pip install -r requirements.txt

ENTRYPOINT python3.7 server.py
