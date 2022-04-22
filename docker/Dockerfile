ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

COPY . /general-3d-humans/

RUN apt-get update && apt-get install -y git \
	ninja-build \
	libglib2.0-0 \
	libsm6 \
	libxrender-dev \
	libxext6 \
	libgl1-mesa-glx \
	python3-pip \
	vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.7 -m pip install \
	cython \
	xtcocotools \
	cycler==0.10.0 \
	decorator==4.4.0 \
	easydict==1.9 \
	imageio==2.5.0 \
	kiwisolver==1.1.0 \
	matplotlib==3.1.1 \
	networkx==2.3 \
	numpy \
	opencv-python==4.1.2.30 \
	Pillow==6.2.0 \
	protobuf==3.10.0 \
	pyparsing==2.4.2 \
	python-dateutil==2.8.0 \
	PyWavelets==1.0.3 \
	scikit-image==0.15.0 \
	scipy==1.3.1 \
	six==1.12.0 \
	tensorboard \
	kornia \
	h5py==2.10.0 \
	scikit-learn \
	plotly \
	kaleido \
	pandas \
	seaborn \
	poseval@git+https://github.com/svenkreiss/poseval.git

WORKDIR /general-3d-humans

