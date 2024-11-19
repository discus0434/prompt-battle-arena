SHELL := /bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

build:
	cd ${CURRENT_DIR}/view/txt2img \
	&& npm install \
	&& npm run build \
	&& cd ${CURRENT_DIR} \
	&& pip install -e .

debug:
	cd ${CURRENT_DIR} \
	&& python server.py --debug

launch:
	cd ${CURRENT_DIR} \
	&& LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" \
		python server.py

install-latest-node:
	apt update \
	&& apt install -y nodejs npm \
	&& npm install n -g \
	&& n stable \
	&& apt purge -y nodejs npm \
	&& apt autoremove -y \
	&& ln -sf /usr/local/bin/node /usr/bin/node \

install-dependency:
	pip install --upgrade pip \
	&& pip install triton \
	&& pip install git+https://github.com/huggingface/diffusers.git \
	&& pip install transformers accelerate peft nvitop sentencepiece python-dotenv \
		protobuf pydantic fastapi uvicorn wheel ninja k-diffusion openai \
		aesthetic-predictor-v2-5 nudenet pandas \
	&& pip install -e .

uninstall-old-torch:
	pip uninstall -y nvidia-cublas-cu12
	pip uninstall -y nvidia-cuda-cupti-cu12
	pip uninstall -y nvidia-cuda-nvrtc-cu12
	pip uninstall -y nvidia-cuda-runtime-cu12
	pip uninstall -y nvidia-cudnn-cu12
	pip uninstall -y nvidia-cufft-cu12
	pip uninstall -y nvidia-curand-cu12
	pip uninstall -y nvidia-cusolver-cu12
	pip uninstall -y nvidia-cusparse-cu12
	pip uninstall -y nvidia-nccl-cu12
	pip uninstall -y nvidia-nvjitlink-cu12
	pip uninstall -y nvidia-nvtx-cu12
	pip uninstall -y torch torchvision torchaudio
