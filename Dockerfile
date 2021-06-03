FROM nvidia/cuda:10.1-cudnn7-devel
ENV PYTHONUNBUFFERED 1
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev wget git && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# install pip
ENV PATH="/root/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install torch
RUN pip install --user torch==1.6 torchvision==0.7 -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN pip install pytz

# install fvcore and detectron2
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
ENV FVCORE_CACHE="/tmp"
# install detectron2
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --user -e detectron2_repo

COPY src /home/src