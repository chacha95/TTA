FROM nvidia/cuda:10.1-cudnn7-devel
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    && apt-get install -y apt-utils ca-certificates wget unzip git sudo python3-opencv python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --user \
	&& rm get-pip.py

# install torch
RUN pip install --no-cache-dir --user torch==1.8 torchvision==0.9 -f https://download.pytorch.org/whl/cu101/torch_stable.html

# install fvcore and detectron2
ENV FORCE_CUDA="1"
ENV FVCORE_CACHE="/tmp"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --no-cache-dir --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install --no-cache-dir --user -e detectron2_repo

# COPY directory
COPY src /home/appuser/src
# pretrained model download
RUN sudo wget "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl" \
     -O "/home/appuser/src/model/faster_rcnn_R_50_FPN_3x.pkl"

# install COCO2017 dataset
ENV COCO_DIR="/home/appuser/dataset/COCO2017"
ENV COCO_IMG="${COCO_DIR}/val2017.zip"
ENV COCO_ANNOTATION="${COCO_DIR}/annotations_trainval2017.zip"
RUN sudo mkdir -p "/home/appuser/dataset/COCO2017"
RUN sudo wget "http://images.cocodataset.org/zips/val2017.zip" -O "${COCO_IMG}" \
    && sudo unzip "${COCO_IMG}" -d "${COCO_DIR}" && sudo rm -rf "${COCO_IMG}"
RUN sudo wget "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -O "${COCO_ANNOTATION}" \
    && sudo unzip "${COCO_ANNOTATION}" -d "${COCO_DIR}" && sudo rm -rf "${COCO_ANNOTATION}"
