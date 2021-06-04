FROM nvidia/cuda:10.1-cudnn7-devel
ENV PYTHONUNBUFFERED 1
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    && apt-get install -y apt-utils ca-certificates python3-dev wget git sudo python3-opencv \
	&& python3 python3-dev python3-dev python3-setuptools gcc git openssh-client less curl libxtst-dev \
	libxext-dev libxrender-dev libfreetype6-dev libfontconfig1 libgtk2.0-0 libxslt1.1 libxxf86vm1 \
    rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser/src

ENV PATH="/home/appuser/.local/bin:${PATH}"
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
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --user -e detectron2_repo

COPY src /home/appuser/src
# pretrained model download
ENTRYPOINT ["/home/appuser/src/script/download_model.sh"]
# install pycharm
CMD ["/home/appuser/src/script/pycharm_install.sh"]