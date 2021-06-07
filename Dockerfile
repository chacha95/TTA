FROM nvidia/cuda:10.1-cudnn7-devel
ENV PYTHONUNBUFFERED 1
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    && apt-get install -y apt-utils ca-certificates wget git sudo python3-opencv python3-dev \
    openssh-client less curl libxtst-dev libxext-dev libxrender-dev libfreetype6-dev libfontconfig1 libgtk2.0-0 libxslt1.1 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

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

# install pycharm tar file
ENV file="/home/appuser/pycharm.tar.gz"
RUN wget "https://download.jetbrains.com/python/pycharm-community-2020.3.1.tar.gz?_ga=2.208079378.1482907034.1608610329-1913990257.1603182848" -O ${file} \
    && tar -xvf ${file} -C /home/appuser && rm -rf ${file}
RUN echo "alias pycharm='bash /home/appuser/pycharm-community-2020.3.1/bin/pycharm.sh'" >> ~/.bashrc
RUN ["/bin/bash", "-c", "source ~/.bashrc"]

# COPY directory
COPY src /home/appuser/src
# pretrained model download
ENTRYPOINT ["/home/appuser/src/script/download_model.sh"]
