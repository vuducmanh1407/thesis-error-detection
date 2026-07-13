FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include

RUN apt-get update && apt-get install -y apt-file && apt upgrade -y
RUN apt install -y build-essential curl git cmake pkg-config checkinstall

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
RUN conda update -y -n base -c defaults conda
RUN conda install -y pip

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy