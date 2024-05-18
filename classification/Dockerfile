ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

WORKDIR /mmpretrain

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MIM
RUN pip install openmim

# Install MMPretrain
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpretrain.git
WORKDIR ./mmpretrain
RUN mim install --no-cache-dir -e .

#JDK
apt update
apt install default-jdk

# ONNX
RUN pip install onnxruntime-gpu==1.12.0 pycuda
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.0/onnxruntime-linux-x64-1.12.0.tgz
RUN tar -zxvf onnxruntime-linux-x64-1.12.0.tgz && rm onnxruntime-linux-x64-1.12.0.tgz
ENV ONNXRUNTIME_DIR=/openmmlab/onnxruntime-linux-x64-1.12.0
ENV LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH


# TORCHSERVE
RUN pip install torchserve torch-model-archiver nvgpu
ADD ./torchserve /torchserve

# Convert models
ADD ./convert_model /mmpretrain/convert_model

# Upload model
RUN apt-get install -y curl unzip
RUN curl -LO https://huggingface.co/Vades/GravelClassificationModels/resolve/main/GravelClassificationDeit3s.zip \
    && unzip GravelClassificationModels.zip \
    && mv GravelClassificationDeit3s.mar /torchserve/model_store/GravelClassificationDeit3s.mar \
    && mv GravelClassificationDeit3s.pth /mmpretrain/convert_model/models/GravelClassificationDeit3s.pth \
    && rm GravelClassificationModels.zip

EXPOSE 554 8080 8081 8082 8888
