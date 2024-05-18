ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

WORKDIR /openmmlab

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1" \
    DEBIAN_FRONTEND=noninteractive \
    APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install the required packages
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
      ca-certificates openjdk-11-jre-headless git wget vim curl ffmpeg libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx libopencv-dev libspdlog-dev libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# CMAKE
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz
RUN tar -xzvf cmake-3.20.0-linux-x86_64.tar.gz && rm cmake-3.20.0-linux-x86_64.tar.gz
RUN ln -sf $(pwd)/cmake-3.20.0-linux-x86_64/bin/* /usr/bin/

RUN apt-get update
RUN apt-get -y install gcc-7
RUN apt-get -y install g++-7

# TensorRT
COPY TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz /openmmlab/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
RUN tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz && rm TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
ENV TENSORRT_DIR=/openmmlab/TensorRT-8.6.1.6
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_DIR/lib:$TENSORRT_DIR

RUN cd TensorRT-8.6.1.6/python \
    && pip install tensorrt-*-cp310-none-linux_x86_64.whl \
    && pip install tensorrt_lean-*-cp310-none-linux_x86_64.whl \
    && pip install tensorrt_dispatch-*-cp310-none-linux_x86_64.whl

RUN cd TensorRT-8.6.1.6/uff \
    && pip install uff-0.6.9-py2.py3-none-any.whl

RUN cd TensorRT-8.6.1.6/graphsurgeon \
    && pip install graphsurgeon-0.4.6-py2.py3-none-any.whl

RUN cd TensorRT-8.6.1.6/onnx_graphsurgeon \
    && pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

# Install MMEngine and MMCV
RUN pip install --no-cache-dir openmim && \
    mim install --no-cache-dir opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64 && \
    mim install --no-cache-dir "mmengine>=0.7.1" "mmcv==2.1.0"

# Install MMDetection
RUN git clone https://github.com/open-mmlab/mmdetection.git mmdetection \
    && cd mmdetection \
    && pip install --no-cache-dir -e .
RUN sed -i -e 's/labels = instances.labels/labels = [int(i.item()) for i in instances.labels]/g' mmdetection/mmdet/visualization/local_visualizer.py

# Install MMYOLO
RUN pip install albumentations cython numpy prettytable
RUN git clone https://github.com/open-mmlab/mmyolo.git mmyolo && \
    cd mmyolo \
    && mim install --no-cache-dir --no-deps -e .
RUN sed -i -e 's/2.1.0/2.1.1/g' mmyolo/mmyolo/__init__.py

# ONNX
RUN pip install onnxruntime-gpu==1.12.0 pycuda
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.0/onnxruntime-linux-x64-1.12.0.tgz
RUN tar -zxvf onnxruntime-linux-x64-1.12.0.tgz && rm onnxruntime-linux-x64-1.12.0.tgz
ENV ONNXRUNTIME_DIR=/openmmlab/onnxruntime-linux-x64-1.12.0
ENV LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

# Install MMDEPLOY
#ENV CUDNN_DIR=/usr/lib/x86_64-linux-gnu
ARG VERSION

RUN git clone -b main https://github.com/open-mmlab/mmdeploy &&\
    cd mmdeploy &&\
    if [ -z ${VERSION} ] ; then echo "No MMDeploy version passed in, building on master" ; else git checkout tags/v${VERSION} -b tag_v${VERSION} ; fi &&\
    git submodule update --init --recursive &&\
    sed -i -e 's/-std=c++14/-std=c++17/g' setup.py &&\
    sed -i -e 's/LoadImageFromNDArray/mmdet.LoadImageFromNDArray/g' mmdeploy/codebase/mmdet/deploy/object_detection.py &&\
    mkdir -p build && cd build &&\
    cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" .. &&\
    make -j$(nproc) &&\
    cd .. &&\
    mim install -e .

# PPLCV
RUN git clone https://github.com/openppl-public/ppl.cv.git && \
    cd ppl.cv &&\
    git checkout tags/v0.7.0 -b v0.7.0 && \
    ./build.sh cuda

# MMDEPLOY
ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH

RUN cd mmdeploy \
    && rm -rf build/CM* build/cmake-install.cmake build/Makefile build/csrc \
    && mkdir -p build && cd build \
    && cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DCMAKE_CXX_COMPILER=g++ \
        -Dpplcv_DIR=/openmmlab/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
        -DMMDEPLOY_CODEBASES=all \
    && make -j$(nproc) && make install

ENV LD_LIBRARY_PATH="/openmmlab/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

# Fix undefined symbol bug
RUN echo -e "\nexport LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:${CUDNN_DIR}/lib64:${LD_LIBRARY_PATH}\nldconfig" >> /root/.bashrc

# TORCHSERVE
RUN pip install torchserve torch-model-archiver nvgpu
ADD ./torchserve /torchserve

# Convert models
ADD ./convert_model /openmmlab/mmdeploy/convert_model

# Upload model
RUN apt-get install -y curl unzip
RUN curl -LO https://huggingface.co/Vades/GravelDetectionYOLOv8/resolve/main/GravelDetectionYOLOv8s.zip \
    && unzip GravelDetectionYOLOv8s.zip \
    && mv GravelDetectionYOLOv8s.mar /torchserve/model_store/GravelDetectionYOLOv8s.mar \
    && mv GravelDetectionYOLOv8s.pth /openmmlab/mmdeploy/convert_model/models/GravelDetectionYOLOv8s.pth \
    && rm GravelDetectionYOLOv8s.zip

EXPOSE 554 8080 8081 8082 8888