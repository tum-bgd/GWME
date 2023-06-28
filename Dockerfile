FROM tensorflow/tensorflow:2.6.1-gpu

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Munich

# Fix GPG error
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install apt dependencies
RUN apt-get update &&\
    apt-get install -y \
    tzdata \
    git \
    protobuf-compiler

RUN python -m pip install --upgrade pip

RUN pip install cmake
RUN pip install notebook
RUN pip install geojson
RUN pip install cython

WORKDIR /app
# COPY . .

# Install Onject Detection API
RUN git clone https://github.com/tensorflow/models
RUN git clone  https://github.com/cocodataset/cocoapi/tree/master
RUN cd cocoapi/PythonAPI && make &&\
    cp -r pycocotools ../../models/research &&\
    cd ../../models/research &&\
    protoc object_detection/protos/*.proto --python_out=. &&\
    cp object_detection/packages/tf2/setup.py . &&\
    python -m pip install .

# CMD [ "bash" ]
