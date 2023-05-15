#!/bin/bash
apt-get update && apt-get install wget libgl1-mesa-glx ffmpeg libsm6 libxext6 -y --no-install-recommends
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz -O onnxruntime-linux-x64-1.8.1.tgz
rm -rf onnxruntime-linux-x64-1.8.1
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
