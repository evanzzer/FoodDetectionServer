#!/bin/bash
apt-get update && apt install python3-pip -y
pip install -r ./requirements.txt
pip install kaggle

wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz -O ~/onnxruntime-linux-x64-1.8.1.tgz
rm -rf ~/onnxruntime-linux-x64-1.8.1
tar -zxvf ~/onnxruntime-linux-x64-1.8.1.tgz -C ~/
export ONNXRUNTIME_DIR=~/onnxruntime-linux-x64-1.8.1
echo "export ONNXRUNTIME_DIR=$ONNXRUNTIME_DIR" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

mkdir ~/model
kaggle kernels output vanzzer/mmdeploy-food-detection-3-0-rev-1 -p ~/model
