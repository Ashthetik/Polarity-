#!/bin/sh


# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip make

mkdir opencv && cd opencv

# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip

unzip opencv.zip
unzip opencv_contrib.zip

# Create build directory
mkdir build && cd build

cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
make -j 8 
sudo make install 

cd ../..

rm -rf opencv/