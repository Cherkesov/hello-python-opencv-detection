#!/usr/bin/env bash

sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
sudo apt-get install -y libgtk2.0-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libatlas-base-dev gfortran

cd ~
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

sudo apt-get install python2.7-dev
pip install numpy

cd ~
rm -rf ./opencv/
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.0.0

cd ~
rm -rf ./opencv_contrib/
git clone https://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout 3.0.0

cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..
make -j4
sudo make install
sudo ldconfig