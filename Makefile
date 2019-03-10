path=~

pi_setup:
	# install vim1
	sudo apt-get install vim -y
	# install pip
	sudo apt-get install python3-pip
	# uninstall and reinstall pip3
	sudo python3 -m pip uninstall pip && sudo apt install python3-pip --reinstall

kinect_setup:
	
	#install dependency for matplotlib
	sudo apt-get build-dep python3-matplotlib
	#install matplotlib
	pip3 install matplotlib
	#install scikit-learn
	sudo -H pip3 install scikit-learn 
	#install plotly
	sudo -H pip3 install plotly
	#install numpy
	sudo apt-get install python3-numpy 
	#install jupyter notebook
	sudo python3 -m pip install jupyter
	#install cython3
	sudo apt-get install cython3
	#install python-dev
	sudo apt-get install python3-dev
	#intall openCV
	make openCV
openCV:
	make openCV_dependencies
	cd $(path); wget -O opencv.zip https://github.com/opencv/opencv/archive/3.3.1.zip; unzip opencv.zip; cd ./opencv-3.3.1; mkdir build; cd build; cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..; make -j3; sudo make install

openCV_dependencies:
	sudo apt-get -y install build-essential checkinstall cmake pkg-config yasm
	sudo apt-get -y install git gfortran
	sudo apt-get -y install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
	sudo apt-get -y install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
	sudo apt-get -y install libjpeg8-dev libjasper-dev libpng12-dev
	sudo apt-get -y install libtiff5-dev
	sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
	sudo apt-get -y install libxine2-dev libv4l-dev
	sudo apt-get -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
	sudo apt-get -y install qt5-default libgtk2.0-dev libtbb-dev
	sudo apt-get -y install libatlas-base-dev
	sudo apt-get -y install libfaac-dev libmp3lame-dev libtheora-dev
	sudo apt-get -y install libvorbis-dev libxvidcore-dev
	sudo apt-get -y install libopencore-amrnb-dev libopencore-amrwb-dev
	sudo apt-get -y install x264 v4l-utils
	# optional dependencies
	sudo apt-get -y install libprotobuf-dev protobuf-compiler
	sudo apt-get -y install libgoogle-glog-dev libgflags-dev
	sudo apt-get -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen


kinect_install:
	#install the necessary dependencies or the kinect
	sudo apt-get install git-core cmake freeglut3-dev pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev
	sudo reboot

machine_learning_setup:
	ls

hardware_setup:
	ls
