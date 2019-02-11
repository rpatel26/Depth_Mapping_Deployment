pi_setup:
	# install vim
	sudo apt-get install vim -y
	# install pip
	sudo apt-get install python3-pip
	# uninstall and reinstall pip3
	sudo python3 -m pip uninstall pip && sudo apt install python3-pip --reinstall

kinect_setup:
	#install openCV
	#sudo pip3 install opencv-python
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

kinect_install:
	#install the necessary dependencies or the kinect
	sudo apt-get install git-core cmake freeglut3-dev pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev
	#clone the github repository into your system
	git clone git://github.com/OpenKinect/libfreenect.git

machine_learning_setup:
	ls

hardware_setup:
	ls
