from __future__ import print_function
# import keras
# from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from keras import optimizers
# from keras.models import Model
from keras.applications.vgg16 import VGG16
# from keras import optimizers
# from keras.layers import Dropout, Flatten, Dense
# from keras.utils.np_utils import to_categorical
import numpy as np
# from keras.layers.core import Lambda
# from keras import backend as K
# from keras import regularizers
# import cv2
# from scipy import misc
# from scipy.ndimage import imread
import os
# import argparse
# from model import build_model

DEFAULT_WIDTH = 32
DEFAULT_HEIGHT = 32

'''
Function Name: load_data()
Function Description: this function loads data from a designated dataset directory and returns
	the training set and testing set as numpy array, as well as the number of classes in the dataset
Parameters:
	- path: path of the dataset directory (default = './dataset')
	- max_x: width of the image (default = 32)
	- max_y: height of the image (default = 32)
	- prop: the proportion data reserved for testing set (default = 0.2)
Return Values:
	- x_train: trainind dataset
	- x_test: testing dataset
	- y_train: labels for the training set
	- y_test: labels for the testing set
	- num_classes: number of classes in the dataset
'''
def load_data(path = './dataset/', max_x = DEFAULT_WIDTH, max_y = DEFAULT_HEIGHT, prop = 0.2):
	x_train = np.empty([0, max_x, max_y, 3])
	x_test = np.empty([0, max_x, max_y, 3])

	y_train = np.empty([0])
	y_test = np.empty([0])
	num_classes = -1

	for dirpath, dirname, filename in os.walk(path):
		for f in filename:
			fp = os.path.join(dirpath, f)	# image file
			print("file: ", fp)

			# TODO: load the image and resize the image to appropriate size
        
	return (x_train, y_train), (x_test, y_test), num_classes


'''
Possible arguments:
- dataset directory
- batch size
- epoch
- learning rate
- bench test
- output model name
- output model location
'''
if __name__ == '__main__':

	dataset_path = './dataset/' 

	num_classes = 0
	x_train = []
	x_test = []
	y_train = []
	y_test = []
	(x_train, y_train), (x_test, y_test), num_classes = load_data(path = dataset_path)

	# TODO: comment out once the load_data() is compoeted
	# exit(-1)

	model = VGG16(weights = 'imagenet', include_top=False, 
					input_shape = (DEFAULT_WIDTH,DEFAULT_HEIGHT,3))

	model.summary()

	model = model.output

	# TODO: build model here


	# TODO: remove once model setup in done 
	exit(-1)








