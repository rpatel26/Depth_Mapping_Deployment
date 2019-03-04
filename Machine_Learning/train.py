from __future__ import print_function
import os
import numpy as np

from keras.applications.vgg16 import VGG16
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.models import Model

from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import cv2
from scipy import misc
from scipy.ndimage import imread

import argparse
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
def load_data2(path = './dataset/', max_x = DEFAULT_WIDTH, max_y = DEFAULT_HEIGHT, prop = 0.2):
	x_train = np.empty([0, max_x, max_y, 3])
	x_test = np.empty([0, max_x, max_y, 3])

	y_train = np.empty([0])
	y_test = np.empty([0])
	num_classes = -1

	for dirpath, dirname, filename in os.walk(path):
		for f in filename:
			fp = os.path.join(dirpath, f)	# image file
			print("file: ", fp)

	return (x_train, y_train), (x_test, y_test), num_classes



def load_data(path = './dataset/', max_x = DEFAULT_WIDTH, max_y = DEFAULT_HEIGHT, prop = 0.2):
	x_train = np.empty([0, max_x, max_y, 3])
	x_test = np.empty([0, max_x, max_y, 3])

	y_train = np.empty([0])
	y_test = np.empty([0])
	label = -1

	for dirpath, dirname, filename in os.walk(path):
		x_data = []
		y_data = [] 
		for f in filename:
			fp = os.path.join(dirpath, f)	# image file
			image = imread(fp)
			# print("loading file: ", fp)
			image = cv2.resize(image, (max_y,max_x))
					
			if len(image.shape) == 3:
				# image is rgb
				x_data.append(image)
				y_data.append(label)
                
		if label != -1:
			x_data = np.array(x_data)
			y_data = np.array(y_data)
			num_of_image = x_data.shape[0]
			
			num_of_test = int(num_of_image * prop)
			num_of_train = num_of_image - num_of_test
				
			x_data_train = x_data[0:num_of_train, :]
			x_data_test = x_data[num_of_train:, :]
				
			y_data_train = y_data[0:num_of_train]
			y_data_test = y_data[num_of_train:]
			
			x_train = np.concatenate((x_train, x_data_train), axis = 0)
			x_test = np.concatenate((x_test, x_data_test), axis = 0)
			
			y_train = np.concatenate((y_train, y_data_train), axis = 0)
			y_test = np.concatenate((y_test, y_data_test), axis = 0)
				
	
		label += 1
        
	return (x_train, y_train), (x_test, y_test), label


''' 
Function name: load_model()
Function Description: this function builds the model and load weight for training
Parameters:
	- num_classes: number of objects being trained 
Return Value:
	- model: object contraining the model, with weights loaded
'''
def load_model2(num_classes):
	# TODO: use VGG16 to load lower layers of vgg16 network and declare it as base_model
    # TODO: use 'imagenet' for weights, include_top=False, (IMG_H, IMG_W, NUM_CHANNELS) for input_shape

	base_out = base_model.output
    # TODO: add a flatten layer
    # TODO: add a dense layer with 256 units and relu activation function
    # TODO: add dropout layer with 0.5 rate
    # TODO: and another dense layer for output. This final layer should have the same number of units as classes 
	
	model = Model(inputs=model.inputs, outputs=predictions) 
	model.summary()

	return model

def load_model(num_classes):
	# model = VGG16(weights = 'imagenet', include_top=False, 
	# 				input_shape = (DEFAULT_WIDTH,DEFAULT_HEIGHT,3))

	# base_out = model.output
	# base_out = Flatten()(base_out)
	# base_out = Dense(256, activation = 'relu')(base_out)
	# base_out = Dropout(0.5)(base_out)
	# predictions = Dense(num_classes, activation = 'softmax')(base_out)

	# model = Model(inputs=model.inputs, outputs=predictions)

	x_shape = [DEFAULT_WIDTH,DEFAULT_HEIGHT,3]
	
	model = Sequential()
	model.add(Conv2D(32, (3,3), input_shape = x_shape, activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Conv2D(16, (3,3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Flatten())
	model.add(Dense(32, activation = 'relu'))
	model.add(Dropout(0.5))
	
	prediction = model.add(Dense(num_classes, activation = 'softmax'))
	model.summary()	

	return model


def normalize(X_train,X_test):
	#this function normalize inputs for zero mean and unit variance
	# it is used when training a model.
	# Input: training set and test set
	# Output: normalized training set and test set according to the trianing set statistics.
	mean = np.mean(X_train,axis=(0,1,2,3))
	std = np.std(X_train, axis=(0, 1, 2, 3))
	X_train = (X_train-mean)/(std+1e-7)
	X_test = (X_test-mean)/(std+1e-7)
	return X_train, X_test

def train_model(model, xTrain, yTrain, xTest, yTest,
		num_classes, batchSize = 128, max_epoches = 250,learningRate = 0.001, outFile = 'personal_train.h5'):
	
	batch_size = batchSize
	maxepoches = max_epoches
	learning_rate = learningRate
	lr_decay = 1e-6
	lr_drop = 20
	

	(x_train, y_train), (x_test, y_test) = (xTrain, yTrain),(xTest, yTest)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train, x_test = normalize(x_train, x_test)

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	def lr_scheduler(epoch):
		return learning_rate * (0.5 ** (epoch // lr_drop))
	reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=True,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=True,  # divide each input by its std
		zca_whitening=True,  # apply ZCA whitening
		rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=True)  # randomly flip images
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(x_train)



	#optimization details
	sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


	# training process in a for loop with learning rate drop every 25 epoches.

	historytemp = model.fit_generator(datagen.flow(x_train, y_train,
									 batch_size=batch_size),
						steps_per_epoch=x_train.shape[0] // batch_size,
						epochs=maxepoches, 
						validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=1)


	model.save_weights(outFile)
	print("Model saved as: ", outFile)

	return model

def train_model2(model, xTrain, yTrain, xTest, yTest,
		num_classes, batchSize = 128, max_epoches = 250,learningRate = 0.001, outFile = 'personal_train.h5'):
	
	batch_size = batchSize
	maxepoches = max_epoches
	learning_rate = learningRate
	lr_decay = 1e-6
	lr_drop = 20
	

	(x_train, y_train), (x_test, y_test) = (xTrain, yTrain),(xTest, yTest)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train, x_test = normalize(x_train, x_test)

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	def lr_scheduler(epoch):
		return learning_rate * (0.5 ** (epoch // lr_drop))
	reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=True,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=True,  # divide each input by its std
		zca_whitening=True,  # apply ZCA whitening
		rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=True)  # randomly flip images
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(x_train)



	#optimization details
	sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


	# training process in a for loop with learning rate drop every 25 epoches.

	historytemp = model.fit_generator(datagen.flow(x_train, y_train,
									 batch_size=batch_size),
						steps_per_epoch=x_train.shape[0] // batch_size,
						epochs=maxepoches, 
						validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=1)


	# TODO: save model weight to the file specified by the 'outFile' parameter
	model.save_weights(outFile)
	print("Model saved as: ", outFile)

	return model


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

	# TODO: remove exit(-1) when load_data() is completed
	# exit(-1)


	# TODO: remove exit(-1) once load_model() is completed
	model = load_model(num_classes) 
	exit(-1)

	# TODO: remove exit(-1) once train_model() is completed
	model = train_model(model, x_train, y_train, x_test, y_test, num_classes)
	exit(-1)








