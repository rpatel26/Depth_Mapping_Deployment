from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import cv2
from scipy import misc
from scipy.ndimage import imread
import os, glob, time
import argparse
from keras.models import load_model


IMG_SRC_DIR = "./source"  # ec2
RESULT_DEST_DIR = "./result/result.txt"


def build_model(num_classes):
	# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

	model = Sequential()
	weight_decay = 0.0005
	x_shape = [32,32,3]

	model.add(Conv2D(64, (3, 3), padding='same',
					 input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))

	model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	return model

if __name__ == '__main__':

    ## Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help = "Directory containing training set")
    parser.add_argument("--model_name", help = "Name of the trained model")

    args = parser.parse_args()

    dataset_path = './dataset/' if args.data_dir is None else args.data_dir
    #model_path = './cifar10vgg.h5' if args.model_name is None else args.model_name
    model_path = args.model_name

    if model_path is None:
        print("Must specify an input model")
        exit(-1)
    
    print("Classifying Result")

    # retun classification at index + 1
    classes = [x[0] for x in os.walk(dataset_path)]
    num_classes = len(classes) - 1

    print("number of classes = ", num_classes)    
    for i in classes:
        print(i)
    
    model = build_model(num_classes)
    model.load_weights(model_path)
  
    count = 0 
    while True:
        cpt = sum([len(files) for r, d, files in os.walk(IMG_SRC_DIR)])
        if(cpt == 1):
            file_list = glob.glob(os.path.join(IMG_SRC_DIR,'*')) 
            
            image = cv2.imread(file_list[0])

            if image is None:
                # print("Image is of type None")
                continue

            print("File detected!!")
            print(file_list)

            image = cv2.resize(image, (32,32))
            image = np.expand_dims(image, axis = 0)
    
            predicted_values = model.predict(image) # sum of every element adds up to 1
            result = classes[np.argmax(predicted_values, axis = 1)[0] + 1] 

            print("result = ",result)
            print("count = ", count)
            fp = open(RESULT_DEST_DIR, "w")
    
            fp.write("Result: ")
            fp.write(result)
            fp.write("\n")
            
            '''
            fp.write(str(count))
            fp.write("\n")  
            '''

            fp.close()
            count = count + 1
 
            #time.sleep(5) 
            os.remove(file_list[0])
            # os.remove(RESULT_DEST_DIR)
