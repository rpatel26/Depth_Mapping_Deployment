# works on keras version 2.0.8 and tensorflow version 1.4.0
# works on keras version 2.0.8 and tensorflow version 1.4.0
# did work on keras version 2.2.2 and tensorflow version 1.4.1
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
import os
import argparse

class cifar10vgg:
    def __init__(self,train=False):
#         self.num_classes = 2
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        (self.x_train, self.y_train), (self.x_test, self.y_test), self.num_classes = self.load_data()
        self.num_classes += 1
        
        batch_start = 5
        batch_step = 5
        batch_end = 100 + batch_step

        max_epoch_start = 50
        max_epoch_end = max_epoch_start 
        
        learning_start = 0.000001
        learning_step = 0.000005
        learning_end = 1 + learning_step
        
        self.fileName = 'output.txt'
        self.fp = open(self.fileName, "w")
        self.fp.write("Results:\n")
        
        max_loss = 0.7
        
        self.model = self.build_model()

        if train:
            for i in range(batch_start, batch_end, batch_step):
                for j in range(max_epoch_start, max_epoch_end, max_epoch_step):
                    for k in np.arange(learning_start, learning_end, learning_step):
                        self.model = self.train(self.model, self.x_train, self.y_train,
                                           self.x_test, self.y_test, 
                                                batchSize = i, max_epoches = j, learningRate = k)
                        predicted_x = self.model.predict(self.x_test)
                        residuals = np.argmax(predicted_x,1)==self.y_test#np.argmax(self.y_test,1)

                        loss = sum(residuals)/len(residuals)
                        print("the validation 0/1 loss is: ",loss)
                        if(max_loss <= loss):
                            print("Loss Improved")
                            self.fp.write("loss updated\n")
                            
                            self.fp.write("loss = ")
                            self.fp.write(str(loss))
                            self.fp.write("\n")
                            
                            self.fp.write("batch size = ")
                            self.fp.write(str(i))
                            self.fp.write("\n")
                            
                            self.fp.write("max epoch = ")
                            self.fp.write(str(j))
                            self.fp.write("\n")
                            
                            self.fp.write("learning rate = ")
                            self.fp.write(str(k))
                            self.fp.write("\n\n")

                            max_loss = loss
        else:
            self.model.load_weights('cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
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
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=False,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size, verbose=1)

    def load_data(self, path = './dataset/', max_x = 32, max_y = 32, prop = 0.2):
#         print("loading dataset")
        
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
                        print("loading file: ", fp)
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

    def train(self, model, xTrain, yTrain, xTest, yTest,
              batchSize = 32, max_epoches = 50 ,learningRate = 0.01):
        # best result:
        # batch_size = 80; maxepoches = 1000; learning_rate = 0.001
        # 0/1 validation loss: 0.8656
        # mess around with batch_size and maxepoches for results
        #training parameters
        
        print("batch size = ", batchSize)
        print("max epoches = ", max_epoches)
        print("learning rate = ", learningRate)
        batch_size = batchSize
        maxepoches = max_epoches
        learning_rate = learningRate
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        # x_train: (-1, 32,32,3) numpy array
        # y_train: (-1, 1) numpy array
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        (x_train, y_train), (x_test, y_test) = (xTrain, yTrain),(xTest, yTest)

#         print("Finished loading dataset")
#         print("size of x_train = ", x_train.shape)
#         print("size of x_test = ", x_test.shape)
#         print("size of y_train = ", y_train.shape)
#         print("size of y_test = ", y_test.shape)
#         return

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
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
#         model.save_weights('cifar10vgg.h5')
        model.save_weights('personal_train.h5')
        return model

if __name__ == '__main__':
    '''
    dog = misc.imread("./dog2.jpg")
    dog = cv2.resize(dog, (32,32))
    dog = np.expand_dims(dog, axis = 0) 

    dog_label = np.array([5])
    dog_label = np.expand_dims(dog_label, axis = 0)
    '''
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
   
    ''' 
    print("dog = ", dog.shape)  
    print("x_test = ", x_test.shape) 
    print("y_test = ", y_test.shape) 
    x_test = np.concatenate((x_test, dog), axis = 0)
    y_test = np.concatenate((y_test, dog_label), axis = 0)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print("dog = ", dog.shape)  
    print("x_test = ", x_test.shape) 
    print("y_test = ", y_test.shape) 
    '''
    model = cifar10vgg(train=True)

    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)==np.argmax(y_test,1)

#     print("Result = ", residuals[-1])
    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)
    print("AWS has been succesfully setup."\n)

