import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras import regularizers

def build_model(num_classes):
	model = Sequential()
	weight_decay = 0.0005
	x_shape = [32,32,3]	

	model.add(Conv2D(64, (3, 3), padding='same',
	                 input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))

	# TODO: add an activitaion layer with 'relu' as the activataion function
	# TODO: add a  batch bormalization layer
	# TODO: add a dropout layer with 30% dropout rate

	# TODO: add a flatter layer
	# TODO: add a dense layer with 512 units
	# TODO: add an activitation layer with 'relu' as the activation function
	# TODO: add a batch normalization layer

	# TODO: add a dropout layer with 50% dropout rate
	# TODO: add a dense layer with the units equal to the number of classes
	# TODO: add a activation layer with 'softmax' as the activation function	

	model.summary()

	return model

num_classes = 10
model = build_model(num_classes)