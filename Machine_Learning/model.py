import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras import regularizers

def build_model():
	model = Sequential()
	weight_decay = 0.0005
	x_shape = [32,32,3]	

	model.add(Conv2D(64, (3, 3), padding='same',
	                 input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))

	model.summary()

	return model

model = build_model()