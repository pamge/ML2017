#!/usr/local/bin/python3
import sys
import math
import numpy
import time
from io import BytesIO
import tensorflow
import keras.utils
import keras.models
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, AveragePooling2D, Dropout

def read_test_data(test_path):
	test_features = []
	test_data = numpy.genfromtxt(test_path, delimiter = ',', skip_header = 1, dtype = None)
	for _, feature in test_data:
		test_features.append(feature.split(b' '))
	test_features = numpy.array(test_features, dtype = 'float32') / 255
	return test_features.reshape(test_features.shape[0], 48, 48, 1)
	
def read_train_data(train_path):
	train_labels, train_features = [], []
	train_data = numpy.genfromtxt(train_path, delimiter = ',', skip_header = 1, dtype = None)
	for label, feature in train_data:
		train_labels.append(int(label))
		train_features.append(feature.split(b' '))
	train_labels = keras.utils.to_categorical(train_labels, 7)
	train_features = numpy.array(train_features, dtype = 'float32') / 255
	return train_features.reshape(train_features.shape[0], 48, 48, 1), train_labels	

def create_image_generator(train_labels, train_features):
    generator = ImageDataGenerator( 
    	width_shift_range = 0.10,
        height_shift_range = 0.10,
        rotation_range = 10,
        shear_range = 0.10,
        zoom_range = 0.10,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )
    generator.fit(train_features)
    return generator.flow(train_features, train_labels, batch_size = 500)

def train(train_labels, train_features, generator = False):
	model = keras.models.Sequential()
	# convolution block
	model.add(Conv2D(32, (3, 3), input_shape = [48, 48, 1], activation = 'relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
	# convolution block
	model.add(Conv2D(64, (3, 3), input_shape = [48, 48, 1], activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
	model.add(Dropout(0.7))
	# Flatten
	model.add(Flatten())
	# fully connected layer
	model.add(Dense(units = 1 << 11, activation = 'relu'))
	model.add(Dropout(0.2))
	# fully connected layer
	model.add(Dense(units = 1 << 11, activation = 'relu'))
	model.add(Dropout(0.2))
	# softmax
	model.add(Dense(units = 7, activation = 'softmax'))
	# compile model
	model.summary()
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	# fit model
	if generator == False:
		model.fit(train_features, train_labels, shuffle = True, validation_split = 0.15, batch_size = 256, epochs = 200)
	else:
		# test_labels = train_labels[:int(len(train_labels) * 0.15), :]
		# test_features = train_features[:int(len(train_features) * 0.15), :]
		# train_labels = train_labels[int(len(train_labels) * 0.15):]
		# train_features = train_features[int(len(train_features) * 0.15):]
		generator = create_image_generator(train_labels, train_features)
		model.fit_generator(generator, samples_per_epoch = train_labels.shape[0] * 2, epochs = 150)
		# model.fit_generator(generator, samples_per_epoch = train_labels.shape[0] * 2, epochs = 150, validation_data = (test_features, test_labels))
	# save model
	t = int(time.time())
	print('model_%d' % t)
	model.save('model_%d' % t)
	return model

def test(model, test_features):
	# DNN only
	test_labels = []
	predictions = model.predict(test_features)
	for i in range(len(test_features)):
		test_label = numpy.argmax(predictions[i])
		test_labels.append(test_label)
	return test_labels

def output_labels(path, labels):
	with open(path, 'w') as f:
		f.write('id,label\n')
		for i in range(len(labels)):
			f.write('%d,%d\n' % (i, labels[i]))
		return
	raise OSError(2, 'No such file or directory', path)
	
def main():
	# use tensorflow as backend
	tensorflow_config = tensorflow.ConfigProto()
	tensorflow_config.gpu_options.allow_growth = True
	keras.backend.tensorflow_backend.set_session(tensorflow.Session(config = tensorflow_config))
	# test or train
	if len(sys.argv) == 4 and sys.argv[1] == 'test':
		test_features = read_test_data(sys.argv[2]) # test.csv
		model = keras.models.load_model('best_model')
		test_labels = test(model, test_features)
		output_labels(sys.argv[3], test_labels) # result.csv
	elif len(sys.argv) == 3 and sys.argv[1] == 'train':
		train_features, train_labels = read_train_data(sys.argv[2]) # train.csv
		model = train(train_labels, train_features, True)
	else:
		print('Usage: python hw3.py test test.csv result.csv')
		print('Usage: python hw3.py train train.csv')
		exit(-1)
	
if __name__ == '__main__':
    main()
