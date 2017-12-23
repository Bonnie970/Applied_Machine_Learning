from __future__ import print_function
import os
import argparse
import numpy as np
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime # for filename conventions
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers
from tensorflow.python.lib.io import file_io # for better file I/O

batch_size = 256
num_classes = 40
epochs = 50

fine_tune_flag = False
model_to_load = 'model-cnn3-epoch100-batch256-aug.h5'


classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36,
               40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

def load_class(file, classes):
    print('Loading Categories from '+file)
    y_train = np.loadtxt(file, delimiter=",")
    # 6. Preprocess class labels
    Y_train = [classes.index(i) for i in y_train]
    Y_train = np_utils.to_categorical(Y_train, len(classes))
    return Y_train

# Create a function to allow for different training data and other options
def train_model():

    model = Sequential()
    # model.add(Lambda(norm, input_shape=(64,64,1)))
    model.add(Convolution2D(32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(64, 64, 1),
                            data_format="channels_last"))
    model.add(Convolution2D(32, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(Convolution2D(32, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization())
    # model.add(Convolution2D(32, kernel_size=(3, 3),padding= "same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(256, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(Convolution2D(256, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(Convolution2D(256, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))


    model.summary()
    
	
	### Fine tuning
	if fine_tune_flag:
		model.load_weights(model_to_load)
		# frozen first 18 layers
		for layer in model.layers[:18]:
			layer.trainable = False
		
		model.summary()


		model.compile(loss='categorical_crossentropy',
					  optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
					  metrics=['accuracy'])
	### normal training 				  
	else: 
		model.compile(loss='categorical_crossentropy',
					  optimizer='adam',
					  metrics=['accuracy'])
	
    

    ### Train/fine tune with augmentation
    classes_str = ['%d'%i for i in classes]
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.3,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        'trainImages',  # this is the target directory
        target_size=(64, 64),  # all images will be resized to 150x150
        batch_size=batch_size,
        color_mode='grayscale',
         class_mode='categorical',
        classes=classes_str )

    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
        'valImages',
        target_size=(64, 64),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
		classes=classes_str)

    model.fit_generator(
        train_generator,
        steps_per_epoch=47520 // batch_size,
        epochs=epochs,
		validation_data=validation_generator,
        validation_steps=2480 // batch_size
	)

    model.save('model-cnn3-epoch{}-batch{}-aug-fine.h5'.format(epochs, batch_size))

    ### Prediction
    predict(model, trial=0, classes=classes, batch_size=batch_size)

def predict(model,trial, classes, batch_size):
    print('Predicting ...')

    testfolder = "testImages"
    os.chdir(testfolder)
    filename = '.png'
    test_x = np.zeros((10000, 64, 64), dtype=np.float)
    for i in range(10000):
        image = misc.imread(str(i) + filename)
        test_x[i, :] = image

    test_x = test_x.astype('float32').reshape(10000, 64, 64, 1)
    test_x /= 255
    os.chdir('..')

    test_y = model.predict(test_x, batch_size=batch_size, verbose=0)
    print('Saving prediction results to test_trial_{}.csv'.format(trial))
    with open('test_trial_{}.csv'.format(trial), 'w') as f:
        f.write('Id,Label\n')
        id = 1
        for y in test_y:
            f.write('{},{}\n'.format(id, classes[y.argmax()]))
            id += 1
    f.close()


if __name__ == '__main__':
    
	train_model()
