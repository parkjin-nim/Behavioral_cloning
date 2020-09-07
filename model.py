import os
import csv
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#%matplotlib inline

# keras ==2.4.3
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import pickle

from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle

samples = []
with open('../comb_data/driving_log1.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

data1_size = len(samples)
# print("data1 size:",len(samples))

with open('../comb_data/driving_log2.csv') as csvfile:
    reader = csv.reader(csvfile)
    counter = 0
    for line in reader:
        samples.append(line)
        counter +=1

data2_size = counter
# print("data2 size:",len(samples))
shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# print(len(train_samples), len(validation_samples))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            #print("batch samples from:",offset," to: ", offset+batch_size, " len: ", len(batch_samples))
            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(line[3])
                row = [source_path.split('/')[-1] for source_path in batch_sample[0:3]]

                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                path = "../comb_data/IMG/" # fill in the path to your training IMG directory
                img_center = np.asarray(Image.open(path + row[0]))
                img_left = np.asarray(Image.open(path + row[1]))
                img_right = np.asarray(Image.open(path + row[2]))

                # add images and angles to data set
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])
                
                augmented_images, augmented_measurements = [],[]
                for image, measurement in zip([img_center, img_left, img_right], 
                                              [steering_center, steering_left, steering_right]):
                    augmented_images.append(cv2.flip(image,1))
                    augmented_measurements.append(measurement*-1.0)

                images += augmented_images
                angles += augmented_measurements
                #print(len(images), len(angles))
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
print("The number of total samples is...", data1_size*3*2+data2_size*3)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# print("Building model...")
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
# input(3x160x320) is cropped to 3x65x320
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))


print("Training starts..")
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(len(validation_samples)/batch_size), 
            epochs=10, verbose=1)