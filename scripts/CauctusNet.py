import tensorflow as tf 
import numpy as np 
import pandas as pd 
import os

from PIL import Image
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
import keras.backend.tensorflow_backend as KTF

nb_classes = 4

def load_data():
    # if dataframe is None:
    #     dataframe = pd.read_csv('../input/aerial-cactus-identification/train.csv')
    # dataframe['has_cactus'] = dataframe['has_cactus'].apply(str)

    
    # The data, shuffled and split between train and test sets:
    X_train=np.load(os.path.join("dataset","trainX.npy"))
    X_test=np.load(os.path.join("dataset","testX.npy"))
    Y_train=np.load(os.path.join("dataset","trainY.npy"))
    Y_test=np.load(os.path.join("dataset","testY.npy"))
    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    np.save("meanimage.npy",mean_image)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.


    gen = ImageDataGenerator(validation_split=0.1, 
                             horizontal_flip=True, 
                             vertical_flip=True,
                             rotation_range = 3,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             shear_range = 0.1)

    # trainGen = gen.flow_from_dataframe(dataframe, directory='../input/aerial-cactus-identification/train/train', x_col='id', y_col='has_cactus', has_ext=True, target_size=(32, 32),
    #     class_mode=mode, batch_size=batch_size, shuffle=True, subset='training')
    # testGen = gen.flow_from_dataframe(dataframe, directory='../input/aerial-cactus-identification/train/train', x_col='id', y_col='has_cactus', has_ext=True, target_size=(32, 32),
    #     class_mode=mode, batch_size=batch_size, shuffle=True, subset='validation')
    
    gen.fit(X_train)

    return   gen,X_train,X_test,Y_train,Y_test


  
def baseline_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())

    model.add(GlobalAveragePooling2D())
    model.add(Dense(4, activation='softmax'))

    return model

