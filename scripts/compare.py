import tensorflow as tf 
import keras 
import numpy as np
import pandas as pd 
import os
import keras.backend.tensorflow_backend as KTF
import keras.backend as K
from keras.utils import np_utils

from PIL import Image
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)
KTF.set_session(sess)


if KTF.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3



X_train=np.load(os.path.join("dataset","trainX.npy"))
X_test=np.load(os.path.join("dataset","testX.npy"))
Y_train=np.load(os.path.join("dataset","trainY.npy"))
Y_test=np.load(os.path.join("dataset","testY.npy"))


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Y_train = np_utils.to_categorical(Y_train, nb_classes)
# Y_test = np_utils.to_categorical(Y_test, nb_classes)

X_max = tf.argmax(X_train,axis = CHANNEL_AXIS)
X_max_shape = K.int_shape(X_max)
X_max_shape = np.array(X_max_shape)
X_max = sess.run(X_max)

# print(X_max_shape[-1])
X_predict = []


def predict():
    

    for item in X_max :
        b = 0
        g = 0
        r = 0
        acc_rate = 0
        for i in  range(X_max_shape[1]):
            for j in range(X_max_shape[2]) :
                if item[i][j] == 0:
                    b += 1
                elif item[i][j] == 1: 
                    g += 1
                elif item[i][j] == 2:
                    r += 1

        if b>=g and b>=r:
            X_predict.append(0)
        elif g >=b and g>=r:
            X_predict.append(1)
        elif r >= g  and r >= b:
            X_predict.append(2)


    for i in range(4420)  :
        if X_predict[i] == Y_train[i,-1]:
            acc_rate += 1

    acc_rate /= 4420

    print("acc rate:",acc_rate)



predict()