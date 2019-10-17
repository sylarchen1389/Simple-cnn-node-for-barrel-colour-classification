import CauctusNet 
from CauctusNet import baseline_model

import tensorflow as tf
import numpy as np 
from tqdm import tqdm

from keras.utils import np_utils
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam, SGD


from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd 


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)  # 设置session


lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('CactusNet.csv')


batch_size = 64
nb_classes = 4
nb_epoch = 10
# data_augmentation = True

# X_train,Y_train
traindf = pd.read_csv('trainY.csv',error_bad_lines=False)
Y_train = traindf['colour']
Y_train = np_utils.to_categorical(Y_train, nb_classes)
X_train = np.empty((traindf.shape[0],64,64,3))

for n in tqdm(range(traindf.shape[0])):

        image =Image.open('D:\\Code_D\\CauctusNet\\dataset\\train\\'+traindf.id[n])
        image = image.resize((64, 64), Image.ANTIALIAS)
        data = np.array(image)
        data=data.transpose((1,0,2)) 
        
        X_train[n] = data.reshape((1,64,64,3))


# X_test,Y_test

testdf = pd.read_csv('testY.csv',error_bad_lines=False)
Y_test = testdf['colour']
Y_test = np_utils.to_categorical(Y_test, nb_classes)
X_test = np.empty((testdf.shape[0],64,64,3))

for n in tqdm(range(testdf.shape[0])):

        image =Image.open('D:\\Code_D\\CauctusNet\\dataset\\test\\'+testdf.id[n])
        image = image.resize((64, 64), Image.ANTIALIAS)
        data = np.array(image)
        data=data.transpose((1,0,2)) 
        
        X_test[n] = data.reshape((1,64,64,3))


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


gen = ImageDataGenerator(rescale=1/255.,validation_split=0.1, 
                         horizontal_flip=True, 
                         vertical_flip=True,
                         rotation_range = 3,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         shear_range = 0.1)


# mean_image = np.mean(X_train, axis=0)
# X_train -= mean_image
# mean_image = np.mean(X_test, axis=0)
# X_test -= mean_image
# X_train /= 128.
# X_test /= 128.

gen.fit(X_train)
gen.fit(X_test)

model = baseline_model()
print(model.summary())
model.load_weights('baseline.h5')
opt = Adam(1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# cbs = [ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1, min_lr=1e-8, verbose=1)]
model.fit_generator(gen.flow(X_train, Y_train, batch_size=batch_size), 
                    steps_per_epoch=1000, epochs=nb_epoch, 
                    validation_data=gen.flow(X_test, Y_test, batch_size=batch_size), 
                    validation_steps=100,
                    shuffle=True, 
                    callbacks=[lr_reducer, early_stopper, csv_logger])

model.save('baseline.h5')