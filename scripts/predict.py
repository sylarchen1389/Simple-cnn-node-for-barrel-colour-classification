import CauctusNet
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import tensorflow as tf
import pandas as pd

from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as KTF


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)  # 设置session

model=load_model("baseline.h5")
mean_image=np.load("meanimage.npy")
size=(64,64)

predictX=np.empty((3,64,64,3)) 


testdf = pd.read_csv('predict.csv',error_bad_lines=False)
print(testdf.head())
pred = np.empty((testdf.shape[0],))


gen = ImageDataGenerator(rescale=1/255.,validation_split=0.1, 
                         horizontal_flip=True, 
                         vertical_flip=True,
                         rotation_range = 3,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         shear_range = 0.1)

for n in tqdm(range(testdf.shape[0])):

        image =Image.open('D:\\Code_D\\CauctusNet\\dataset\\test\\'+testdf.id[n])
        image = image.resize((64, 64), Image.ANTIALIAS)
        data = np.array(image)
        data=data.transpose((1,0,2)) 
        data = data.reshape((1,64,64,3))
        gen.fit(data)

        a = model.predict(data)
        print(a[0])
        pred[n] = np.argmax(np.array(a[0]))

testdf = pd.read_csv('testY.csv',error_bad_lines=False)
Y_test = testdf['colour']

Y_test = np.argmax(Y_test,axis= 0 )

testdf['colour'] = pred
testdf.to_csv('predict.csv', index=False)
print("acc: ", np.sum(Y_test == pred)/testdf.shape[0])