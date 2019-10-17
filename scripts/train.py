import CauctusNet 


import tensorflow as tf
import numpy as np 
from tqdm import tqdm
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam, SGD


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)  # 设置session

lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('CactusNet.csv')


batch_size = 256
nb_classes = 4
nb_epoch = 10
# data_augmentation = True


gen,X_train,X_test,Y_train,Y_test = CauctusNet.load_data()

model = CauctusNet.baseline_model()
print(model.summary())
model.load_weights('baseline.h5')
opt = Adam(1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# cbs = [ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1, min_lr=1e-8, verbose=1)]
model.fit_generator(gen.flow(X_train, Y_train, batch_size=batch_size), 
                    steps_per_epoch=1000, epochs=nb_epoch, 
                    validation_data=(X_test, Y_test), 
                    validation_steps=100, shuffle=True, 
                    callbacks=[lr_reducer, early_stopper, csv_logger])

model.save('baseline.h5')
model.evaluate(X_test,Y_test)