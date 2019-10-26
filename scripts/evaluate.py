from PIL import Image
import os
import numpy as np
import random
from keras.models import load_model
from keras.utils import np_utils
size=(64,64)
nb_classes=4


datadir=os.path.join(os.getcwd(),"dataset","barrels")


evaX=np.empty((1000,64,64,3))
evaY=np.empty((1000,1),dtype="int8")

reddir=os.path.join(datadir,"r_barrel")
yellowdir=os.path.join(datadir,"y_barrel")
bluedir=os.path.join(datadir,"b_barrel")
redlist=os.listdir(reddir)
yellowlist=os.listdir(yellowdir)
bluelist=os.listdir(bluedir)

evaPath=[]
for i in range(1000):
    if i<333:
        index=int(random.random()*len(redlist))
        evaPath.append(os.path.join(reddir,redlist[index]))
        evaY[i]=0
    elif i<666:
        index=int(random.random()*len(bluelist))
        evaPath.append(os.path.join(bluedir,bluelist[index]))
        evaY[i]=1
    else:
        index=int(random.random()*len(yellowlist))
        evaPath.append(os.path.join(yellowdir,yellowlist[index]))
        evaY[i]=2

for i in range(1000):
    image=Image.open(evaPath[i])
    image=np.array(image.resize(size,Image.ANTIALIAS))
    evaX[i]=image

evaY = np_utils.to_categorical(evaY, nb_classes)
evaX = evaX.astype('float32')
# subtract mean and normalize
mean_image = np.load("meanimage.npy")
evaX -= mean_image
evaX/=128.
model=load_model("baseline.h5")

loss,acc=model.evaluate(evaX,evaY)
print("loss={:.4f},accurcy={:.4f}".format(loss,acc))
