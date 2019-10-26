from PIL import Image
import os
import numpy as np
import random


size=(64,64)
datadir=os.path.join(os.getcwd(),"data")
list=os.listdir(datadir)
trainX=np.empty((int(len(list)*0.7),64,64,3))
trainY=np.empty((int(len(list)*0.7),1),dtype="int8")
testX=np.empty((int(len(list)*0.3),64,64,3))
testY=np.empty((int(len(list)*0.3),1),dtype="int8")
traincnt=0
testcnt=0
for path in list:
    fullpath=os.path.join(datadir,path)
    if path[-3:]=="jpg":
        im=Image.open(fullpath)
        out=im.resize(size,Image.ANTIALIAS)
        X=np.array(out)
        Y=0
        if path[:3]=="red":
            Y=0
        elif path[:4]=="blue":
            Y=1
        elif path[:6]=="yellow":
            Y=2
        elif path[:4]=="none":
            Y=3
        if random.random()<0.8:
            if traincnt<trainX.shape[0]:
                trainX[traincnt]=X
                trainY[traincnt]=Y
                traincnt+=1
            elif testcnt<testX.shape[0]:
                testX[testcnt]=X
                testY[testcnt]=Y
                testcnt+=1
        else:
            if testcnt<testX.shape[0]:
                testX[testcnt]=X
                testY[testcnt]=Y
                testcnt+=1
            elif traincnt<trainX.shape[0]:
                trainX[traincnt]=X
                trainY[traincnt]=Y
                traincnt+=1

print("Size of Dateset for Train:%d",traincnt)
print("Size of Dateset for Test:%d",testcnt)
np.save("trainX.npy",trainX)
np.save("trainY.npy",trainY)
np.save("testX.npy",testX)
np.save("testY.npy",testY)