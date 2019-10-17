import pandas as pd 
import numpy as np 
import random
from tqdm import tqdm

randomRate = 1.2


def randomTrainDf(df):

    for i in tqdm(range(int(df.shape[0]*randomRate))):
        i = random.randint(0,df.shape[0]-1)
        j = random.randint(0,df.shape[0]-1)

        if i==j :
            continue
        temp_id = df[:]['id'][i]
        temp_colour = df[:]['colour'][i]

        df[:]['id'][i] =  df[:]['id'][j]
        df[:]['colour'][i] =  df[:]['colour'][j]

        df[:]['id'][j] = temp_id
        df[:]['colour'][j] = temp_colour

def randomTestDf(testdf,predictdf):
     for i in tqdm(range(int(testdf.shape[0]*randomRate))):
        i = random.randint(0,testdf.shape[0]-1)
        j = random.randint(0,testdf.shape[0]-1)

        if i==j :
            continue
        temp_id = testdf[:]['id'][i]
        temp_id_predict = predictdf[:]['id'][i]
        temp_colour = testdf[:]['colour'][i]

        testdf[:]['id'][i] =  testdf[:]['id'][j]
        testdf[:]['colour'][i] =  testdf[:]['colour'][j]
        predictdf[:]['id'][i] = predictdf[:]['id'][j]

        testdf[:]['id'][j] = temp_id
        testdf[:]['colour'][j] = temp_colour
        predictdf[:]['id'][j] = temp_id_predict

traindf = pd.read_csv('trainY.csv',error_bad_lines=False)
testdf = pd.read_csv('testY.csv',error_bad_lines=False)
predictdf = pd.read_csv('predict.csv',error_bad_lines=False)
print(traindf[:]['id'][2])

randomTrainDf(traindf)
randomTestDf(testdf,predictdf)

traindf.to_csv('trainY.csv',index = False)
testdf.to_csv('testY.csv',index = False)
predictdf.to_csv('predict.csv', index=False)

