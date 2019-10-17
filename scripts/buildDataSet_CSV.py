import pandas as pd
import numpy as np

""""df is the csv file name wait to be wroted"""

class DataSetBuilder(path,df):
    df = pd.read_csv(path,df)
    
    def build(self,BuleBarrelNum,RedBarrelNum,YellowBarrelNum):


        
        for i in range(BuleBarrelNum):