import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('dataModels.xlsx', sheet_name='trainImagesSheet', header=None) #You need `pip install xlrd` to do excel reading
pdf = pd.read_excel('dataModels.xlsx', sheet_name='trainLabelsSheet', header=None) #You need `pip install xlrd` to do excel reading

def trainData():
    return df

def trainLabels():
    return pdf

n_samples = 14970
n_features = 35
trainDat = trainData()
print(trainDat[0][0])
