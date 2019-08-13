import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('D:\\smc2019\\svM\\Data.xls.xlsx', sheet_name='Data', header=None) #You need `pip install xlrd` to do excel reading

def rdf():
    return df

def read(x,y):
    return df[x][y]