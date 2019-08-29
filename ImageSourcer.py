import cv2
import numpy as np
import os
import pandas as pd
from imagePInternals import *
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import openpyxl

Directory = 'C:/Users/hamis/Dropbox/Coding/Hamish/Vectorizer Python/shapes'
os.chdir(Directory)

#Collects the data from the image library, and utilises the image processor
folders, labels, images, imFeatures = ['triangle', 'star', 'square', 'circle'], [], [], []
ID = 1
for folder in folders:
    print(folder)
    for path in os.listdir(os.getcwd() +'/'+folder):
        imageFeature = imageProcessor(folder+'/'+path)
        img = imageFeature.img
        images.append("ID" + str(ID))
        labels.append(folder)
        imFeatures.append(imageFeature.detail())
        ID +=1


trainLabels = []
trainImages = []

for image, label, features in zip(images, labels, imFeatures):
    DFCSet = []
    PerimeterSet = []
    for i in range(16):
            Perimeter = features["Perimeter"]
            try:
                PerimeterSet.append(Perimeter[i])
            except:
                PerimeterSet.append(0)
        #Appends the image data right next to the image itself to for the X value
    for i in range(16):
        DFC = features["DistFromCentre"]
        try:
            DFCSet.append(DFC[i])
        except IndexError:
            DFCSet.append(0)
            continue
    InsertList = []
    InsertList.append(image)
    InsertList.append(features["Vertices"])
    InsertList.extend(PerimeterSet)
    InsertList.append(features["SumOfAngles"])
    InsertList.extend(DFCSet)
    trainImages.append(InsertList)                    
    trainLabels.append(label)
    

#trainImages = np.array(trainImages, dtype = object)
#testImages = np.array(testImages, dtype = object)
#trainLabels = np.array(trainLabels, dtype = object)
#testLabels = np.array(testLabels, dtype = object)

#Goes through the process of writing each data model to an excel sheet

dataModels = openpyxl.Workbook()
trainImagesSheet = dataModels.active
trainImagesSheet = dataModels.create_sheet("xd")
trainImagesSheet.title = "trainImagesSheet"

rowNumber = 1

for trainImage in trainImages:
    columnNum = 1
    for i in range(len(trainImage)):
        trainImagesSheet.cell(row = rowNumber, column = columnNum).value = (trainImage[i])
        columnNum += 1
    rowNumber +=1

rowNumber = 1

trainLabelsSheet = dataModels.active
trainLabelsSheet = dataModels.create_sheet("xd")
trainLabelsSheet.title = "trainLabelsSheet"
for trainLabel in trainLabels:
    trainLabelsSheet.cell(row = rowNumber, column = 1).value = trainLabel

    rowNumber += 1
    

dataModels.save(filename = 'C:/Users/hamis/Documents/SVMModels/dataModels.xlsx')





