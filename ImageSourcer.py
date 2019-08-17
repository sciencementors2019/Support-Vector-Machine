import cv2
import numpy as np
import os
import pandas as pd
from imagePInternals import *
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

Directory = 'C:/Users/hamis/Dropbox/Coding/Hamish/Vectorizer Python/shapes'
os.chdir(Directory)

#Collects the data from the image library, and utilises the image processor
folders, labels, images, imFeatures = ['triangle', 'star', 'square', 'circle'], [], [], []
for folder in folders:
    print(folder)
    for path in os.listdir(os.getcwd() +'/'+folder):
        imageFeature = imageProcessor(folder+'/'+path)
        img = imageFeature.img
        images.append(img)
        labels.append(folders.index(folder))
        imFeatures.append(imageFeature.detail())

#Sorts the data into a Train/Test ratio of 1:5 / 20%:80%
toTrain = 0
trainLabels, testLabels = [],[]
trainImages, testImages = [], []
trainImages.append([])
testImages.append([])
for image, label, features in zip(images, labels, imFeatures):
    if toTrain < 5:
        #Appends the image data right next to the image itself to for the X value
        trainImages.append([image, features["Vertices"], features["Perimeter"], features["SumOfAngles"], features["DistFromCentre"]] )
        trainLabels.append(label)
        toTrain += 1
    else:
        testImages.append([image, features["Vertices"], features["Perimeter"], features["SumOfAngles"], features["DistFromCentre"]])
        testLabels.append(label)
        toTrain = 0

#trainImages = np.array(trainImages, dtype = object)
#testImages = np.array(testImages, dtype = object)
#trainLabels = np.array(trainLabels, dtype = object)
#testLabels = np.array(testLabels, dtype = object)

#Creates and sklearn model that can be used for the SVM
trainModel = pd.DataFrame(trainImages)
testModel = pd.DataFrame(testImages)
trainLabels = pd.DataFrame(trainLabels)
testLabels = pd.DataFrame(testLabels)

#Saves as .csv file
filenameTestModel = 'C:/Users/hamis/Documents/SVMModels/shape_test_model.csv'
filenameTrainModel = 'C:/Users/hamis/Documents/SVMModels/shape_train_model.csv'
filenameTestLabel = 'C:/Users/hamis/Documents/SVMModels/shape_test_label.csv'
filenameTrainLabel = 'C:/Users/hamis/Documents/SVMModels/shape_train_label.csv'
pickle.dump(trainModel, open(filenameTrainModel, 'wb'))
pickle.dump(testModel, open(filenameTestModel, 'wb'))
pickle.dump(trainLabels, open(filenameTrainLabel, 'wb'))
pickle.dump(testLabels, open(filenameTestLabel, 'wb'))



