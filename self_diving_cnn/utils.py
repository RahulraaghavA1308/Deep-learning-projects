
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import imgaug
from imgaug import augmenters as iaa
import cv2
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization, Cropping2D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

# def getName(filePath):
#     return filePath.split('/')[-1]
 
def importDataInfo():
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed','center_name']
    data = pd.read_csv('driving_log.csv',names = columns)
    # data['center']=data['center'].apply(getName)
    data['center'] = data['center_name']

    return data



def balanceData(data,display=True):
    nBin = 31
    samplesPerBin = 1000
    hist, bins = np.histogram(data['steering'], nBin)
    # print(bins)
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)
 
    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
    return data

def loadData(data):
  imagesPath = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    imagesPath.append(f'IMG/{indexed_data[0]}')# retrive image path
    steering.append(float(indexed_data[3]))
  imagesPath = np.asarray(imagesPath)           # store imagepath as numpy array
  steering = np.asarray(steering)   
  return imagesPath, steering
    
    
def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering

def preProcess(img):
    img = img[40:155,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
    
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])   # for validation we don't want to augement the image
            steering = steeringList[index]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))
       

def plot_fit_history(fit_history_obj):
    """Plot loss and validation loss of the trained model on the same graph."""
    plt.plot(fit_history_obj.history['loss'])
    plt.plot(fit_history_obj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

