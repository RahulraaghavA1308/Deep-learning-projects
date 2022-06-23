print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # For removing warnings in terminal

from utils import *
from model import *
# import pandas as pd
import numpy as np
import PIL
from IPython import display

# path = "data\data"

data = importDataInfo()
# print(data.head())
# print("Size of data : ",end=' ')
# print(len(data))

data = balanceData(data=data,display = False)  #Checking if data is balanced and balancing it

# print("Size of data : ",end=' ')
# print(len(data))

# Getting Imagepath and steering data
imagesPath, steerings = loadData(data)
# display.Image(imagesPath[0])
# print(imagesPath[0:5])

# Train-Test-Split
from sklearn.model_selection import train_test_split
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

# augmentImage(imagesPath,steerings)

model = createModel()
# model.summary()
n_epoch = 5
batch_size = 100

history = model.fit(batchGen(xTrain,yTrain,batch_size,0),steps_per_epoch = 200,epochs = n_epoch, 
                    validation_data = batchGen(xVal,yVal,batch_size,0),validation_steps = 50)

print("Enter model Name : ", end = " ")
model_name = input()
model.save(model_name)

print("Model is saved as ", end = '')
print(model_name)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim([0,0.1])

plt.legend(['Training','Testing'])
plt.title('LOSS')
plt.xlabel('EPOCH')
plt.show()
