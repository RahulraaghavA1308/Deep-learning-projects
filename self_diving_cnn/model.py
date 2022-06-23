import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization, Cropping2D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

def createModel():
  model = Sequential()
 
  model.add(Conv2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
  model.add(Conv2D(36, (5, 5), (2, 2), activation='elu'))
  model.add(Conv2D(48, (5, 5), (2, 2), activation='elu'))
  model.add(Conv2D(64, (3, 3), activation='elu'))
  model.add(Conv2D(64, (3, 3), activation='elu'))
 
  model.add(Flatten())
  model.add(Dense(100, activation = 'elu'))
  model.add(Dense(50, activation = 'elu'))
  model.add(Dense(10, activation = 'elu'))
  model.add(Dense(1))
 
  model.compile(Adam(lr=0.0001),loss='mse')
  return model