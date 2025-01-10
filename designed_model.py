import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import OneHotEncoder


def designed_CNN():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(65, 158, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1,name='Dropout_least'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu',name='Dense_1'))
    model.add(Dropout(0.25,name='Dropout_last'))
    model.add(Dense(30, activation='softmax',name='Dense_2'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = designed_CNN()
model.summary()



