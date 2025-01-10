import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import designed_model


# Hyperparameter settings
batch_size=50
epochs=10
validation_split=0.1


#load CNN
model = designed_model.designed_CNN()
model.summary()


#train CNN
label_input = OneHotEncoder(sparse = False).fit_transform(label.reshape(label.shape[0],1))
history = model.fit(data, label_input, batch_size,epochs, validation_split)


# remove the final layer
feature_ext = Model(inputs=model.input, outputs=model.get_layer('Dropout_last').output)
feature_ext.save('./data/model/feature_extractor.h5')


