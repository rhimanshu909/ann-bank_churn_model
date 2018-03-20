#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1.) Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]    #removing dummy variable trap

# Splitting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2.) ANN

# Keras
import keras
from keras.models import Sequential #initialize ann
from keras.layers import Dense #add layers

# Initialize the ANN -defining as sequence of layers or defining a graph
classifier = Sequential()

# Add the input and first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
# Keras 2 API: `Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")

# Add second hidden layer (optional in this)
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Add output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to training set
classifier.fit(X_train, Y_train, batch_size=10, epochs=100)

# 3.) Predictions and Evaluations

# Predicting the test_set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5) # Defining threshold as 50% as confusion matrix won't need probability

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred) # Accuracy= (1526+158)/2000 = 84.2
