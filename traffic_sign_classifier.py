import tensorflow as tf

import pickle

training_file    = "/home/brian/Documents/Training/Udacity/Self_Driving_Car/data/train.p"
validation_file  = "/home/brian/Documents/Training/Udacity/Self_Driving_Car/data/valid.p"
testing_file     = "/home/brian/Documents/Training/Udacity/Self_Driving_Car/data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Visualise the dataset
import numpy as np
# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train).size

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#test = X_train[1]
#print(test)

### Visualisations
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.

plt.figure(1)
plt.subplot(211)
plt.imshow(X_train[20410])

### PreProcess