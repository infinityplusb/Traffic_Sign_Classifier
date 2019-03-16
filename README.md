## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image_split]: histogram_of_image_split.png "Class Split"
[sample_images]: sample_images.png "Sample Images"
[le_net]: LeNet.png "LeNet Classic"
[new_signs]: new_signs.png "New Signs"
[top_n_predictions]: top_n_predictions.png "Top Predictions"

Overview
---
#### The submission includes a basic summary of the data set.
In this project, the intention is to build a CNN architecture to read in 32x32 images to recognise images.
The Dataset contains `34799` images for training, `4410` for validation and `12630` for testing. The dataset has `43` classes of image

![Classes][image_split]

A sample of 2 images of each class are included below

![Samples][sample_images]

#### The submission describes the preprocessing techniques used and why these techniques were chosen.

The initial attempt to create an accurate model was met with futility.
As more data would be required to accurately train the some augmentation methods were used to supplement the existing dataset.
Firstly, some of the images were flipped.
Some images can be flipped horizontally, some vertically, and would still belong in the same class. Like `straight forward signs`.
Others like `turn right` signs can be flipped but would now be `turn left`.

As such, the project iterates through each class, and where the class images can be flipped horizontally they were, and then appended onto the new array
```
extended_train = np.append(extended_train, X_train[y_train == image][:, :, ::-1, :], axis = 0)
```
Where images could be flipped horizontally and below to a new class, they were, and the new class was added to the y_extension array
```
flip_class = cross_flippable[cross_flippable[:, 0] == image][0][1]
extended_train = np.append(extended_train, X_train[y_train == flip_class][:, :, ::-1, :], axis = 0)
# Fill labels for added images set to current class.
extended_y = np.append(extended_y, np.full((extended_train.shape[0] - extended_y.shape[0]), image, dtype = int))
```

Vertical flips were done
```
extended_train = np.append(extended_train, extended_train[extended_y == image][:, ::-1, :, :], axis = 0)
```
and vertical and horizontal flips
```
extended_train = np.append(extended_train, extended_train[extended_y == image][:, ::-1, ::-1, :], axis = 0)
```

This increased the sample size from `34799` to `59788`

Further, adding 5 skewness and rotation transforms to the images, `xtra_train_extended[k] = cv2.warpAffine(extended_train[i],M,(32,32))` increased the sample size to `298940` .


#### The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

![Classic LeNet Architecture][le_net]

The architecture is exactly a LeNet architecture, with 2 convolution layers with relu activation and pooling. This is followed by 3 fully connected layers.
The first layer takes in 32x32 images, and outputs 6 layers with a size of 28x28.
The pooling layer, outputs 6 layers of size 14x14.
An additional dropout layer is added to reduce overfitting.

The second convolution layer outputs 16 10x10 layers. Pooling again reduces the layer size, by outputting 5x5. Again a dropout layer is added to reduce overfitting.

The fully connected layers start with 120 nodes, reducing down to 84, then finally to the number of classes being predicted (43). The connection between layers is triggered by relu activations.

#### The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.


- Learning Rate  : 0.001     
- First Dropout  : 0.25    
- Second Dropout : 0.75    
- Batch Size     : 500    
- Epochs        : 100    
- Valid Accuracy : 0.953    
- Test Accuracy : 0.932    


#### The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.
The network performs considerably poorly on my 7 chosen new signs.

![New Signs][new_signs]

Understandably one (the first) was not in the dataset (no left turn) but in general the performance was quite poor, with a success rate of 0.286 or 2 images out of 7.
This is potentially due to the fact that none of the signs have borders (with random backgrounds) like the trained images, that the network was trained in colour and so is perhaps affected by brightness or colour saturation within the image, rather than picking up the features.

#### The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

The top n predictions shows the signs that are accurately predicted (roundabout and no entry), but gives no obvious indication as to why the other signs are performing poorly.

![Top Predictions][top_n_predictions]

Nearly all signs are predicted with 100% accuracy for the first value, with no other likely predictions which may indicate an issue with the implementation rather than of the model itself.
