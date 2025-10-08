# MNIST Object Detection
There are numerous examples of classification and regression models online and their ways of operation is rather extensively explored. 
However, object detection models seem to be combining the two types of models in order to detect and classify what they observe. 
This is a study of the ["You Only Look Once" research paper](https://arxiv.org/abs/1506.02640 "You Only Look Once reserach paper") which aims to improve object detection and allow for the instant detection of objects.

## Project Overview
- **Single shot detection:** Predict both bounding boxes and class label in a single forward pass
- **Grid-based predictions:** Slice the image into smaller parts to locate where the items are
- **Two-stage training:** Train a CNN for feature extraction, then add detection layers
- **Synthetic data creation:** Create synthetic data and augment it for improved model training and performance

## Part 1: MNIST digit classification
The first part of the process is to create a classifier that can correctly identify the MNIST digits. This proved to be trivial and extremely well documented.
After 5 minutes of training the model achieved 99.98% accuracy on the train dataset and 99.95% accuracy on the test dataset.

![Graph of the MNIST classifier results](https://github.com/FilipposBatistatos/MNIST_ObjectDetection/figures/MNISTclassifier.png?raw=true)

## Part 2: Create a dataset using the MNIST digits
The MNIST dataset consist of images of handwritten digits, all in the center of the canvas. This provides little challenge to the network, hence a new dataset was created which using the digits and randomly placing them on a blank canvas.
Creating data also meant creating the appropriated labels. The labels are formatted in the same way as the ones on the paper, `class_id x_center y_center width height`.
Interestingly the center, width and height are all normalised to the image size, so something right in the middle of image would be labelled as `0.5 0.5`.
This is an optimisation for machine learning performance. 

There are some challenges that come up when creating your own dataset, of course it needs to be varied hence the size of the numbers and their rotation and position is random. This creates other issues though.
![Graph of the MNIST classifier results](https://github.com/FilipposBatistatos/MNIST_ObjectDetection/figures/SyntheticDataOverlappingExample.png?raw=true)

With some more optimisation in the placing algorithm we can achieve better results.
![Graph of the MNIST classifier results](https://github.com/FilipposBatistatos/MNIST_ObjectDetection/figures/SyntheticDataExample.png?raw=true)
But we will hold onto some of these images for experimentation.

## Part 3: Create a CNN
A critical part of this architecture is the Convolutional Layers. They act as a feature extractor on the image, allowing the model to identify both the location and the class of features.
This is similar to Part 1, but with different architecture. The research paper states that the CNN was trained to the data for classification. It is assumed that it would be given an image including a multiple of objects and it would be able to classify all of them.
For the sake of experimentation two CNNs were trained and save one on the synthetic, multiclass dataset and one on the original MNIST data set. 
Interestingly both of the networks trained faster.

## Part 4: Object detection - regression
This step involves adding two fully connected layers onto the end of the pre-trained CNN.
Then the model is trained again with a new criterion which includes the classification cross entropy, the binary cross entropy for the multi class detection and MSE loss for the position of each class. It is worth mentioning that the regression loss was weighted 5 time higher than then classification loss.

Three networks were trained and accessed. Their architectures were all the same.
`Model A`: Pretrained CNN on synthetic, multiclass dataset with two fully connected layers
`Model B`: Pretrained CNN on MNIST dataset with two fully connected layers
`Model C`: Freshly initialised CNN with two fully connected layers

The networks were trained for 10 epoch, accessed and saved then trained for an additional 10 epoch.

# Results
![Graph of the MNIST classifier results](https://github.com/FilipposBatistatos/MNIST_ObjectDetection/figures/DetectionResults.png?raw=true)

At 10 epoch all three networks struggled, the classification performance of models `A` and `B` were strong with accuracy of 96% and 92% respectively. However, Model C was struggling with accuracy in the low 50%.
At 20 epoch Models A and B were able to locate and classify with significant accuracy the images in the above figure are classified using model A - the strongest performer.
Model C was showing very slow improvement, hence further development was haulted. 

The results are interesting, the model sometimes detects a number twice, and others completely misses a number, but interestingly accurately predicts the overlaping image that it was tested on even though it was never trained on it. It seems to particularly struggle with the number 2. 

# Closing thoughts 
This has been an rewarding exercise in recreating the YOLO paper. 
Techniques learned here can be applied to a variety of other Machine Learning tasks. Most importantly pre-training when trying to achieve multi-task learning as well as to how to train for that. 
