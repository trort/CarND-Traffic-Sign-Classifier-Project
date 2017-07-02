# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/train_distri.png "Visualization"
[image2]: ./images/low_contrast.png "Original low contrast image"
[image3]: ./images/normalized.png "normalized image"
[image4]: ./new_ext_examples/P1_200.jpg "Traffic Sign 1"
[image5]: ./new_ext_examples/P2_200.jpg "Traffic Sign 2"
[image6]: ./new_ext_examples/P3_200.jpg "Traffic Sign 3"
[image7]: ./new_ext_examples/P4_200.jpg "Traffic Sign 4"
[image8]: ./new_ext_examples/P5_200.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used numpy shape property to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 with 3 color channels
* The number of unique classes/labels in the data set is 43

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data distribute among all possible classes. Apparently, the distribution is not even. The most common class has 2000 samples, only 10x of the least common class, so the classifier should work reasonably good even without resampling.

![alt text][image1]

Also, I tried to randomly plot some of the images to process. A few of them have contrast so low that I cannot even see the image without preprocessing.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Preprocessing

As a first step, I normalized the image data because the original dataset clearly includes many samples with low contract. I normalized each color channel of each image, to make all points in an image have color ranging between 0 to 1.

Here is an example of the previous shown traffic sign image after normalization. Now the red triangle and deer inside can be clearly seen.

![alt text][image3]

I decided NOT to convert the images to grayscale because in reality, color of sign is one of the most important information for human beings to identify the sign.


#### 2. The final model architecture.

My final model structure is based on the LeNet structure. But since the original LeNet only gives 89% accuracy for the validation set, a few changes are made.
First change is that I am using both 5 x 5 kernels and 3 x 3 kernels, effectively building two CNN networks. The fully connected network uses both flattened concatenated together.

###### The first CNN

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
|	Flatten	|	outputs		400 |

###### The second CNN

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 30x30x3 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x3 				|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 13x13x5 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x5 				|
|	Flatten	|	outputs		180 |

###### The fully connected network

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| outputs of the previous CNNs 180 + 400  							|
| Fully connected    	| 120 hidden neurons 	|
| RELU					|												|
| Dropout					|												|
| Fully connected    	| 84 hidden neurons 	|
| RELU					|												|
| Dropout					|												|
| Output    	| 43 logits 	|

#### 3. Model training

To train the model, I used the AdamOptimizer in tensorflow to minimize the softmax cross entropy. The final set of parameters I used was learning rate of 0.001, 20 epochs with batch size of 128. The epoch number is set to have validation set accuracy converge to maximum value.

#### 4. Final results

The original LeNet structure and parameters only gives a validation accuracy of 0.89. To improve, I first added another convolutional network with kernel size 3x3 instead of 5x5 since a 5x5 kernel may cover too much area on a 32x32 image. Second improve is to add dropout in the fully connected network, which helps to prevent overfitting and increase validation accuracy significantly from 0.92 to 0.95. The dropout is NOT applied in the convolutional network.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.960
* test set accuracy of 0.940

### Test a Model on New Images

#### 1. Five German traffic signs found on the web are tested with the model.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first and fifth images might be difficult to classify because of the tilted angle. The second image might be difficult to classify because of the rotation.  

#### 2. The model's predictions on these new traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									|
| Speed limit (70km/h)     			| Speed limit (60km/h) 										|
| Beware of ice/snow					| Beware of ice/snow											|
| Roundabout mandatory	      		| Roundabout mandatory					 				|
| Road work			| Road work      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This value is significantly lower than the test set accuracy, only because there are only 5 samples. The model fails to predict the second sign, but correctly identified there is a speed limit, with only the value wrong. This is probabily due to low image resolution.

#### 3. Certainty of the model when predicting on each of the five new images.

Here we will look at the softmax probabilities for each prediction. The top 5 softmax probabilities for each image along with the sign type are shown in the notebook.

For the first, fourth, and fifth images, the model is very sure about the predictions, and the predictions are indeed correct.

For the second image, the model is not sure whether this is a speed limit sign of 60 km/h (probability = 0.46) or 80 km/h (probability = 0.22). The image is actually a speed limit sign of 70 km/h. The confusion might come from low image resolution in the training set.

For the third image, the model is relatively sure that this is a beware of ice/snow sign (probability = 0.67). The alternative is a slippery road sign (probability = 0.33). The image is a beware of ice/snow sign. But the slippery road sign has the same shape and color, just with different pattern in the center.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

The visual output of the trained network's feature maps are visualized at the end of the notebook. The second layer outputs are hard to interpret, but the first layer outputs provides some intuition. For the 5 x 5 network, the features include derivative map focusing on different directions, and a black/white map. In contrast, the 3 x 3 network only generated two derivative maps and the third neuron is never activated.

### Possible further improvements
1. Normalized the images to have 0 mean and range [-1, 1] instead of range [0, 1].
2. Visualize the tensorflow computation graph following the tutorial [here](https://www.tensorflow.org/get_started/graph_viz).
3. Implement early terminating to determine the epoch number.
4. Balance the number of samples in each class. [This page](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3) shows how to do that.
5. Alternative network structures.

### Further readings
* About [CNN](http://www.deeplearningbook.org/contents/convnets.html)
* About [optimizer](http://sebastianruder.com/optimizing-gradient-descent/index.html#adam)
