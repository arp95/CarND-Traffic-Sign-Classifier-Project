#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/histogram.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./data/1.jpg "Traffic Sign 1"
[image5]: ./data/2.jpg "Traffic Sign 2"
[image6]: ./data/3.jpg "Traffic Sign 3"
[image7]: ./data/4.jpg "Traffic Sign 4"
[image8]: ./data/5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/arp95/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
(34799, 32, 32, 3)

* The size of the validation set is ?
(4410, 32, 32, 3)

* The size of test set is ?
(12630, 32, 32, 3)

* The shape of a traffic sign image is ?
(32, 32, 3)

* The number of unique classes/labels in the data set is ?
43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the number of examples for each traffic sign present in the dataset. We would also observe how the variation of examples for each traffic sign inturn affects the model accuracy and hence histogram equalization technique was used to ensure equal distribution of different classes of traffic signs.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the dataset by dividing the images array by 255 to ensure each pixel of the image was between 0 and 1. Normalizing the dataset was important as I was able to increase the accuracy of my model by 2% just by using this simple technique mentioned above.

Next I applied histogram equalization to the image dataset as I found out the dataset didnt contain equal distribution of the different classes of traffic signs. This ensured in applying equalization to the different classes of traffic signs.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description 	| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image | 
| Convolution 3x3     	| 1x1 stride, depth = 32, valid padding, outputs 30x30x32 	|
| RELU	
| Max pooling	      	| 2x2 stride,  outputs 15x15x32 |
| Convolution 3x3	| 1x1 stride, depth = 64, valid padding, outputs 13x13x64 |
| RELU	
| Max pooling	      	| 2x2 stride,  outputs 6x6x64 |
| Convolution 3x3	| 1x1 stride, depth = 128, valid padding, outputs 4x4x128 |
| RELU	
| Max pooling	      	| 2x2 stride,  outputs 2x2x128 |
| Fully connected	| input = 512, output = 84 |
| RELU
| Softmax		| input = 84, output = 43 |
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer to minimize the loss with batch size of 128, number of epochs equal to 60 and learning rate equal to 0.001. Also I applied dropout after each layer except the output layer to ensure my model didnt overfit to the training data and performed well on the validation data as well. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 99.9%
* validation set accuracy of ? 96.8%
* test set accuracy of ? 96.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

Initially I tried using LeNet architecture which consisted of 2 convolutional layers and two fully connected layers. However it didnt perform that well as compared to the current model I used having three convolutional layers and one fully connected layer. I chose it because this was to be taken as a starting point and also I wanted to first check how the project flow was going.

* What were some problems with the initial architecture?

I still doubt that LeNet architecture gives really good results as I see many people performing well on that. Maybe augmenting my dataset would have given good results on that architecture.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

My model architecture consisted of three convolutional layers and one fully connected layer. I read a research paper linked () and found that they performed well with this architecture. So when i tried to test my model on that architecture it gave an accuracy over 95% which was really awesome as I didnt apply any augmentation of that sort on my dataset.


* Which parameters were tuned? How were they adjusted and why?

I tried adjusting the learning rate where I first kept it to 0.001 and got an accuracy of . On keeping it 0.0001 and 0.005 I found out that the accuracy didnt improve that much which showed that using 0.001 was optimal for me. Also i tried using a batch size of 256 but it decreased my models accuracy which inturn made me not go for it.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Important design choices for my model were using three convolutional layers and one fully connected layer instead of the LeNet architecture which uses two convolutional layers and two fully connected layers. I found out using this model gave satisfactory results and hence I opted to go with it for my project. Also applying dropout increased my models accuracy which is self explanoratory as applying dropout ensures my model doesnt overfit to training data and performs well on any data given apart from it.

If a well known architecture was chosen:
* What architecture was chosen? The model i used wasnt a well known architecture.
* Why did you believe it would be relevant to the traffic sign application? - 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? -
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

For first image classifying the sign containing a truck might be difficult as the model might consider it as a bicycle and give a different prediction.

For second and fourth image, I dont see any reason for my model classifying it as incorrect sign.

For the third image, my model might mis-classify it as a road work instead of pedestrian sign. The same seems true for the fifth image which might be a reason for my model mis-classifying it.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons 
  prohibited     			| Vehicles over 3.5 metric tons prohibited | 
| Bicycle crossing    			| Bicycle crossing |
| Pedestrian				| Pedestrian |
| Priority Road	      			| Priority Road |
| Road Work				| Road Work |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.8%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

For the first image, the top five soft max probabilities were

| Probability         	|     Prediction    | 
|:---------------------:|:---------------------------------------------:| 
| .70         			| Vehicles over 3.5 metric tons prohibited | 
| .21     			| Speed limit (60 km/h) |
| .07				| Speed limit (80 km/h) |
| .04	      			| No vehicles |
| .02				| Speed limit (100 km/h) |

For the second image, the top five soft max probabilities were

| Probability         	|     Prediction    | 
|:---------------------:|:---------------------------------------------:| 
| .65         			| Bicycle crossing | 
| .25     			| Bumpy road |
| .06				| Road work |
| .04	      			| Children crossing |
| .01				| SLippery road |

For the third image, the top five soft max probabilities were

| Probability         	|     Prediction    | 
|:---------------------:|:---------------------------------------------:| 
| .63         			| Pedestrian | 
| .27     			| General caution |
| .06				| Children crossing |
| .03	      			| Road narrows on the right |
| .01				| Right of way at next intersection |

For the fourth image, the top five soft max probabilities were

| Probability         	|     Prediction    | 
|:---------------------:|:---------------------------------------------:| 
| .67         			| Priority Road | 
| .21     			| Stop |
| .05				| No passing for vehicles over 3.5 metric tons |
| .04	      			| End of passing for vehicles over 3.5 metric tons |
| .01				| Speed limit (80 km/h) |

For the fifth image, the top five soft max probabilities were

| Probability         	|     Prediction    | 
|:---------------------:|:---------------------------------------------:| 
| .66         			| Road Work | 
| .22     			| Dangerous curve to the right |
| .05				| No passing for vehicles over 3.5 metric tons |
| .04	      			| Speed limit (80 km/h) |
| .02				| Traffic signals |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications? - 


