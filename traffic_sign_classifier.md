#**Traffic Sign Recognition** 

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

[lization.jpgimage1]: ./examples/visualization.jpg "visualization"
[image2]: ./examples/grayscale.jpg "grayscaling"
[image3]: ./examples/limit20.jpg "limit speed 20"
[image4]: ./examples/limit20_translation.jpg "limit speed 20"
[image5]: ./examples/sign1.jpg "traffic sign 1"
[image6]: ./examples/sign2.jpg "traffic sign 2"
[image7]: ./examples/sign3.png "traffic sign 3"
[image8]: ./examples/sign4.png "traffic sign 4"
[image9]: ./examples/sign5.png "traffic sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 * 32 * 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how skewed the data distributed.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale is small and convenient to process by computer.
Furthermore grayscale is enough to identify.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the data should has mean zero and equal variance.

I decided to generate additional data because the trainning samples are very unbalanced.
Some labels have very little samples, while some have many more samples.

To add more data to the the data set, I used the following techniques:
randomly pick from 1 to 5, and increase the trainning samples up to 5000.
1. adjust the brightness of the image radomly
2. rotate the image radomly
3. translate the image radomly
4. shear the image radomly
5. scale the image radomly

Here is an example of an original image and an augmented image:

![alt text][image3] 
![alt text][image4]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | input = 14x14x24 output = 10x10x64     		|
| RELU					|												|
| Max pooling	      	| input = 10x10x64, output = 5x5x64             |
| Dropout	         	| 0.5                                           |
| Fully connected		| input = 1024, output = 480        			|
| Fully connected		| input = 480, output = 43                      |


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with learning rate 0.0001. Every epoch i feed it with a batch of 128 input images. Before dense, I dropped half of the training samples.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.973
* test set accuracy of 0.965

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
lenet, because the dataset is small and the sample of dataset is simple, so i think lenet is fully qualified.

* What were some problems with the initial architecture?
In the beginning, I don't konw how to determine the number of the filters and the output of the fully connected.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I just adjust the feature map size, the number of the filters and the output of the fully connected. 

* Which parameters were tuned? How were they adjusted and why?
I tuned the number of the filters. In order to get more features, I add more filters.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I tuned the number of the filters to find fitting filters in convolution layer. To prevent overfitting, I dropped half of samples before fully connection layer.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Speed limit (20km/h)  | Speed limit (50km/h) 							|
| No passing		    | Yield											|
| Ahead only	    	| Yield     					 				|
| Road work             | Yield                                         |

Except the stop sign, the others are incorrect. I have review code, but can't find the reason.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .100         			| Stop sign   									| 
| .0     				| Children crossing								|
| .0     				| Roundabout mandatory                          |
| .0     				| Speed limit (30km/h)                          |
| .0     				| Speed limit (120km/h)                         |



For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


