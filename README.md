# **Behavioral Cloning(Immitation Learning)** 

---

## Summary

This project is based on NVIDIA's [e2e learning for self-driving car](./end-to-end-dl-using-px.pdf). 
The goal is to implent the idea as an assignment in the course of Udacity Self-driving car ND.
Steps are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./examples/drifting.jpg "Drifting"
[image2]: ./examples/cornering.jpg "Cornering issue"
[image3]: ./examples/track2_ex.jpg "Track 2 charp corners"
[image4]: ./examples/recovering_from_offroad.jpg "Recovery from off-track Image"
[image5]: ./examples/recovering_from_leftlane.jpg "Recovery from left lane Image"
[image6]: ./examples/training_and_validation_loss.png "Training & validation loss per epoch"

---
## Intro.

This project is Udacity Self-driving car ND project to implement behavioral cloning based on the NVIDIA's [e2e learning for self-driving car](./end-to-end-dl-using-px.pdf). My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 

To run the code, you first need to check dependencies from [here](https://github.com/udacity/CarND-Term1-Starter-Kit) and install Udacity simulator from [here](https://github.com/udacity/self-driving-car-sim). After that, using the simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

My model consists of a convolution neural network with 5x5, 3x3 filter sizes and depths between 24 and 64 (model.py lines 101-105) 
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 99). 

The model contains dropout layers in order to reduce overfitting (model.py line 107, 109, and 111). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 41). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I also used data augmentation for input images. Performing data augmentation is a form of regularization, enabling our model to generalize better.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

### 1. Dataset

Training data was was captured from the output of Udacity's simulator and selected to keep the vehicle driving on the road. I used a combination of driving in the center of the lane, recovering from the left and right sides of the road, and recovering from off the road to on the road. In the process of data augmentation from initial 25020 to 67923, preprocessing them all in one shot caused memory error. Under assumption that our datasets are too large to fit into memory, i used Keras fit_generator function that allows back-propagation in batch and supports data augmentation on the fly.

For details about how I created the training data, see the next section. 


### 2. Solution Design

The overall strategy for deriving a model architecture was to enhance the model based on DAVE-2 model. My first step was to use a convolution neural network model similar to the it and make feature dimesions a little bigger. I left the depth of architecture(5 conv + 3 FC) might be appropriate because its parameters are only 250 thousand big and proven to be tractable.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to have three dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior, data augmentations were enforced.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 3. Model Architecture

The final model architecture (model.py line 96 - 113) consisted of a convolution neural network with the following layers and layer sizes. The network has about 220 thousand parameters.

My final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 3x160x320 RGB image                           |
| Normalization         | 3x160x320                                     |
| Cropping2D            | 3x65x320                                      |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 24x31x158  |
| RELU                  |                                               |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 36x14x77   |
| RELU                  |                                               |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 48x5x37    |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 64x3x35    |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 64x3x33    |
| RELU                  |                                               |
| Flatten               | output 2112                                   |
| Dropout 0.5           |                                               |
| Fully connected       | input 2112 outputs 100                        |
| Dropout 0.5           |                                               |
| Fully connected       | input 100 outputs 50                          |
| Dropout 0.5           |                                               |
| Fully connected       | input 50 outputs 10                           |



### 4. Creation of the Training Set & Training Process

To capture good driving behavior, I first utilized [sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) to train the network. The provided data from the class has video recording of several laps of driving on the first track and amounts to 25020 images. Driving on the simulator showed that the vehicle is drifting on and off the center of road. 

![alt text][image1]

**correcting drift effect**
I then augmented data by using images of the left side and right side of the road together. Since these images are a little off from the center, giving adjusted steering measurements for the side camera images is required in order to use them as training data. As a result, the vehicle successfully recovered from the left side and right sides of the road back to center, correcting the drift effect. 

**learning sharp cornering**
The observation of the simulation result of the vehicle trained by using only track 1 showed relatively okay behavior on a test drive on the tack 1. When closed to corners, however, the vehicle tend to react rather slowly to the sharp corner, barely avoiding collision with the corner stone. This may be because lack of learning enough cornering. Track 2 seems to have lots of sharp curves. Learning driving on track 2 should probably help solving this cornering issue. 'run_track1.mp4' video was taken at this stage of model, and 1:15 ~ 20 part of video shows this off-center issue when vehicle's cornering.

![alt text][image2]


So, i recorded additional 17883 images of track 2 simulations, summing up to 42903 images. Then i repeated the same correction process on track two in order to get more data points. The recorded video of track 2 simulations contain two laps of normal driving and one with harder try in order to keep vehicle in the center of the road. 

![alt text][image3]

I also added several clips of recovering from the off-road to the center of road. 

![alt text][image4]
![alt text][image5]

Indeed, the result of model trained by driving on both tracks showed much strong tendency to keep the vehicle in the center of road, even around sharp corners, successfully soving the cornering issue. 'run_track2.mp4' video was taken at the final stage of model, and it shows no off-center issue when vehicle's cornering. 

**immitation learning meaning just good as much as the humans do**
Driving straight forward, the vehicle wiggles a little, which was understandable considering my poor jittery arrow key control on many sharp curves of Track 2. Like it said in the behavioral cloning, the model seems to be as much good as the humans do.

**removing left-turn bias**
[sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity has left turn bias due to the data imbalance. This can be overcome using another augmentation of data as we did in the traffic sign classification. In this case, i preprocessed data by giving flip transformations to double the original data so that the model learn driving opposite direction at the same time. 

Now the total number of images of data is 67923.

**driving on two-lan road**
For the sencond track, however, i chose not to add the flip transformations. Because it is two-lane road instead of one in the first track, and flipping the an image would make the vehicle look driving forward on the left lane of road. We probably don't want the model to learn it due to legal issues in some countries. So in order to make sure of the model not to be confused, there are no flipping transformations for the second track images.  

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used 10 epochs and adam optimizer so that manually training the learning rate wasn't necessary. Despite the small number of epochs, both training and validation loss steadily decrease together with little fluctuation druring training. And thier values show very close each other, meaning the model is not overfitting.

![alt text][image6]
