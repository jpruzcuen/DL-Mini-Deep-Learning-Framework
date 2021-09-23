# DL-Mini-Deep-Learning-Framework
# EE559-Deep Learning (EPFL) - Project 1: "Classification, weight sharing, auxiliary losses"

This project was created alongside Philip Ngo and David Khatanassian (Spring 2021).

In this project we built a miniature deep network framework using only PyTorch's empty tensor and its attributes (not using autograd or the neural netowrk modules.), as well as the standard math library. The objective of this project was to build a framework that enables a user to create simple deep networks. These are tested in a classification task where 1k data points uniformly sampled from the square area {x,y} ∈ [0,1] are labeled as follows:
* 1 , inside disk centered at (0.5,0.5) with radius 1/sqrt(2π)
* 0 , else


### Files (alphabetical order)


### Project overview 
The framework has a superclass _Module_ which is used to pass features between subclasses. There are four types of subclasses:
* Models: General model for building the network. Includes the _Sequential_ class (emulating nn.Sequential) and _Network_ class, used for creating a networks in the more standard way. A sample network created with the _Network_ is shown below: 

<p align="center">
  <img src="https://github.com/jpruzcuen/DL-Mini-Deep-Learning-Framework/blob/main/images/Net.png" width="40%" height="40%">
</p>
  
* Projection: Only contains the _Linear_ class which is used for initializing linear convolutional layers. Each of its instances stores the layer parameters (gradients, activation function, input/output value and activated output value). After initializing a  _Linear_ instance it is added to an internal list of _Module_ 
* Activation: Contains the modules of the activation functions, including ReLU, tanh and identity. 
* Computation: Contains the modules of the loss functions, including MSE loss, Binary Cross Entropy loss and Softmax loss (BCE with softmax instead of sigmoid). 

An overview of the framework is shown below:

<p align="center">
  <img src="https://github.com/jpruzcuen/DL-Mini-Deep-Learning-Framework/blob/main/images/Framework.png" width="60%" height="60%">
</p>
