# DL-Mini-Deep-Learning-Framework
# EE559-Deep Learning (EPFL) - Project 1: "Classification, weight sharing, auxiliary losses"

This project was created alongside Philip Ngo and David Khatanassian (Spring 2021).

In this project we built a miniature deep network framework using only PyTorch's empty tensor and its attributes (not using autograd or the neural netowrk modules.), as well as the standard math library. The objective of this project was to build a framework that enables a user to create simple deep networks. These are tested in a classification task where 1k data points uniformly sampled from the square area {x,y} ∈ [0,1] are labeled as follows:
* 1 , inside disk centered at (0.5,0.5) with radius 1/sqrt(2π)
* 0 , else


### Files (alphabetical order)


### Project overview 
The framework has a superclass _Module_ which is used to pass features between subclasses. There are four types of subclasses:
* Models: General model for building the network. Includes _Sequential_ (emulating nn.Sequential) and _Network_ which follows the standard way to declare a network class:

![Image 1](https://github.com/jpruzcuen/DL-Digit-comparison/blob/main/images/Framework.png)


* Projection
* Activation
* Computation


![Image 2](https://github.com/jpruzcuen/DL-Digit-comparison/blob/main/images/Net.png)
