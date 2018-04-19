# Handwritten-Digits-Classification

In this project, we implement a Multilayer Perceptron Neural Network and evaluate its performance in classifying handwritten digits. The same network is used to analyze a more challenging face dataset and compare the performance of the neural network against a deep neural network using the TensorFlow library.

The input has three datasets. First data set is used for training the network and has predefined labels on it. The second set is used for validation to tune hyper-parameters for Neural Network (number of units in the hidden layer and λ). Finally, the trained network will predict output classifiers on testing data set.

First, we start with implementing neural network with forward pass and back propagation, then use regularization(λ) to control overfitting of data. Then we run the deep neural network code and compare the results with normal neural network. 
We also run the convolutional neural network code and print out the results in the form of a confusion matrix and plot.

From the experiment, it can be observed that a simple neural network with single hidden layer provides better accuracy in relatively less training time. As the number of layers increase, the training time increases due to increased cost and complexity.
