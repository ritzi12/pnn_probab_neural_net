# Probabilitic Neural Network (PNN) for Classification
In this notebook understanding PNN and its related concepts . Concepts of Parzen Window or KDE(kernel density estimate) .Kernel functions as non-parametric method to ascertain data distribution through an example. Implementation of PNN using python for classification tasks.

### Article 
- For detailed description and background go through the blog written by me -[Blog](https://www.analyticsvidhya.com/blog/2023/04/bayesian-networks-probabilistic-neural-network-pnn/)

## Introduction
Bayesian Networks or statistics form an integral part of many statistical learning approaches. It involves using new evidence to modify the prior probabilities of an event. It uses conditional probabilities to improve the prior probabilities, which results in posterior probabilities. In simple terms, suppose you want to ascertain the probability of whether your friends will agree to play a match of badminton given certain weather conditions. Similarly, Bayes Inference forms an integral part of Bayesian Networks as a tool for modeling uncertain beliefs. In this article, we explore one type of Bayesian Networks application, a Probabilistic Neural Network(PNN), and learn in-depth about its implementation through a practical example.

### Learning Objectives

1. Understanding PNN and its related concepts
2. Concepts of Parzen Window or KDE(kernel density estimate)
3. Kernel functions as non-parametric method to a certain data distribution through an example.
4. Implementation of PNN using python for classification tasks

### What is Bayesian Network?
A Bayesian Network uses the Bayes theorem to operate and provides a simple way of using the Bayes Theorem to solve complex problems. In contrast to other methodologies where probabilities are determined based on historical data, this theorem involves the study of probability or belief in a result.

Although the probability distributions for the random variables (nodes) and the connections between the random variables (edges), which are both described subjectively, are not perfectly Bayesian by definition, the model can be considered to embody the “belief” about a complex domain.

In contrast to the frequentist method, where probabilities are solely dependent on the previous occurrence of the event, bayesian probability involves the study of subjective probabilities or belief in an outcome.A Bayesian network captures the joint probabilities of the events the model represents.

### What is Probabilistic Neural Network(PNN)?
A Probabilistic Neural Network (PNN) is a type of feed-forward ANN in which the computation-intensive backpropagation is not used It’s a classifier that can estimate the pdf of a given set of data. PNNs are a scalable alternative to traditional backpropagation neural networks in classification and pattern recognition applications. When used to solve problems on classification, the networks use probability theory to reduce the number of incorrect classifications.

![image](https://github.com/ritzi12/pnn_probab_neural_net/assets/80144294/09b4764c-d993-4359-b11e-6411dc5e84cb)

The PNN aims to build an ANN using methods from probability theory like Bayesian classification & other estimators for pdf. The application of kernel functions for discriminant analysis and pattern recognition gave rise to the widespread use of PNN.

### Concepts of Probabilistic Neural Networks (PNN)
An accepted norm for decision rules or strategies used to classify patterns is that they do so in a way that minimizes the “expected risk.” Such strategies are called “Bayes strategies” and can be applied to problems containing any number of categories/classes.

In the PNN method, a Parzen window and a non-parametric function approximate each class’s parent probability distribution function (PDF). The Bayes’ rule is then applied to assign the class with the highest posterior probability to new input data. The PDF of each class is used to estimate the class probability of fresh input data. This approach reduces the likelihood of misclassification. This Kernel density estimation(KDE) is analogous to histograms, where we calculate the sum of a gaussian bell computed around every data point. A KDE is a sum of different parametric distributions produced by each observation point given some parameters. We are just calculating the probability of data having a specific value denoted by the x-axis of the KDE plot. Also, the overall area under the KDE plot sums up to 1. Let us understand this using an example.

By replacing the sigmoid activation function, often used in neural networks, with an exponential function, a probabilistic neural network ( PNN) that can compute nonlinear decision boundaries that approach the Bayes optimal is formed.

### Parzen Window
The Parzen-Rosenblatt window method, also known as the Parzen-window method, is a well-liked non-parametric approach for estimating a probability density function p(x) for a particular point p(x) from a sample p(xn), which does not necessitate any prior knowledge or underlying distribution assumptions. This process is also known as kernel density estimation.

Estimating the class-conditional density (“likelihoods”) p(x|wi) in classification using the training dataset where p(x) refers to a multi-dimensional sample that belongs to a particular class wi is a prominent application of the Parzen-window technique.

For detailed description of Parzen windows, refer to this [link](https://sebastianraschka.com/Articles/2014_kernel_density_est.html).

### Understanding Kernel Density Estimation
Kernel density estimation(KDE) is analogous to histograms, where we calculate the sum of a gaussian bell computed around every data point. A KDE is a sum of different parametric distributions produced by each observation point given some parameters. We are just calculating the probability of data having a specific value denoted by the x-axis of the KDE plot. Also, the overall area under the KDE plot sums up to 1. Let us understand this using an example.

![image](https://github.com/ritzi12/pnn_probab_neural_net/assets/80144294/115ac17e-e1b0-41eb-814c-1578c57d7891)
Example of different Types of Kernel

Now we will see a distribution of the “sepal length” feature of Iris Dataset and its corresponding kde.
![image](https://github.com/ritzi12/pnn_probab_neural_net/assets/80144294/d9f959e8-5699-4756-a8d2-4f51197c00af)
Distribution of Sepal Length of Iris Dataset

Now using the above-mentioned kernel functions, we will try to build kernel density estimate for sepal length for different values of smoothing parameter(bandwidth).

![image](https://github.com/ritzi12/pnn_probab_neural_net/assets/80144294/e83db7be-b563-49c7-bbfb-e4ca157e29ee)
KDE Plot for different types of the kernel and bandwidth values

As we can see, triangle, gaussian, and epanechnikov give better approximations at 0.8 and 1.0 bandwidth values. As we increase, the bandwidth curve becomes more smooth and flattened, and if we decrease, the bandwidth curve becomes more zigzag and sharp-edged. Thus, bandwidth in PNN can be considered similar to the k value in KNN

### KNN and Parzen Windows
Parzen windows can be considered a k-Nearest Neighbour (KNN) technique generalization. Rather than choosing k nearest neighbors of a test point and labeling it with the weighted majority of its neighbors’ votes, one can consider all observations in the voting scheme and assign their weights using the kernel function.

In the Parzen windows estimation, the interval’s length is fixed, but the number of samples that fall within an interval changes over time. For the k nearest neighbor density estimate, the opposite is true.

## Architecture of PNN
The below image describes the architecture of PNN, which consists of four significant layers, and they are:

* Input Layer
* Pattern Layer
* Summation Layer
* Output Layer
  
Let us now try to understand each layer one by one.

![image](https://github.com/ritzi12/pnn_probab_neural_net/assets/80144294/689e3f78-6c5b-47fd-a009-cf3520be3a9e)

### Input Layer
In this layer, each feature variable or predictor of the input sample is represented by a neuron in the input layer. For example, if you have a sample with four predictors, the input layer should have four neurons. If the predictor is a categorical variable with N categories, then we convert it to an N-1 dummy and use N-1 neurons. We also normalize the data using suitable scalers. The input neurons then send the values to each of the neurons in the hidden layer, the next pattern layer.

### Pattern Layer
This layer has one neuron for each observation in the training data set. A hidden neuron first determines the Euclidean distance between the test observation and the pattern neuron to apply the radial basis kernel function. For the Gaussian kernel, the multivariate estimates can be expressed as,
![image](https://github.com/ritzi12/pnn_probab_neural_net/assets/80144294/b526d1c8-a956-4af8-826b-0fd701babe9d)

where,

For each neuron “i” in the pattern layer, we find the Euclidean distance between the test input and the pattern.

Sigma = Smoothing parameter

d= each feature vector size

x = test input vector

xi = pattern ith neuron vector

### Summation Layer
This layer consists of 1 neuron for each class or category of the target variable. Suppose we have three classes. Then we will have three neurons in this layer. Each Type of pattern layer neuron is joined to its corresponding Type neuron in the summation layer. Neurons in this layer sum and average the values of pattern layer neurons attached to it. vi is the output of each neuron here.

![image](https://github.com/ritzi12/pnn_probab_neural_net/assets/80144294/27d17e63-8946-4f81-9b78-e4c003ede23d) 
*Source: Paper by Specht 1990*

### Output Layer
The output layer predicts the target category by comparing the weighted votes accumulated in the pattern layer for each target category.

## Algorithm of PNN
The following are the high-level steps of the PNN algorithm:

1.  Standardize the input features and feed them to the input layer.

2. In the pattern Layer, each training observation forms one neuron and kernel with a specific smoothing parameter/bandwidth value used as an activation function. For each input observation, we find the kernel function value K(x,y) from each pattern neuron, i.e., training observation.

3. Then sum up the K(x,y) values for patterns in the same class in the summation layer. Also, take an average of these values. Thus, the number of outputs for this layer equals the number of classes in the “target” variable.

4. The final layer output layer compares the output of the preceding layer, i.e., the summation layer. It checks the maximum output for which class label is based on average K(x,y) values for each class in the preceding layer. The predicted class label is assigned to input observation with the highest value of average K(x,y).

## Conclusion
Thus, we saw using PNN; we get high accuracy and f1 score with based on optimal kernel and bandwidth selection.  Also, the best-performing kernels were Gaussian, Triangular, and Epanechnikov kernels. The following are the key takeaways:

1. PNN enables us to build fast and less complex networks involving few layers.

2. We saw various combinations of kernel functions can be employed, and the optimal kernels can be chosen based on performance metrics.

3. PNN  is less time-consuming as it does not involve complex computations.

4. PNN can capture complex decision boundaries due to nonlinearity introduced by kernels which are present as activation functions.

Thus, PNN has wide scope and implementations in various domains.

### References

Papers - 
* [Specht 1990](https://www.sciencedirect.com/science/article/abs/pii/089360809090049Q)
* Pnn Classification()

#### If you found this repo useful please mark it as STAR ! :)




