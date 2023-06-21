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
