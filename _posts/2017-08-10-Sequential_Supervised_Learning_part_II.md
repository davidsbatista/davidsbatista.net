---
layout: post
title: Maximum Entropy Markov Models
date: 2017-08-10 00:00:00
tags: [maximum markov models, sequence classifier, tutorial]
categories: [blog]
comments: true
disqus_identifier: 20170810
preview_pic:
---

This is the second part of a series of posts about sequential supervised learning applied to NLP.

In this post I will try to explain how to build a sequence classifier based on a __Logistic Regression__ classifier.

In a [previous post](../../09/Sequential_Supervised_Learning_part_I/) I wrote about the __Naïve Bayes Model__ and how it is connected with the __Hidden Markov Model__. Both are __generative models__, the __Logistic Regression__ classifier however is a __discriminative model__.

A classifier chooses which output label $$y$$ to assign an input $$x$$, by choosing from all the possible $$y_{i}$$ the one that maximizes $$p(y \mid x)$$

The Naive Bayes classifier estimates $$p(y \mid x)$$ indirectly, by applying the Baye's theorem to $$p(y \mid x)$$, and then computing the class conditional distribution/likelihood $$p(x \mid y)$$ and the prior $$p(y)$$.

In contrast a discriminative model directly computes $$p(y \mid x)$$ by discriminating among the different possible values of the class $$y$$ instead of computing a likelihood.

## __Logistic Regression__

Logistic regression is an algorithm use for classification, which is has it's roots in linear regression.

When used to solve NLP tasks, it estimates $$p( y\mid x)$$ by extracting features from the input text and combining them linearly (i.e., multiplying each feature by a weight and then adding them up), and then applying a function to this combination:

$$P(y|x) = \frac{1}{Z} \ \exp \sum_{i=1}^{N} w_{i} \cdot f_{i}$$

where $$f_{i}$$ is a feature and $$w_{i}$$ the weight associated to the feature. The $$\exp$$ surrounding the weight-feature dot product ensures that all values are positive and the denominator $$Z$$ is needed to force all values into a valid probability where the sum is 1.

The extracted features, are binary-valued features, i.e., only takes the values 0 and 1, and are commonly called indicator functions. Each of these features is calculated by a function that is associated with the input $$x$$ and the class $$y$$. Each indicator function is represented as $$f_{i}(y,x)$$, the feature $$i$$ for class $$y$$, given observation $$x$$.

$$P(y|x) = \frac{\exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)} {\sum\limits_{y \in Y} \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)}  $$


### __Trainning__

In training the logistic regression classifier we want to find the ideal weights for each feature that will make the classes of the training example more likely.

Logistic regression is trained with conditional maximum likelihood estimation. This means we choose the parameters $$w$$ that maximize the (log) probability of the $$y$$ labels in the training data given the observations $$x$$.


### __Classification__

In classification logistic regression chooses a class by using the equation defined above to compute the probability of each class and then choose the one that yields the maximum probability.


<!--
#### Multinomial Logistic Regression

Multinomial Logistic Regression, also known as Maximum Entropy, specially in the NLP community is the Logistic regressions extended to a multi-class scenario.

__TODO__: same equations as above but for a for multi-class scenario

https://www.quora.com/What-is-the-relationship-between-Log-Linear-model-MaxEnt-model-and-Logistic-Regression
-->

---

## __Maximum Entropy Markov Model__

The idea of the Maximum Entropy Markov Model (MEMM) is to make use of both the Hidden Markov Models and the Maximum Entropy models also known as multinomial Logistic Regression.

The Hidden Markov Model exploits the Markov assumption to search over a label space, the problem is that this model is restricted on the types of features it can include, basically only the identify feature of the observation.

__MaxEnt (Logistic Regression)__
* classify a single observation into one of a set of discrete classes
* can incorporate arbitrary/overlapping features

__HMM (Hidden Markov Models)__
* sequence tagging — assign a class to each element in a sequence
* independent assumption (cannot incorporate arbitrary/overlapping features)

__Maximum Entropy Markov Models__:
* combines HMM and MaxEnt

Again, we  want to model our learning problem based on a sequence:

$$p(y_{1},\dots,y_{m}∣x_{1},\dots,x_{m}) $$

As in the Hidden Markov Model, in a Maximum Entropy Markov Model this probability is factored into Markov transition probabilities, where the probability of transitioning to a particular label depends only on the observation at that position (i.e., $x_{i}$) and the previous label (i.e., $y_{i}$), we are again using the independence assumption:

$$p(y_{1},\dots,y_{m}∣x_{1},\dots,x_{m}) = $$

$$ = \prod_{i=1}^{m} p(y_{i} \mid y_{1} \dots y_{i-1}, x_{1} \dots, x_{m}) $$

$$ = \prod_{i=1}^{m} p(y_{i} \mid y_{i-1}, x_{1} \dots x_{m}) $$

Having applied these independence assumptions, we then model each term using a log-linear model, just like in the equations above, but with the Markov assumption:

$$ p(y_{i} \mid y_{i-1}, x_{1} \dots x_{m}) = \frac{\exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)} {\sum\limits_{y \in Y} \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)}  $$

$$w_{i}$$ are the weights to be learned, associated to each feature $$f_{i}(x,y)$$.

<!--
http://www.mit.edu/~6.863/spring2011/jmnew/6.pdf

http://www.cs.columbia.edu/~smaskey/CS6998-0412/slides/week13_statnlp_web.pdf
https://www.youtube.com/watch?v=Qn4vZvOEqB0
http://www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf
http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf
-->

### Training (parameters inference): __TODO__

### Testing (decoding a sequence): __TODO__

<!-- greedy inference vs. Viterbi -->



### Observations:

* The advantage for HMM is the use of feature vectors. Transition probability can be sensitive to any word in the input sequence. MEMMs support long-distance interactions over the whole observation sequence.

* We have exponential model for each state to tell us the conditional probability of the next states

* This type of model directly models the conditional distribution of the hidden states given the observations, rather than modeling the joint distribution.

* Label bias problem
