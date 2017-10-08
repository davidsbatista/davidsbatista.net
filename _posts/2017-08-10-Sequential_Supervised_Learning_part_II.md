---
layout: post
title: Maximum Entropy Markov Models
date: 2017-08-10 00:00:00
tags: [maximum markov models, sequence modelling, tutorial]
categories: [blog]
comments: true
disqus_identifier: 20170810
preview_pic:
---

This is the second part of a series of posts about sequential supervised learning applied to NLP.


The Naïve Bayes Model and the Hidden Markov Model, are generative models, because they require us to model the class prior distribution: $$P(Y)$$, and class conditional distribution: $$P(X \mid Y)$$.

A discriminative model directly computes $$P(Y \mid X)$$ by discriminating among the different possible values of the class y rather than first computing a likelihood.

#### Logistic Regression

Logistic regression is an algorithm use for classification, which is has it's roots in linear regression.

When applied to NLP tasks, it estimates $$P( Y\mid X)$$ by extracting features from the input text and combining them linearly (i.e., multiplying each feature by a weight and adding them up), and then applying a function to this combination.

$$P(y|x) = \frac{1}{Z} \ \exp \sum_{i=1}^{N} w_{i} \cdot f_{i}$$

The $$\exp$$ surrounding the weight-feature dot product ensures that all values arae positive and the denominator $$Z$$ exists to transform everything into a valid probability where the sum is 1.

The extracted features, are binary-valued features, i.e., only takes the values 0 and 1, and are commonly called indicator functions. Each of these features is calculated by a function that is associated with the input $$x$$ and the class $$y$$. Each indicator function is represented as $$f_{i}(y,x)$$, the feature $$i$$ for class $$y$$, given observation $$x$$.

$$P(y|x) = \frac{\exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)} {\sum\limits_{y \in Y} \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)}  $$

#### Multinomial Logistic Regression

Multinomial Logistic Regression, also known as Maximum Entropy, specially in the NLP community is the Logistic regressions extended to a multi-class scenario.

__TODO__: same equations as above but for a for multi-class scenario

<!--
https://www.quora.com/What-is-the-relationship-between-Log-Linear-model-MaxEnt-model-and-Logistic-Regression
-->


#### Maximum Entropy Markov Model

The idea of the Maximum Entropy Markov Model (MEMM) is to make use of both the Hidden Markov Models and the Maximum Entropy models.

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
