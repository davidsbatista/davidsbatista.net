---
layout: post
title: Maximum Entropy Markov Models and Logistic Regression
date: 2017-08-10 00:00:00
tags: [maximum markov models, sequence classifier, tutorial]
categories: [blog]
comments: true
disqus_identifier: 20170810
preview_pic: /assets/images/2017-08-19-Maximum_Entropy_Markov_Models.png
---


This is the second part of a series of posts about sequential supervised learning applied to NLP. It can be seen as a follow up on the previous post, where I tried do explain the relationship between HMM and Naive Bayes. In this post I will try to explain how to build a sequence classifier based on a Logistic Regression classifier, i.e.,  using a discriminative approach.

<br>

## __Discriminative vs. Generative Models__

In a [previous post](../../09/HHM_and_Naive_Bayes/) I wrote about the __Naïve Bayes Model__ and how it is connected with the __Hidden Markov Model__. Both are __generative models__, in contrast the __Logistic Regression__ classifier which is a __discriminative model__, this post will start, by explaining this difference.

<!-- \newcommand{\argmax}[1]{\underset{#1}{\operatorname{arg}\,\operatorname{max}}\;} -->

In general a machine learning classifier chooses which output label $$y$$ to assign to an input $$x$$, by selecting from all the possible $$y_{i}$$ the one that maximizes $$P(y\mid x)$$.

The Naive Bayes classifier estimates $$p(y \mid x)$$ indirectly, by applying the Baye's theorem, and then computing the class conditional distribution/likelihood $$P(x \mid y)$$ and the prior $$P(y)$$.

$$ \hat{y} = \underset{y}{\arg\max}\ P(y \mid x) = \underset{y}{\arg\max} \ P(x \mid y) \cdot P(y)$$

This indirection makes Naive Bayes a generative model, a model that is trained to generated the data $$x$$ from the class $$y$$. The likelihood $$p(x \mid y)$$, means that we are given a class $$y$$ and will try to predict which features to see in the input $$x$$.

In contrast a discriminative model directly computes $$p(y \mid x)$$ by discriminating among the different possible values of the class $$y$$ instead of computing a likelihood. The Logistic Regression classifier is one of such type of classifiers.

$$ \hat{y} = \underset{y}{\arg\max} \ P(y \mid x)$$

<br>

## __Logistic Regression__

Logistic regression is supervised machine learning algorithm used for classification, which is has it's roots in linear regression.

When used to solve NLP tasks, it estimates $$p( y\mid x)$$ by extracting features from the input text and combining them linearly i.e., multiplying each feature by a weight and then adding them up, and then applying a function the exponential function to this combination:

$$P(y|x) = \frac{1}{Z} \ \exp \sum_{i=1}^{N} w_{i} \cdot f_{i}$$

where $$f_{i}$$ is a feature and $$w_{i}$$ the weight associated to the feature. The $$\exp$$ (i.e., exponential function) surrounding the weight-feature dot product ensures that all values are positive and the denominator $$Z$$ is needed to force all values into a valid probability where the sum is 1.

The extracted features, are binary-valued features, i.e., only takes the values 0 and 1, and are commonly called indicator functions. Each of these features is calculated by a function that is associated with the input $$x$$ and the class $$y$$. Each indicator function is represented as $$f_{i}(y,x)$$, the feature $$i$$ for class $$y$$, given observation $$x$$.

$$P(y|x) = \frac{\exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)} {\sum\limits_{y' \in Y} \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y') \bigg)}$$

<!--

http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf

file:///Users/dsbatista/Desktop/CRFs/HIDDEN%20MARKOV%20AND%20MAXIMUM%20ENTROPY%20MODELS.pdf

file:///Users/dsbatista/Desktop/CRFs/Logistic%20Regression.pdf

https://www.quora.com/What-is-the-relationship-between-Log-Linear-model-MaxEnt-model-and-Logistic-Regression
-->

### __Trainning__

By training the logistic regression classifier we want to find the ideal weights for each feature, that is, the weights that will make training example fit the classes to which they belong.

Logistic regression is trained with conditional maximum likelihood estimation. This means that we will choose the parameters $$w$$ that maximize the probability of the $$y$$ labels in the training data given the observations $$x$$:

$$\hat{w} = \underset{w}{\arg\max} \sum_{j} \log \ P(y^{j} \mid y^{j})$$

__TODO__: why log?

The objective function to maximize is:

$$L(w) = \sum_{j} \log\ P(y^{j} \mid y^{j})$$

which by replacing with expanded form presented before and by applying the division log rules, takes the following form:

$$ L(w) = \sum\limits_{j} \log \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i} (x^{j},y^{j}) \bigg) - \sum\limits_{j} \log {\sum\limits_{y' \in Y} \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x^{j},y'^{j}) \bigg)}$$

Maximize this objective, i.e. fiding the optimal weights, are typically solved by methods like stochastic gradient ascent, L-BFGS, or conjugate gradient.


### __Classification__

In classification, logistic regression chooses a class by computing the probability of a given observation belonging to each of all the possible classes, then we can choose the one that yields the maximum probability.

$$\hat{y} = \underset{y \in Y} {\arg\max} \ P(y \mid x)$$

$$\hat{y} = \underset{y \in Y} {\arg\max} \frac{\exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)} {\sum\limits_{y' \in Y} \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y') \bigg)}  $$

<br>

---

<br>


## __Maximum Entropy Markov Model__

The idea of the Maximum Entropy Markov Model (MEMM) is to make use of both the HMM framework to __predict sequence labels given an observation sequence, but incorporating the multinomial Logistic Regression (aka Maximum Entropy)__, which gives freedom in the type and number of features one can extract from the observation sequence.
<!--

The HMM taggingmodel is based on probabilities of the form P(tag|tag) and P(word|tag). That means that if we want to include some source of knowledge into the tagging process, we must find a way to encode the knowledge into one of these two probabilities. But many knowledge sources are hard to fit into these models. For example, we saw in Sec. ?? that for tagging unknown words, useful features include capitalization, the presence of hyphens, word endings, and so on. There is no easy way to fit these features into an HMM-style model; as we discussed in Ch. 5, P(capitalization|tag), P(hyphen|tag), P(suffix|tag), and so on
-->

















The MEMM was proposed as way to have richer set of observation features:

* _"a representation that describes observations in terms of many overlapping features, such as capitalization, word endings, part-of-speech, formatting, position on the page, and node memberships in WordNet, in addition to the traditional word identity."_

and also to solve the prediction problem with a discriminative approach:

* _"the traditional approach sets the HMM parameters to maximize the likelihood of the observation sequence; however, in most text applications [...] the task is to predict the state sequence given the observation sequence. In other words, the traditional approach inappropriately uses a generative joint model in order to solve a conditional problem in which the observations are given._"




In the Maximum Entropy Markov Models the transition and observation functions (i.e., the HMM matrices $$A$$ and $$B$$ from the previous post) are replaced by a single function:

$$P(s \mid s',o)$$

the probability of the current state $$s$$ given the previous state $$s'$$ and the current observation $$o$$. The figure below shows this difference in computing the state/label/tag transitions.

<figure>
  <img style="width: 65%; height: 65%" border="5" src="/assets/images/2017-08-19-HMM_and_MEMM.png">
</figure>

In contrast to HMMs, in which the current observation only depends on the current state, the current observation in an MEMM may also depend on the previous state.

Note that the HMM model includes distinct probability estimates for each transition and observation, while the MEMM gives one probability estimate per hidden state, which is the probability of the next tag given the previous tag and the observation.

unlike the HMM, the MEMM can condition on any useful feature of the input observation. In the HMM this wasn’t possible because the HMM is likelihood based, and hence would have needed compute the likelihood of each feature of the observation.

We will use MaxEnt for this last piece, estimating the probability of each local tag given the previous tag, the observed word, and, as we will see, any other features we want to include.

## MaxEnt

 - $$P(s \mid s',o)$$ is split into $$|S|$$ separately trained transition functions 
- Each of these functions is given by an exponential model

The use of state-observation transition functions rather than the separate transition and observation functions as in HMMs allows us to model transitions in terms of multiple, nonindependent features of observations

To do this, we turn to exponential models fit by maximum entropy.

maximum-likelihood distribution and has the exponential form

The original MEMM paper, published in 2000, used a generalized iterative scaling (GIS) algorithm to fit the multinomial logistic regression (MaxEnt). That algorithms has been largely surpassed by gradient-based methods such as L-BFGS.



<!--


Tabela com descricao de algoritmo (training)

file:///Users/dsbatista/Desktop/CRFs/memm-icml2000.pdf
https://liqiangguo.wordpress.com/page/2/



State Estimation from Observations
- changes in the recursive Viterbi step
- changes in the Baum-Welch

-->

























<!--
Again, we  want to model our learning problem based on a sequence:

$$p(y_{1},\dots,y_{m}∣x_{1},\dots,x_{m}) $$

As in the Hidden Markov Model, in a Maximum Entropy Markov Model this probability is factored into Markov transition probabilities, where the probability of transitioning to a particular label depends only on the observation at that position (i.e., $x_{i}$) and the previous label (i.e., $y_{i}$), we are again using the independence assumption:

$$p(y_{1},\dots,y_{m}∣x_{1},\dots,x_{m}) = $$

$$ = \prod_{i=1}^{m} p(y_{i} \mid y_{1} \dots y_{i-1}, x_{1} \dots, x_{m}) $$

$$ = \prod_{i=1}^{m} p(y_{i} \mid y_{i-1}, x_{1} \dots x_{m}) $$

Having applied these independence assumptions, we then model each term using a log-linear model, just like in the equations above, but with the Markov assumption:

$$ p(y_{i} \mid y_{i-1}, x_{1} \dots x_{m}) = \frac{\exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)} {\sum\limits_{y \in Y} \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)}  $$

$$w_{i}$$ are the weights to be learned, associated to each feature $$f_{i}(x,y)$$.
-->

<!--
http://www.mit.edu/~6.863/spring2011/jmnew/6.pdf
http://www.cs.columbia.edu/~smaskey/CS6998/slides/statnlp_week10.pdf
http://www.cs.columbia.edu/~smaskey/CS6998-0412/slides/week13_statnlp_web.pdf
https://www.youtube.com/watch?v=Qn4vZvOEqB0
http://www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf
http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf
-->

<br>

---

<br>

### __MEMM Important Observations__

* The advantage for HMM is the use of feature vectors. Transition probability can be sensitive to any word in the input sequence. MEMMs support long-distance interactions over the whole observation sequence.

* We have exponential model for each state to tell us the conditional probability of the next states

* This type of model directly models the conditional distribution of the hidden states given the observations, rather than modelling the joint distribution.

* It has the Markov Assumption just like HMM

* Label bias problem


## __References__

* [Chapter 7: "Logistic Regression" in Speech and Language Processing. Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/7.pdf)

* [Maximum Entropy Markov Models for Information Extraction and Segmentation](http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf)

* [LxMLS - Lab Guide July 16, 2017 - Day 2 "Sequence Models"](http://lxmls.it.pt/2017/LxMLS2017.pdf)
