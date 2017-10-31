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

In a [previous post](../../09/Sequential_Supervised_Learning_part_I/) I wrote about the __Naïve Bayes Model__ and how it is connected with the __Hidden Markov Model__. Both are __generative models__, in contrast the __Logistic Regression__ classifier which is a __discriminative model__, this how this post will start, by explaining this difference.

A machine learning classifier chooses which output label $$y$$ to assign to an input $$x$$, by selecting from all the possible $$y_{i}$$ the one that maximizes $$p(y \mid x)$$

The Naive Bayes classifier estimates $$p(y \mid x)$$ indirectly, by applying the Baye's theorem to $$p(y \mid x)$$, and then computing the class conditional distribution/likelihood $$p(x \mid y)$$ and the prior $$p(y)$$.

In contrast a discriminative model directly computes $$p(y \mid x)$$ by discriminating among the different possible values of the class $$y$$ instead of computing a likelihood. The Logistic Regression classifier is one of such type of classifiers.

## __Logistic Regression__

Logistic regression is an algorithm use for classification, which is has it's roots in linear regression.

When used to solve NLP tasks, it estimates $$p( y\mid x)$$ by extracting features from the input text and combining them linearly (i.e., multiplying each feature by a weight and then adding them up), and then applying a function to this combination:

$$P(y|x) = \frac{1}{Z} \ \exp \sum_{i=1}^{N} w_{i} \cdot f_{i}$$

where $$f_{i}$$ is a feature and $$w_{i}$$ the weight associated to the feature. The $$\exp$$ surrounding the weight-feature dot product ensures that all values are positive and the denominator $$Z$$ is needed to force all values into a valid probability where the sum is 1.

The extracted features, are binary-valued features, i.e., only takes the values 0 and 1, and are commonly called indicator functions. Each of these features is calculated by a function that is associated with the input $$x$$ and the class $$y$$. Each indicator function is represented as $$f_{i}(y,x)$$, the feature $$i$$ for class $$y$$, given observation $$x$$.

$$P(y|x) = \frac{\exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)} {\sum\limits_{y' \in Y} \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y') \bigg)}$$

<!--

http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf

file:///Users/dsbatista/Desktop/CRFs/HIDDEN%20MARKOV%20AND%20MAXIMUM%20ENTROPY%20MODELS.pdf

file:///Users/dsbatista/Desktop/CRFs/Logistic%20Regression.pdf

https://www.quora.com/What-is-the-relationship-between-Log-Linear-model-MaxEnt-model-and-Logistic-Regression
-->

### __Trainning__

In training the logistic regression classifier we want to find the ideal weights for each feature that will make the classes of the training example more likely.

Logistic regression is trained with conditional maximum likelihood estimation. This means we choose the parameters $$w$$ that maximize the (log) probability of the $$y$$ labels in the training data given the observations $$x$$.

### __Classification__

In classification, logistic regression chooses a class by computing the probability of a given observation belonging to each of all the possible classes, then we can choose the one that yields the maximum probability.

$$\hat{y} = \underset{y \in Y} {\arg\max} \ P(y \mid x)$$

$$\hat{y} = \underset{y \in Y} {\arg\max} \frac{\exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y) \bigg)} {\sum\limits_{y' \in Y} \exp \bigg( \sum\limits_{i=1}^{N} w_{i} \cdot f_{i}(x,y') \bigg)}  $$


---

## __Maximum Entropy Markov Model__

<!--

1)
In text-related tasks, the observation probabilities are typically represented as a multinomial distribution over a discrete, finite vocabulary of words, and Baum-Welch training is used to learn parameters that maximize the probability of the observation sequences in the training data.

in particular a representation that describes observations in terms of many overlapping features, such as capitalization, word endings, part-of-speech, formatting, position on the page, and node memberships in WordNet, in addition to the traditional word identity.

For example, when trying to extract previously unseen company names from a newswire article, the identity of a word alone is not very predictive; however, knowing that the word is capitalized, that is a noun, that it is used in an appositive, and that it appears near the top of the article would all be quite predictive (in conjunction with the context provided by the state-transition structure).

Note that these features are not independent of each other.

observations to be parameterized with these overlapping features.

2)
The second problem with the traditional approach is that it sets the HMM parameters to maximize the likelihood of the observation sequence; however, in most text applications, including all those listed above, the task is to predict the state sequence given the observation sequence. In other words, the traditional approach inappropriately uses a generative joint model in order to solve a conditional problem in which the observations are given.



maximum entropy Markov models (MEMMs), in which the HMM transition and observation functions are replaced by a single function

$$P(s \mid s',o)$$

that provides the probability of the current state s given the previous state  s' and the current observation o.

In contrast to HMMs, in which the current observation only depends on the current state, the current observation in an MEMM may also depend on the previous state.

$$P(s \mid s', o)$$

the probability of the transition from state $$s$$ to state $$s'$$ on input $$o$$


State Estimation from Observations
- changes in the recursive Viterbi step
- changes in the Baum-Welch


The use of state-observation transition functions rather than the separate transition and observation functions in HMMs allows us to model transitions in terms of multiple, nonindependent features of observations, which we believe to be the most valuable contribution of the present work

To do this, we turn to exponential models fit by maximum entropy.

Maximum entropy is a framework for estimating probability distributions from data. It is based on the principle that the best model for the data is the one that is consistent with certain constraints derived from the training data, but otherwise makes the fewest possible assumptions. In our probabilistic framework, the distribution with the “fewest possible assumptions” is that which is closest to the uniform distribution, that is, the one with the highest entropy.

As in other conditional maximum entropy models, features do not depend only on the observation but also on the outcome predicted by the function being modeled


Formally, for each previous state $$s'$$ and feature $$a$$, the transition function $$P_{s'}(s \mid o) must have the property that:



maximum-likelihood distribution and has the exponential form



In statistics, generalized iterative scaling (GIS) and improved iterative scaling (IIS) are two early algorithms used to fit log-linear models,[1] notably multinomial logistic regression (MaxEnt) classifiers and extensions of it such as MaxEnt Markov models[2] and conditional random fields. These algorithms have been largely surpassed by gradient-based methods such as L-BFGS[3] and coordinate descent algorithms.[4]

(https://www.wikiwand.com/en/Generalized_iterative_scaling)


Tabela com descricao de algoritmo (training)






file:///Users/dsbatista/Desktop/CRFs/memm-icml2000.pdf
https://liqiangguo.wordpress.com/page/2/
-->

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

http://www.cs.columbia.edu/~smaskey/CS6998/slides/statnlp_week10.pdf


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
