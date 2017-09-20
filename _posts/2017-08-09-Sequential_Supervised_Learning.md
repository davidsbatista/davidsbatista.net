---
layout: post
title: Sequential Supervised Learning
date: 2017-08-09 00:00:00
tags: [conditional random field, hidden markov models, sequence modeling, tutorial]
categories: [blog]
comments: true
disqus_identifier: 20170809
preview_pic:
---

The classical problem in Machine Learning is to learn a classifier that can distinguish between two or more classes, i.e., that can accurately predict a class for a new object given training examples of objects already classified.

In NLP typical examples are for instance, classifying an email as spam or not spam, classifying a movie into genres, classifying a news article into topics, etc.

However there is another type of prediction problems which involve structure. A classical example in NLP is part-of-speech tagging, in this scenario, each $$x_{i}$$ describes a word and each $$y_{i}$$ the associated part-of-speech of the word $$x_{i}$$ (e.g.: _noun_, _verb_, _adjective_, etc.).

Another example, is named-entity recognition, in which, again, each $$x_{i}$$ describes a
word and $$y_{i}$$ is a semantic label associated to that word (e.g.: _person_, _location_, _organization_, _event_, etc.).

In both examples the data consist of sequences of $(x, y)$ pairs. We want to model our learning problem based on that sequence:

$$p(y_1, y_2, \dots, y_m \mid x_1, x_2, \dots, x_m)$$

In NLP problems these sequences can have a sequential correlation. That is, nearby $x$ and $y$ values are likely to be related to each other. For instance, in English, it's common that after the preposition _to_ the part-of-speech tag associated to the following word is a verb.


Note that there are other machine learning problems which also involve sequences but are clearly different. For instance, in time-series, there is also a sequence, but we want to predict a value $$y$$ at point $$t+1$$, and we can use all the previous true observed $$y$$ to predict. In sequential supervised learning we must predict all $$y$$ values in the sequence.

The paper [Machine Learning for Sequential Data: A Review by Thomas G. Dietterich](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf) contains many more examples, and is a good introduction to the supervised sequential learning problem.

# __Hidden Markov Model__

The Hidden Markov Model (HMM) is one of the many possible algorithms to learn how to classify sequences. It has it's roots on the Naive Bayes, and HMM can be seen as a sequential extension to the Naive Bayes model.

#### Naive Bayes classifier

It's based on the Bayes' theorem, where $$y$$ it's a class and $$\vec{x}$$ is a feature vectors associated to an observation:

$$ p(y \mid \vec{x}) = \frac{p(y) \cdot p(\vec{x} \mid y)}{p(\vec{x})} $$

when iterating over all the classes for an observation, the denominator is always the same, we will always asking about the most likely class for the same $$\vec{x}$$, we can then simplify the formula:

$$p(y \mid \vec{x}) = p(y) \cdot p(\vec{x} \mid y) $$

The numerator can also be written as a joint probability:

$$p(y) \cdot p(\vec{x} \mid y) = p(y,\vec{x})$$

by applying the chain rule: $$p(x_{1}, x_{2}, \dots, x_{m}) = \prod_{i=1}^{m} p(x_{i} \mid x_{i-1}, x_{i-2}, \dots, x_{1})$$ we can transform $$p(y,\vec{x})$$ into:

$$ p(y \mid \vec{x}) = p(y) \prod_{i=1}^{m} p(x_{i} \mid x_{i-1}, x_{i-2}, \dots, x_{1}, y) $$

apply the Naïves Bayes assumption: $$p(x_{i} \mid y,x_{j}) = p(x_{i} \mid y)$$ with $$i \neq j$$ i.e. "each feature is conditional independent of every other feature, given the class":

$$ p(y \mid \vec{x}) = p(y) \prod_{i=1}^{m} p(x_{i} \mid y) $$

we get the final Naive Bayes model, which has a consequence of the assumption above,
doesn't capture dependencies between each input variables in $$\vec{x}$$.



#### From Naive Bayes to Hidden Markov Models

To predict a class sequence $$y=(y_{1}, \dots, y_{n})$$ for an observation sequence  $$x=(x_{1}, \dots, y_{n})$$, a simple sequence model can be formulated as a product over single Naïve Models:

$$ p(\vec{y} \mid \vec{x}) = \prod_{i=1}^{n} p(y_{i}) \cdot p(x_{i} \mid y_{i}) $$

Two aspects about this model:

* there is only one feature at each sequence position, namely the identity of the respective observation because, again, there is the assumption that each feature is generated independently (conditioned on y).

* it doesn't capture interactions between the observable variables $$x_{i}$$.

It is however reasonable to assume that there are dependencies between the observations at consecutive sequence positions $$y_{i}$$, remember the example above about the part-of-speech tags ?

The Hidden Markov Model introduces state transition probabilities, first order if it only accounts for the previous state:

$$ p(\vec{y} \mid \vec{x}) = \prod_{i=1}^{n} p(y_{i} \mid y_{i-1}) \cdot p(x_{i} \mid y_{i}) $$

which written in it's more general form:

$$ p(\vec{x}) = \sum_{y \in Y} \prod_{i=1}^{n} p(y_{i} \mid y_{i-1}) \cdot p(x_{i} \mid y_{i}) $$

where Y represents the set of all possible label sequences \vec{y}.

### Training (parameters inference): __TODO__

<!--
https://stats.stackexchange.com/questions/95144/training-a-hidden-markov-model-multiple-training-instances
-->


### Testing (decoding a sequence): __TODO__

<!--
k^m possible state sequences,
  m = size of sentence
  k = number of states, i.e., number of possible Y label tags
- Viterbi algorithm
-->


### Observations:

* There is only one feature at each sequence position, namely the identity i.e.m the value of the respective observation.

* Each state depends only on its immediate predecessor, that is, each state $$y_{i}$$ is independent of all its ancestors $$y_{1}, y_{2}, \dots, y_{i-2}$$ given its previous state $$y_{i-1}$$.

* Each observation variable $$x_{i}$$ depends only on the current state $$y_{i}$$.

<!--
With these assumptions, we can specify an HMM using three probability distributions:

first, the distribution p(y1) over initial states;
second, the transition distribution p(yt|yt−1);

and finally, the observation distribution p(xt|yt).

That is, the joint probability of a state sequence y and an observation sequence x factorizes:

p(y, x) =   p(yt|yt−1)p(xt|yt), (1.8) t=1

where, to simplify notation, we write the initial state distribution p(y1) as p(y1|y0).

Represented by two distribution probabilities.

* transition distribution $$P(y_t|y_{t−1})$$, which tells how adjacent y values are
related

* observation distribution $$P(x|y)$$, which tells how the observed
x values are related to the hidden y values.

These distributions are assumed to be stationary (i.e., the same for all times t).

https://jyyuan.wordpress.com/2014/01/28/baum-welch-algorithm-finding-parameters-for-our-hmm/

- Expected Maximization for HMM
    - E-step: Use inference (forwards-backwards algorithm)
    - M-step: Recompute parameters with weighted data

The HMM generates xi and yi as follows. Suppose there are K possible
labels 1,...,K. Augment this set of labels with a start label 0 and a terminal
label K + 1. Let yi,0 = 0. Then, generate the sequence of y values according
to P(yi,t|yi,t−1) until yi,t = K + 1. At this point, set Ti := t. Finally, for each
t = 1,...,Ti, generate xi,t according to the observation probabilities P(xi,t|yi,t).

In a sequential supervised learning problem, it is straightforward to determine
the transition and observation distributions. P(yi,t|yi,t−1) can be computed by
looking at all pairs of adjacent y labels (after prepending 0 at the start and
appending K + 1 to the end of each yi). Similarly, P(xj |y) can be computed by
looking at all pairs of xj and y.
-->

Both models presented before, the Naïve Bayes and the Hidden Markov models are
generative models, that is, they model the joint distribution $p(y, x)$.



----



# __Maximum Entropy Models__

The previous models, the Naïve Bayes and the Hidden Markov, are generative models, because they require us to model the class prior distribution: $$P(Y)$$, and class conditional distribution: $$P(X \mid Y)$$.

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

### Observations:

* The advantage for HMM is the use of feature vectors. Transition probability can be sensitive to any word in the input sequence. MEMMs support long-distance interactions over the whole observation sequence.

* We have exponential model for each state to tell us the conditional probability of the next states

* Label bias problem




----





# __Conditional Random Fields__

### Training (parameters inference): __TODO__

### Testing (decoding a sequence): __TODO__

### Observations:

* __TODO__

<!--
A first key idea in CRFs will be to define a feature vector that maps an entire
input sequence x paired with an entire state sequence s to some d-dimensional feature vector.

IDEA: maps an entire input sequence x paired with an entire state sequence s to
some d-dimensional feature vector.

The chief difference between MEMM and CRF is that MEMM is locally renormalized and suffers from the label bias problem, while CRFs are globally renormalized.
-->
