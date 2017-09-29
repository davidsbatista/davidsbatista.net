---
layout: post
title: Sequence Learning - Hidden Markov Models
date: 2017-08-09 00:00:00
tags: [hidden markov models, sequence modeling, tutorial]
categories: [blog]
comments: true
disqus_identifier: 20170809
preview_pic:
---

# __Introduction__

The classical problem in Machine Learning is to learn a classifier that can distinguish between two or more classes, i.e., that can accurately predict a class for a new object given training examples of objects already classified.

In NLP typical examples are for instance, classifying an email as spam or not spam, classifying a movie into genres, classifying a news article into topics, etc. However there is another type of prediction problems which involve structure.

A classical example in NLP is part-of-speech tagging, in this scenario, each $$x_{i}$$ describes a word and each $$y_{i}$$ the associated part-of-speech of the word $$x_{i}$$ (e.g.: _noun_, _verb_, _adjective_, etc.).

Another example, is named-entity recognition, in which, again, each $$x_{i}$$ describes a word and $$y_{i}$$ is a semantic label associated to that word (e.g.: _person_, _location_, _organization_, _event_, etc.).

In both examples the data consist of sequences of $(x, y)$ pairs. We want to model our learning problem based on that sequence:

$$p(y_1, y_2, \dots, y_m \mid x_1, x_2, \dots, x_m)$$

In NLP problems these sequences can have a sequential correlation. That is, nearby $x$ and $y$ values are likely to be related to each other. For instance, in English, it's common that after the preposition _to_ the part-of-speech tag associated to the following word is a verb.


Note that there are other machine learning problems which also involve sequences but are clearly different. For instance, in time-series, there is also a sequence, but we want to predict a value $$y$$ at point $$t+1$$, and we can use all the previous true observed $$y$$ to predict. In sequential supervised learning we must predict all $$y$$ values in the sequence.

The paper [Machine Learning for Sequential Data: A Review by Thomas G. Dietterich](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf) contains many more examples, and is a good introduction to the supervised sequential learning problem.

The Hidden Markov Model (HMM) was one the first algorithms to classify sequences. It has it's roots on the Naive Bayes model, and an HMM can be seen as a sequential extension to the Naive Bayes model.

## Naive Bayes classifier

It's based on the Bayes' theorem, where $$y$$ it's a class and $$\vec{x}$$ is a feature vector associated to an observation:

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

#### __Naive Bayes Important Observations__

* __TODO__


## From Naive Bayes to Hidden Markov Models

To predict a class sequence $$y=(y_{1}, \dots, y_{n})$$ for an observation sequence  $$x=(x_{1}, \dots, y_{n})$$, a simple sequence model can be formulated as a product over single Naïve Bayes Models:

$$ p(\vec{y} \mid \vec{x}) = \prod_{i=1}^{n} p(y_{i}) \cdot p(x_{i} \mid y_{i}) $$

Two aspects about this model:

* there is only one feature at each sequence position, namely the identity of the respective observation because, again, there is the assumption that each feature is generated independently (conditioned on y).

* it doesn't capture interactions between the observable variables $$x_{i}$$.

It is however reasonable to assume that there are dependencies between the observations at consecutive sequence positions $$y_{i}$$, remember the example above about the part-of-speech tags ?

The Hidden Markov Model introduces state transition probabilities, first order if it only accounts for the previous state:

$$ p(\vec{y} \mid \vec{x}) = \prod_{i=1}^{n} p(y_{i} \mid y_{i-1}) \cdot p(x_{i} \mid y_{i}) $$

which written in it's more general form:

$$ p(\vec{x}) = \sum_{y \in Y} \prod_{i=1}^{n} p(y_{i} \mid y_{i-1}) \cdot p(x_{i} \mid y_{i}) $$

where Y represents the set of all possible label sequences $$\vec{y}$$.

----

## __Hidden Markov Models__

Set of states/labels: $$Q = q_{1}, q_{2}, \cdots, q_{N}$$

Set of observations/words: $$O = o_{1}, o_{2}, o_{N}$$

* An initial probability
* A final probability
* A transition probability, matrix A, i.e., the probability from going from one label to another
* An emission probability, matrix B, i.e., probability of an observation being generated from a state $$q_{i}$$.

Within this framework we can define 3 three major problems which can be efficiently solved by relying on dynamic programming and by making use of the independence assumptions of the HMM model.

#### __1 - Computing the likelihood of an observation sequence__

* Given an HMM λ = (A,B) and an observation sequence $$O = o1,o2,...,oT$$
* How to determine the likelihood $$P(O\|λ)$$
* Forward algorithm

#### __2 - Finding best label sequence for an observation sequence__

* Given an HMM λ = (A,B) and sequence of observations O = o1,o2,...,oT
* How to find the most probable sequence of states Q = q1q2q3 ...qT ?
* Posterior Decoding or minimum risk decoding
* Viterbi

#### __3 - Learning the HMM parameters__

* Given an observation sequence O and the set of possible states in the HMM
* How to learn the HMM parameters A and B
* Forward-Backward
* Baum-Welch algorithm

<!--

https://jyyuan.wordpress.com/2014/01/28/baum-welch-algorithm-finding-parameters-for-our-hmm/
http://setosa.io/ev/markov-chains/
https://vimeo.com/154512602

-->


#### __HMM Important Observations__

* There is only one feature at each word/sequence position, namely the identity i.e., the value of the respective observation.

* Each state depends only on its immediate predecessor, that is, each state $$y_{i}$$ is independent of all its ancestors $$y_{1}, y_{2}, \dots, y_{i-2}$$ given its previous state $$y_{i-1}$$.

* Each observation variable $$x_{i}$$ depends only on the current state $$y_{i}$$.


Both models presented before, the Naïve Bayes and the Hidden Markov models are generative models, that is, they model the joint distribution $p(y, x)$.
