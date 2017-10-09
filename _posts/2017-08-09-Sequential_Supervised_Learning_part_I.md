---
layout: post
title: Hidden Markov Models
date: 2017-08-09 00:00:00
tags: [hidden markov models, naive bayes, sequence classifier, tutorial]
categories: [blog]
comments: true
disqus_identifier: 20170809
preview_pic: /assets/images/2017-08-09-Sequential_Supervised_Learning_part_I.png
---

This is the first post, of a series of posts, about sequential supervised learning applied to Natural Language Processing (NLP). In this post I will write about the classical algorithm for sequence learning, the Hidden Markov Model (HMM), explain how it's related with the Naive Bayes Model and it's limitations.

## __Introduction__

The classical problem in Machine Learning is to learn a classifier that can distinguish between two or more classes, i.e., that can accurately predict a class for a new object given training examples of objects already classified.

In NLP typical examples are for instance, classifying an email as spam or not spam, classifying a movie into genres, classifying a news article into topics, etc. However there is another type of prediction problems which involve structure.

A classical example in NLP is part-of-speech tagging, in this scenario, each $$x_{i}$$ describes a word and each $$y_{i}$$ the associated part-of-speech of the word $$x_{i}$$ (e.g.: _noun_, _verb_, _adjective_, etc.).

Another example, is named-entity recognition, in which, again, each $$x_{i}$$ describes a word and $$y_{i}$$ is a semantic label associated to that word (e.g.: _person_, _location_, _organization_, _event_, etc.).

In both examples the data consist of sequences of $(x, y)$ pairs. We want to model our learning problem based on that sequence:

$$p(y_1, y_2, \dots, y_m \mid x_1, x_2, \dots, x_m)$$

In NLP problems these sequences can have a sequential correlation. That is, nearby $x$ and $y$ values are likely to be related to each other. For instance, in English, it's common that after the preposition _to_ the part-of-speech tag associated to the following word is a verb.


Note that there are other machine learning problems which also involve sequences but are clearly different. For instance, in time-series, there is also a sequence, but we want to predict a value $$y$$ at point $$t+1$$, and we can use all the previous true observed $$y$$ to predict. In sequential supervised learning we must predict all $$y$$ values in the sequence.

The paper [Machine Learning for Sequential Data: A Review by Thomas G. Dietterich](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf) contains many more examples, and is a good introduction to the supervised sequential learning problem.

The Hidden Markov Model (HMM) was one the first proposed algorithms to classify sequences. It has it's roots on the Naive Bayes model, and an HMM can be seen as a sequential extension to the Naive Bayes model.

## __Naive Bayes classifier__

The Naive Bayes (NB) classifier is a ___generative model___, which builds a model of each class based on the training examples for each class. Then, in prediction, given an observation, it returns the class most likely to have generated the observation. In contrast ___discriminative models___, like logistic regression, try to learn which features from the training examples are most useful to discriminate between the different possible classes.

The Naive Bayes classifier returns the class that as the maximum posterior probability given the features:

$$ \hat{y} = argmax\ p(y \mid \vec{x})$$

where $$y$$ it's a class and $$\vec{x}$$ is a feature vector associated to an observation. The NB classifier is based on the Bayes' theorem, applying the theroem to the equation above, we get:

$$ p(y \mid \vec{x}) = \frac{p(y) \cdot p(\vec{x} \mid y)}{p(\vec{x})} $$


In classification logistic regression chooses a class by using the equation defined above to compute the probability of each class and then choose the one that yields the maximum probability.

In training, when iterating over all classes, for a given observation, and calculating the probabilities above, the probability of the observation, i.e., the denominator, is always the same, it has no influence, so we can then simplify the formula:

$$p(y \mid \vec{x}) = p(y) \cdot p(\vec{x} \mid y) $$

which, if we decompose the vector of features, is the same as:

$$ p(y \mid \vec{x}) = p(y) \cdot p(x_{1}, x_{2}, x_{3}, \dots, x_{1} \mid y) $$

this is hard to compute, because it involves estimating every possible combination of features. We can be relaxed this computation by applying the Naïves Bayes assumption, which states that:

<center>
<b>
 "each feature is conditional independent of every other feature, given the class"
</b>
 </center>

formerly, $$p(x_{i} \mid y,x_{j}) = p(x_{i} \mid y)$$ with $$i \neq j$$. The probabilities $$p(x_{i} \mid y)$$ are independent given the class $$y$$ and hence can be ‘naively’ multiplied:

$$ p(x_{1}, x_{2}, \dots, x_{1} \mid y) =  p(x_{1} \mid y) \cdot p(x_{2} \mid y), \cdots, p(x_{m} \mid y)$$

pluging this into our equation:

$$ p(y \mid \vec{x}) = p(y) \prod_{i=1}^{m} p(x_{i} \mid y) $$

we get the final Naive Bayes model, which as consequence of the assumption above, doesn't capture dependencies between each input variables in $$\vec{x}$$.


### __Trainning__

### __Classification__

---

## __From Naive Bayes to Hidden Markov Models__

The model presented before predicts a class for a set of features associated to an observation. To predict a class sequence $$y=(y_{1}, \dots, y_{n})$$ for sequence of observation $$x=(x_{1}, \dots, y_{n})$$, a simple sequence model can be formulated as a product over single Naïve Bayes models:

$$ p(\vec{y} \mid \vec{x}) = \prod_{i=1}^{n} p(y_{i}) \cdot p(x_{i} \mid y_{i}) $$

Two aspects about this model:

* there is only one feature at each sequence position, namely the identity of the respective observation due the assumption that each feature is generated independently, conditioned on the class $$y_{i}$$.

* it doesn't capture interactions between the observable variables $$x_{i}$$.

It is however reasonable to assume that there are dependencies between the observations at consecutive sequence positions $$y_{i}$$, remember the example above about the part-of-speech tags ?

This is where the First-order Hidden Markov Model appears, introducing the __Markov Assumption__:


<center>
<b>
 "the probability of a particular state is dependent only on the previous state"
</b>
 </center>

$$ p(\vec{y} \mid \vec{x}) = \prod_{i=1}^{n} p(y_{i} \mid y_{i-1}) \cdot p(x_{i} \mid y_{i}) $$

which written in it's more general form:

$$ p(\vec{x}) = \sum_{y \in Y} \prod_{i=1}^{n} p(y_{i} \mid y_{i-1}) \cdot p(x_{i} \mid y_{i}) $$

where Y represents the set of all possible label sequences $$\vec{y}$$.

----

## __Hidden Markov Model__

A Hidden Markov Model (HMM) is a sequence classifier. As other machine learning algorithms it can be trained, i.e.: given labeled sequences of observations, and then using the learned parameters to assign a sequence of labels given a sequence of observations. Let's define an HMM framework containing the following components:

* states (i.e., labels): $$T = t_{1}, t_{2}, \cdots, t_{N}$$
* observations (i.e., words) : $$W = w_{1}, w_{2}, \cdots, w_{N}$$
* two special states: $$t_{start}$$ and $$t_{end}$$ which are not associated with the observation

and probabilities relating states and observations:

* __initial probability__: an initial probability distribution over states
* __final probability__: a final probability distribution over states
* __transition probability__: a matrix $$A$$ with the probabilities from going from one label to another
* __emission probability__: a matrix $$B$ with the probabilities of an observation being generated from a state

<figure>
  <img style="width: 65%; height: 65%" src="/assets/images/2017-08-09-Sequential_Supervised_Learning_part_I.png">
</figure>

<!--
picture taken from:
http://www.cs.virginia.edu/~hw5x/Course/CS6501-Text-Mining/_site/mps/mp3.html

https://liqiangguo.wordpress.com/page/2/
-->

A First-order Hidden Markov Model has assumptions:

* __Markov Assumption__: the probability of a particular state is dependent only on the previous state.

* __Output Independence__: the probability of an output observation $$w_{i}$$ depends only on the state that produced the observation $$t_{i}$$ and not on any other states or any other observations.

Within this framework we can define 3 three major problems which can be efficiently solved by relying on dynamic programming and by making use of the independence assumptions of the HMM model.


<!--
file:///Users/dsbatista/Desktop/CRFs/tutorial%20on%20hmm%20and%20applications.pdf
-->

#### __Likelihood: computing the probability of an observation sequence__

* Given an HMM $$\gamma$$ = ($$A$$,$$B$$) and an observation sequence $$W = w_{1}, w_{2}, \dots, w_{T}$$
* How to determine the likelihood $$P( W \mid λ)$$, i.e., what is the probability of this observation sequence ?
* Forward algorithm

#### __Decoding: finding best label sequence for an observation sequence__

* Given an HMM $$\gamma$$ = ($$A$$,$$B$$) and an observation sequence $$W = w_{1}, w_{2}, \cdots, w_{T}$$
* How to find the most probable sequence of states $$T = t_{1}, t_{2}, t_{3}, \dots, t_{T}$$ ?
* Posterior Decoding or minimum risk decoding
* Viterbi

#### __Learning: learn the HMM parameters__

* Given an observation sequence $$W$$ and the set of possible states $$T$$ in the HMM
* How to learn the HMM parameters $$A$$ and $$B$$
* Forward-Backward
* Baum-Welch algorithm

<!--

https://jyyuan.wordpress.com/2014/01/28/baum-welch-algorithm-finding-parameters-for-our-hmm/
http://setosa.io/ev/markov-chains/
https://vimeo.com/154512602

-->


#### __HMM Important Observations__

* There is only one feature at each word/observation in the sequence, namely the identity i.e., the value of the respective observation.

* Each state depends only on its immediate predecessor, that is, each state $$t_{i}$$ is independent of all its ancestors $$t_{1}, t_{2}, \dots, t_{i-2}$$ given its previous state $$t_{i-1}$$.

* Each observation variable $$w_{i}$$ depends only on the current state $$t_{i}$$.


## __Links / Papers__

* [Chapter 6: "Naive Bayes and Sentiment Classification" in Speech and Language Processing. Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/6.pdf)
