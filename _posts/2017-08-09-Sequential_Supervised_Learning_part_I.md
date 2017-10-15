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

The Hidden Markov Model (HMM) was one the first proposed algorithms to classify sequences. There are other sequence models, but I will start by explaining the HMMM as a sequential extension to the Naive Bayes model.

## __Naive Bayes classifier__

The Naive Bayes (NB) classifier is a ___generative model___, which builds a model of each class based on the training examples for each class. Then, in prediction, given an observation, it returns the class most likely to have generated the observation. In contrast ___discriminative models___, like logistic regression, try to learn which features from the training examples are most useful to discriminate between the different possible classes.

<figure>
  <img style="width: 55%; height: 55%" border="5" src="/assets/images/2017-08-09-1024px-Bayes_Theorem_MMB_01.jpg">
</figure>

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

Training in Naive Bayes is mainly done by counting features and classes. Note that the procedure described below needs to be done for every class $$y_{i}$$.

To calculate the prior, we simple count how many samples in the training data fall into each class $y_{i}$ divided by the total number of samples:

$$p(y_{i}) = \frac{ N_{y_{i}} } {N} $$

To calculate the likelihood estimate, we count the number of times feature $$w_{i}$$ appears among all features in all samples of class $$y_{i}$$:

$$ p(x_{i} \mid y_{i}) = \frac{count(w_{i},y_{i})} {  \sum_{x_{i} \in X} count(x_{i},y_{i})}$$

This will result in a big table of occurrences of features for all classes in the training data.


### __Classification__

When given a new sample to classify, and assuming that it contains features $$x_{1}, w_{3}, w_{5}$$, we need to compute, for each class $$y_{i}$$:

$$ p(y_{i} \mid x_{1}, w_{3}, w_{5})$$

This is decomposed into:

$$ p(y_{i} \mid x_{1}, w_{3}, w_{5}) = p(y_{i}) \cdot p(y_{i} \mid x_{1}) \cdot p(y_{i} \mid x_{3}) \cdot p(y_{i} \mid x_{5})$$

Again, this is calculated for each class $$y_{i}$$, and the assign the one that has the highest score.




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
* __emission probability__: a matrix $$B$$ with the probabilities of an observation being generated from a state

<figure>
  <img style="width: 65%; height: 65%" src="/assets/images/2017-08-09-Sequential_Supervised_Learning_part_I.png">
</figure>

<!--
picture taken from:
http://www.cs.virginia.edu/~hw5x/Course/CS6501-Text-Mining/_site/mps/mp3.html
https://liqiangguo.wordpress.com/page/2/
-->

A First-order Hidden Markov Model has the following assumptions:

* __Markov Assumption__: the probability of a particular state is dependent only on the previous state. Formally: $$P(t_{i} \mid t_{1}, \cdots, t_{i-1}) = P(t_{i} \mid t_{i-1})$$

* __Output Independence__: the probability of an output observation $$w_{i}$$ depends only on the state that produced the observation $$t_{i}$$ and not on any other states or any other observations. Formally: $$P(w_{i} \mid t_{1} ...q_{i},...,q_{T} ,o_{1},...,o_{i},...,o_{T} ) = P(o_{i} \mid q_{i})$$

Notice how the output assumption is closely related with the Naive Bayes classifier presented before.

We can now define two problems which can be solved by an HMM, the first is learning the parameters associated to a given observation sequence, that is __training__. For instance given words of a sentence and the associated part-of-speech tags, one can learn the latent structure.

The other one is applying a trained HMM to an observation sequence, for instance, having a sentence, __predicting__ each word's part-of-speech tag, using the latent structure from the training data learned by the HMM.


#### __Learning: finding the maximum likelihood parameters__

<!--
* Given an observation sequence $$W$$ and the set of possible states $$T$$ in the HMM
* How to learn the HMM parameters $$A$$ and $$B$$
* Forward-Backward
* Baum-Welch algorithm
-->

<!--
https://jyyuan.wordpress.com/2014/01/28/baum-welch-algorithm-finding-parameters-for-our-hmm/
http://setosa.io/ev/markov-chains/
https://vimeo.com/154512602
-->
<!--
file:///Users/dsbatista/Desktop/CRFs/tutorial%20on%20hmm%20and%20applications.pdf
-->


#### __Decoding: finding the state sequence for an observation sequence__

Given a trained HMM (i.e., that is the transition matrixes $$A$$ and $$B$$) and a new observation sequence $$W = w_{1}, w_{2}, \cdots, w_{N}$$ we want to find the sequence of states that best explains it.

* Posterior Decoding or Minimum Risk Decoding: minimises the probability of error on each hidden state variable $$T_{i}$$

* Viterbi: finds the best state assignment to the sequence $$T_{1} \cdots T_{N}$$ as a whole.



* Viterbi

<!--
Selecting the state with the highest posterior for each position independently,

$$\hat{y} = argmax p(Y_{i} = y_{i} \mid X = x)$$

* sequence posterior:
* state posterior : prob. being in a given state in a certain position given the observed sequence
* transition posterior:


to compute the posteriors we need to compute the likelihood


##### __Likelihood: computing the probability of an observation sequence__

* Given an HMM $$\gamma$$ = ($$A$$,$$B$$) and an observation sequence $$W = w_{1}, w_{2}, \dots, w_{T}$$
* How to determine the likelihood $$P( W \mid λ)$$, i.e., what is the probability of this observation sequence ?
* Forward algorithm


for a particular hidden state Q and an observation O, the likelihood of the observation sequence is:

$$P(W|T) = \prod_{i=1}^{N} P(w_{i} \mid t_{i})$$

Applying the following equation for an already known state sequence:

$$P(3 1 3|hot hot cold) = P(3|hot)×P(1|hot)×P(3|cold)$$

But we don't know what the actual hidden state is. We’ll need to compute the probability of an observation sequence instead by summing over all possible state sequences, weighted by their probability.

First, let’s compute the joint probability of being in a particular state sequence T and generating a particular sequence O of observation events. In general, this is


$$P(W|T) = P(W|T) \cdot  P(T) = \prod{} P(o_{i} \mid t_{i}) \times \prod{} P(q_{i} \mid q_{i-1})$$

The computation of the joint probability of our ice-cream observation 3 1 3 and
one possible hidden state sequence hot hot cold is:

P(3 1 3,hot hot cold) = P(hot|start)×P(hot|hot)×P(cold|hot)×P(3|hot)×P(1|hot)×P(3|cold)

__TODO__: graphical representation

Now that we know how to compute the joint probability of the observations
with a particular hidden state sequence, we can compute the total probability of the observations just by summing over all possible hidden state sequences:

$$P(W) = \sum_{T} P(W \mid T) = \sum_{T} P(W|T) \cdot  P(T) $$


__TODO__: relation with Naive Bayes

For our particular case, we would sum over the eight 3-event sequences cold cold cold, cold cold hot, that is, P(3 1 3) = P(3 1 3, cold cold cold) +P(3 1 3, cold cold hot) +P(3 1 3,hot hot cold) +...

For an HMM with N hidden states and an observation sequence of T observations,
there are $$N^{T}$$ possible hidden sequences.

For a task of part-of-speech tagging, we have 12 tags (N) and having a sentence with 10 words, that would be 12^10 separate observation likelihoods to compute.

__TODO__: Forward algorithm

The forward algorithm computes the observation probability by summing over the probabilities of all possible hidden state paths that could generate the observation sequence, but it does so efficiently by implicitly folding each of these paths into a single forward trellis.

The FB algorithm relies on the independence of previous states assumption, which is illustrated in the trellis view by having arrows only between consecutive states.

The FB algorithm defines two auxiliary probabilities:
  * the forward probability
  * backward probability.
-->





















### __HMM Important Observations__

* The main idea of this post was to see the connection between the Naive Bayes classifier and the HMM as a sequence classifier

* If we make the hidden state of HMM fixed, we will have a Naive Bayes model.

__TODO__: picture (Suppose we use graphical model's notation. Naive Bayes model can be described as)

* There is only one feature at each word/observation in the sequence, namely the identity i.e., the value of the respective observation.

* Each state depends only on its immediate predecessor, that is, each state $$t_{i}$$ is independent of all its ancestors $$t_{1}, t_{2}, \dots, t_{i-2}$$ given its previous state $$t_{i-1}$$.

* Each observation variable $$w_{i}$$ depends only on the current state $$t_{i}$$.


## __Papers__

* [Machine Learning for Sequential Data: A Review by Thomas G. Dietterich](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf)

* [Chapter 6: "Naive Bayes and Sentiment Classification" in Speech and Language Processing. Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/6.pdf)

* [A tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)
