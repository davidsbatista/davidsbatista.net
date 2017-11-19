---
layout: post
title: Conditional Random Fields for Sequence Prediction
date: 2017-11-13 00:00:00
tags: [conditional random fields, sequence prediction, tutorial]
categories: [blog]
comments: true
disqus_identifier: 20171113
preview_pic: /assets/images/2017-11-13-Conditional_Random_Fields.png
---

This is the third and (maybe) the last part of a series of posts about sequential supervised learning applied to NLP. In this post I will talk about Conditional Random Fields (CRF), explain what was the main motivation behind the proposal of this model, and make a final comparison between Hidden Marko Models (HMM), Maximum Entropy Markov Models (MEMM) and CRF for sequence prediction.

<!--
\newcommand{\argmax}[1]{\underset{#1}{\operatorname{arg}\,\operatorname{max}}\;}

https://liqiangguo.wordpress.com/page/2/

http://www.cs.columbia.edu/~smaskey/CS6998/slides/statnlp_week10.pdf
http://www.cs.columbia.edu/~smaskey/CS6998-0412/slides/week13_statnlp_web.pdf

http://videolectures.net/cikm08_elkan_llmacrf/

http://cseweb.ucsd.edu/~elkan/250B/CRFs.pdf

http://www.lsi.upc.edu/~aquattoni/AllMyPapers/crf_tutorial_talk.pdf

http://curtis.ml.cmu.edu/w/courses/index.php/Lafferty_2001_Conditional_Random_Fields

http://www.cs.cornell.edu/courses/cs6784/2010sp/lecture/10-LaffertyEtAl01.pdf

-->

## __Introduction__

CRFs were proposed roughly only year after the Maximum Entropy Markov Models, basically by the same authors. Reading through the original [paper that introduced Conditional Random Fields](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers), one finds at the beginning this sentence:

_"The critical difference between CRF and MEMM is that the latter uses per-state exponential models for the conditional probabilities of next states given the current state, whereas CRF uses a single exponential model to determine the joint probability of the entire sequence of labels, given the observation sequence. Therefore, in CRF, the weights of different features in different states compete against each other."_

This means that in the MEMMs there is a model to compute the probability of the next state, given the current state and the observation. On the other hand CRF computes all state transitions globally, in a single model.

The main motivation for this proposal is the so called Label Bias Problem occurring in MEMM, which generates a bias towards states with few successor states.

### __Label Bias Problem in MEMMs__

Recalling how the transition probabilities are computed in a MEMM model, from the previous post, we learned that the probability of the next state is only dependent on the observation (i.e., the sequence of words) and the previous state, that is, we have exponential model for each state to tell us the conditional probability of the next states:

<figure>
  <img style="width: 25%; height: 25%" src="/assets/images/2017-11-13-HMM.png">
  <figcaption>MEMM transition probability computation. <br> (taken from A. McCallum et al. 2000)</figcaption>
</figure>

This causes the so called __Label Bias Problem__, and Lafferty et al. 2001 demonstrate this through experiments and report it. I will not demonstrate it, but just give the basic intuition taken also from the paper:

<figure>
  <img style="width: 65%; height: 65%" src="/assets/images/2017-11-13-Label_Bias_Problem.png">
  <figcaption>Label Bias Problem. <br> (taken from Lafferty et al. 2001)</figcaption>
</figure>

Given the observation sequence: ___r_ _i_ _b___

_"In the first time step, r matches both transitions from the start state, so the probability mass gets distributed roughly equally among those two transitions. Next we observe i. Both states 1 and 4 have only one outgoing transition. State 1 has seen this observation often in training, state 4 has almost never seen this observation; but like state 1, state 4 has no choice but to pass all its mass to its single outgoing transition, since it is not generating the observation, only conditioning on it. Thus, states with a single outgoing transition effectively ignore their observations."_

_[...]_

_"the top path and the bottom path will be about equally likely, independently of the observation sequence. If one of the two words is slightly more common in the training set, the transitions out of the start state will slightly prefer its corresponding transition, and that word’s state sequence will always win."_

* Transitions from a given state are competing against each other only.

* Per state normalization, i.e. sum of transition probability for any state has to sum to 1.

* MEMM are normalized locally over each observation where the transitions going out from a state compete only against each other, as opposed to all the other transitions in the model.

* States with a single outgoing transition effectively ignore their observations.

* Causes bias: states with fewer arcs are preferred.

The idea of CRF is to drop this local per state normalization, and replace it by a global per sequence normalisation.

So, how dow we formalise this global normalisation? I will try to explain it in the sections that follow.

<br>

---

### __Undirected Graphical Models__

A Conditional Random Field can be seen as an undirected graphical model, or Markov Random Field, globally conditioned on $$X$$, the random variable representing observation sequence.

Lafferty et al. 2001 define a Conditional Random Field as:

* $$X$$ is a random variable over data sequences to be labeled, and $$Y$$ is a random variable over corresponding label sequences.

* The random variables $$X$$ and $$Y$$ are jointly distributed, but in a discriminative framework we construct a conditional model $$p(Y \mid X)$$ from paired observation and label sequences:

Let $$G = (V , E)$$ be a graph such that $$Y = (Y_{v})\  \ v \in V$$, so that $$Y$$ is indexed by the vertices of $$G$$.

$$(X, Y)$$ is a conditional random field when each of the random variables $$Y_{v}$$, conditioned on $$X$$, obey the Markov property with respect to the graph:

$$P(Y_{v} \mid X, Y_{w}, w \neq v) = P(Y_{v} \mid X, Y_{w}, w \sim v)$$

where $$w \sim v$$ means that $$w$$ and $$v$$ are neighbours in G. Thus, a CRF is a random field globally conditioned on the observation $$X$$. This goes already in the direction of what the MEMM doesn't give us, states globally conditioned on the observation.

In theory this graph may have an arbitrary structure as long as it represents the label sequences being modelled, however the simplest and most common graph structured is that in which the nodes corresponding to elements of $$Y$$ form a simple first-order chain, as illustrated in the figure below:

<figure>
  <img style="width: 45%; height: 55%" src="/assets/images/2017-11-13-Conditional_Random_Fields.png">
  <figcaption>Chain-strucutred CRFs globally conditioned on X.
  <br> (taken from Lafferty et al. 2001)</figcaption>
</figure>

Under certain condition, such as the one show in the figure above, the probability of a particular label sequence $$Y$$ given observation sequence $$X$$ can be defined as the normalized product of potential functions.

_If the graph $G = (V,E)$ of $Y$ is a tree (of which a chain is the simplest example), its cliques are the edges and vertices. Therefore, by the fundamental theorem of random fields (Hammersley & Clifford, 1971), the joint distribution over the label sequence Y given X has the form:_

$$ \exp (\sum_{i} \lambda_{j} t_{j} (y_{i-1}, y_{i}, x, i ) +
         \sum_{k} \mu_{k} s_{k} (y_{i}, x, i))
$$

where $$t_{j} (y_{i-1}, y_{i}, x, i)$$ is a transition feature function of the entire observation sequence and the labels at positions $$i$$ and $$i-1$$, $$s_{k} (y_{i}, x, i)$$ is a state feature function of the label at position $$i$$, and $$\lambda_{j}$$ and $$\mu_{k}$$ weight parameters to be learned from the training data.


<br>

<br>

<br>

### __Parameter Estimation__

<!--
http://www.robots.ox.ac.uk/~vgg/sorg/varun_sorg_slides.pdf

http://www.stokastik.in/understanding-conditional-random-fields/

http://videolectures.net/cikm08_elkan_llmacrf/?q=conditional%20random%20fields

-->

<br>
<br>
<br>






### __HMM vs. MEMM vs. CRF__


<figure>
  <img style="width: 100%; height: 100%" src="/assets/images/2017-11-13-HMM-MEMM-CRF.png">
  <figcaption> Graph representation of HMM, MEMM and CRF. <br> (taken from Hanna Wallach 2004)</figcaption>
</figure>


### __CRF Important Observations__

* HMMs don't allow the addition of overlapping arbitrary features of the observations.

* MEMMs are normalized locally over each observation, and hence suffer from the Label Bias problem, where the transitions going out from a state compete only against each other, as opposed to all the other transitions in the model.

* The big difference between MEMM and CRF is that MEMM is locally renormalized and suffers from the label bias problem, while CRFs are globally re-normalized.

* CRFs avoid the label bias problem a weakness exhibited by Maximum Entropy Markov Models (MEMM)




## __Software Packages__

* [python-crfsuite](https://github.com/scrapinghub/python-crfsuite): is a python binding for [CRFsuite](https://github.com/chokkan/crfsuite) which is a fast implementation of Conditional Random Fields written in C++.



## __References__

* ["Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)

* ["Conditional Random Fields: An Introduction". Hanna M. Wallach, February 24, 2004. University of Pennsylvania CIS Technical Report MS-CIS-04-21](http://dirichlet.net/pdf/wallach04conditional.pdf)

* ["An Introduction to Conditional Random Fields" Sutton, Charles; McCallum, Andrew (2010)](https://arxiv.org/pdf/1011.4088v1.pdf)

* ["Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data". presentation by Joe Drish May 9, 2002](https://pdfs.semanticscholar.org/96fc/c6d81896b48320298f7b758aa1fb85ca8cb8.pdf)

* ["Log-Linear Models, MEMMs, and CRFs". Michael Collins](http://www.cs.columbia.edu/~mcollins/crf.pdf)