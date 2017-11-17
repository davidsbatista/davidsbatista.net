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

http://www.stokastik.in/understanding-conditional-random-fields/

-->

## __Conditional Random Fields__

CRFs were proposed roughly only year after the Maximum Entropy Markov Models, basically by the same authors. Reading through the original [paper that introduced Conditional Random Fields](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers), one finds at the beginning this sentence:

_"The critical difference between CRF and MEMM is that the latter uses per-state exponential models for the conditional probabilities of next states given the current state, whereas CRF uses a single exponential model to deter- mine the joint probability of the entire sequence of labels, given the observation sequence. Therefore, in CRF, the weights of different features in different states compete against each other."_

This means that in the previous model, Maximum Entropy Markov Models (MEMM) there is a model for each state to compute the probability of the next state, given the current state and the observation. On the other hand CRF computes all state transitions in a single model.

A first key idea in CRFs will be to define a feature vector that maps an entire input sequence $$X$$ paired with an entire state sequence $$S$$ to some $$d$$-dimensional feature vector.

<figure>
  <img style="width: 45%; height: 55%" src="/assets/images/2017-11-13-Conditional_Random_Fields.png">
  <figcaption>A CRF as an undirected graphical model globally conditioned on X.
  <br> (taken from Lafferty et al. 2001)</figcaption>
</figure>

The main motivation for this is the so called Label Bias Problem, which generates a bias towards states with few successor states.

### __Label Bias Problem__


### __Training and Decoding__


### __CRF Important Observations__

* HMMs don't allow the addition of overlapping arbitrary features of the observations.

* MEMMs are normalized locally over each observation, and hence suffer from the Label Bias problem, where the transitions going out from a state compete only against each other, as opposed to all the other transitions in the model.

* The big difference between MEMM and CRF is that MEMM is locally renormalized and suffers from the label bias problem, while CRFs are globally re-normalized.

* CRFs avoid the label bias problem a weakness exhibited by Maximum Entropy Markov Models (MEMM)




## __Software Packages__

* [python-crfsuite](https://github.com/scrapinghub/python-crfsuite): is a python binding for [CRFsuite](https://github.com/chokkan/crfsuite) which is a fast implementation of Conditional Random Fields written in C++.



## __References__

* ["Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)

* ["Log-Linear Models, MEMMs, and CRFs". Michael Collins](http://www.cs.columbia.edu/~mcollins/crf.pdf)


* ["Conditional Random Fields: An Introduction". Hanna M. Wallach, February 24, 2004. University of Pennsylvania CIS Technical Report MS-CIS-04-21](http://dirichlet.net/pdf/wallach04conditional.pdf)