---
layout: post
title: Conditional Random Fields
date: 2017-08-11 00:00:00
tags: [conditional random fields, sequence modelling, tutorial]
categories: [blog]
comments: true
disqus_identifier: 20170811
preview_pic:
---

This is the third and the last part of a series of posts about sequential supervised learning applied to NLP.

<!--

\newcommand{\argmax}[1]{\underset{#1}{\operatorname{arg}\,\operatorname{max}}\;}

https://liqiangguo.wordpress.com/page/2/
http://www.cs.columbia.edu/~smaskey/CS6998/slides/statnlp_week10.pdf
http://www.cs.columbia.edu/~smaskey/CS6998-0412/slides/week13_statnlp_web.pdf

http://videolectures.net/cikm08_elkan_llmacrf/

http://www.lsi.upc.edu/~aquattoni/AllMyPapers/crf_tutorial_talk.pdf

http://curtis.ml.cmu.edu/w/courses/index.php/Lafferty_2001_Conditional_Random_Fields

http://www.cs.cornell.edu/courses/cs6784/2010sp/lecture/10-LaffertyEtAl01.pdf

http://www.stokastik.in/understanding-conditional-random-fields/


HMMs don't allow the addition of overlapping arbitrary features of the observations.

MeMMs are normalized locally over each observation, and hence suffer from the Label Bias problem, where the transitions going out from a state compete only against each other, as opposed to all the other transitions in the model.


A first key idea in CRFs will be to define a feature vector that maps an entire
input sequence x paired with an entire state sequence s to some d-dimensional feature vector.

IDEA: maps an entire input sequence x paired with an entire state sequence s to
some d-dimensional feature vector.

A conditional random field may be viewed as an undirected graphical model, or Markov random field [3], globally conditioned on X, the random variable representing observation sequences



-->

# __Label Bias Problem__


### CRF vs. MEMM

The critical difference between CRF and MEMM is that the latter uses per-state exponential models for the condi- tional probabilities of next states given the current state, whereas CRF uses a single exponential model to deter- mine the joint probability of the entire sequence of labels, given the observation sequence. Therefore, in CRF, the weights of different features in different states compete against each other.


### __CRF Important Observations__

* The big difference between MEMM and CRF is that MEMM is locally renormalized and suffers from the label bias problem, while CRFs are globally re-normalized.

* CRFs avoid the label bias problem a weakness exhibited by Maximum Entropy Markov Models (MEMM)


## __Software Packages__

*




## __References__

[Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)



* [Conditional Random Fields: An Introduction. Hanna M. Wallach, February 24, 2004. University of Pennsylvania CIS Technical Report MS-CIS-04-21](http://dirichlet.net/pdf/wallach04conditional.pdf)
