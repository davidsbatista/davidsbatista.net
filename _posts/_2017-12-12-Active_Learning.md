---
layout: post
title: Active Learning performing better with less training
date: 2017-12-12 00:00:00
tags: active-learning
categories: blog
comments: true
disqus_identifier: 20171212
preview_pic: /assets/images/2017-12-12-active-learning.png
---

Active Learning is a technique to aid the training of classifier when there is not enough labelled data available, it defines a process of selectively show unlabelled examples to an oracle that after being annotated boost the performance
of a classifier.

<figure>
  <img style="width: 45%; height: 45%" border="5" src="/assets/images/2017-12-12-active-learning.png">
  <figcaption>Active Learning <br> (taken from ...)</figcaption>
</figure>

Scenarios
=========

__Membership query synthesis__

*  In this setting, the learner may request labels for any unlabelled instance in the input space, including (and typically assuming) queries that the learner generates de novo, rather than those sampled from some underlying natural distribution.

* Query synthesis is reasonable for many problems, but labelling such arbitrary instances can be awkward if the oracle is a human annotator.




__Stream-based selective sampling__

* Obtaining an unlabeled instance is free (or inexpensive), so it can first be sampled from the actual distribution, and then the learner can decide whether or not to request its label.

* Each unlabeled instance is typically drawn one at a time from the data source

* The decision whether or not to query an instance can be framed several ways. One approach is to evaluate samples using some “informativeness measure” or “query strategy” (see Section 3 for examples) and make a biased random deci- sion, such that more informative instances are more likely to be queried

* Another approach is to compute an explicit region of uncertainty (Cohn et al., 1994), i.e., the part of the instance space that is still ambiguous to the learner, and only query instances that fall within it

* In other words, if any two models of the same model class (but different parameter settings) agree on all the labeled data, but disagree on some unlabeled instance, then that instance lies within the region of uncertainty.




__Pool-based sampling__

* Assumes that there is a small set of labeled data L and a large pool of unlabelled data U available. _Queries are selectively drawn from the pool_, which is usually assumed to be closed (i.e., static or non-changing), although this is not strictly necessary.

* The main difference between stream-based and pool-based active learning is that the former scans through the data sequentially and makes query decisions individually, whereas the latter evaluates and ranks the entire collection before selecting the best query.




Query strategy Frameworks
=========================

All active learning scenarios involve evaluating the informativeness of unlabelled instances, which can either be generated de novo or sampled from a given distribution. There have been many proposed ways of formulating such query strategies in the literature.



__Uncertainty Sampling__

* In this framework, an active learner queries the instances about which it is least certain how to label.

* __least confident__: when using a probabilistic model for binary classification, uncertainty sampling simply queries the instance whose posterior probability of being positive is nearest 0.5; For problems with three or more class labels, a more general uncertainty sampling variant might query the instance whose prediction is the least confident:

* __margin sampling__: the least confident strategy only considers information about the most probable label. Thus, it effectively “throws away” information about the remaining label distribution. To correct for this, some researchers use a different multi-class uncertainty sampling variant called margin sampling, incorporating the posterior of the second most likely label. Intuitively, instances with large margins are easy, since the classifier has little doubt in differentiating between the two most likely class labels. Instances with small margins are more ambiguous, thus knowing the true label would help the model discriminate more effectively between them.

* __entropy__:


__Query-By-Committee__

* The QBC approach involves maintaining a committee C = {(1), . . . , (C)} of models which are all trained on the current labeled set L, but represent competing hypotheses. Each committee member is then allowed to vote on the labelings of query candidates. The most informative query is considered to be the instance about which they most disagree.

* If we view machine learning as a search for the “best” model within the version space, then our goal in active learning is to constrain the size of this space as much as possible (so that the search can be more precise) with as few labeled instances as possible.

* For measuring the level of disagreement, two main approaches have been pro- posed. The first is vote entropy, Another disagreement measure that has been proposed is average Kullback-Leibler (KL) divergence.

__Expected Model Change__

* selecting the instance that would impart the greatest change to the current model if we knew its label. An example query strategy in this framework is the “expected gradient length” (EGL) approach for discriminative probabilistic model classes.

* In theory, the EGL strategy can be applied to any learning problem where gradient-based training is used. Since discriminative probabilistic models are usually trained using gradient-based optimization, the “change” imparted to the model can be measured by the length of the training gradient (i.e., the vector used to re-estimate parameter values). In other words, the learner should query the instance x which, if labeled and added to L, would result in the new training gradient of the largest magnitude.

* The intuition behind this framework is that it prefers instances that are likely to most influence the model (i.e., have greatest impact on its parameters), regard- less of the resulting query label. This approach has been shown to work well in empirical studies, but can be computationally expensive if both the feature space and set of labelings are very large.


__Expected Error Reduction__

* Another decision-theoretic approach aims to measure not how much the model is likely to change, but how much its generalization error is likely to be reduced. The idea it to estimate the expected future error of a model trained using L [ hx, yi on the remaining unlabeled instances in U (which is assumed to be representative of the test distribution, and used as a sort of validation set), and query the instance with minimal expected future error (sometimes called risk).

* as with EGL in the previous section, we do not know the true label for each query instance, so we approximate using expectation over all possible labels under the current model

* Another interpretation of this strategy is maximizing the expected information gain of the query x, or (equivalently) the mutual information of the output variables over x and U


__Variance Reduction__

* Minimizing the expectation of a loss function directly is expensive, and in general this cannot be done in closed form. However, we can still reduce generaliza- tion error indirectly by minimizing output variance, which sometimes does have a closed-form solution


__Density-Weighted Methods__

* The information density framework described by Settles and Craven (2008), and further analyzed in Chapter 4 of Settles (2008), is a general density-weighting technique. The main idea is that informative instances should not only be those which are uncertain, but also those which are “representative” of the underlying distribution (i.e., inhabit dense regions of the input space).



## References

* http://burrsettles.com/pub/settles.activelearning.pdf

* http://soda.swedish-ict.se/3600/1/SICS-T--2009-06--SE.pdf

