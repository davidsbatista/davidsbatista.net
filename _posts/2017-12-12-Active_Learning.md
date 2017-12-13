---
layout: post
title: Active Learning
date: 2017-12-12 00:00:00
tags: []
categories: [blog]
comments: true
disqus_identifier: 20171212
preview_pic: /assets/images/
---


Scenarios
=========

__Membership query synthesis__

*  In this setting, the learner may request labels for any unlabeled instance in the input space, including (and typically assuming) queries that the learner generates de novo, rather than those sampled from some underlying natural distribution.

* Query synthesis is reasonable for many problems, but labeling such arbitrary instances can be awkward if the oracle is a human annotator.




__Stream-based selective sampling__

* Obtaining an unlabeled instance is free (or inexpensive), so it can first be sampled from the actual distribution, and then the learner can decide whether or not to request its label.

* Each unlabeled instance is typically drawn one at a time from the data source

* The decision whether or not to query an instance can be framed several ways. One approach is to evaluate samples using some “informativeness measure” or “query strategy” (see Section 3 for examples) and make a biased random deci- sion, such that more informative instances are more likely to be queried

* Another approach is to compute an explicit region of uncertainty (Cohn et al., 1994), i.e., the part of the instance space that is still ambiguous to the learner, and only query instances that fall within it

* In other words, if any two models of the same model class (but different parameter settings) agree on all the labeled data, but disagree on some unlabeled instance, then that instance lies within the region of uncertainty.




__Pool-based sampling__

* Assumes that there is a small set of labeled data L and a large pool of unlabeled data U available. _Queries are selectively drawn from the pool_, which is usually assumed to be closed (i.e., static or non-changing), although this is not strictly necessary.

* The main difference between stream-based and pool-based active learning is that the former scans through the data sequentially and makes query decisions individually, whereas the latter evaluates and ranks the entire collection before selecting the best query.




Query strategy Frameworks
=========================

All active learning scenarios involve evaluating the informativeness of unlabeled instances, which can either be generated de novo or sampled from a given distribution. There have been many proposed ways of formulating such query strategies in the literature.



__Uncertainty Sampling__

* In this framework, an active learner queries the instances about which it is least certain how to label.

* __least confident__: when using a probabilistic model for binary classification, uncertainty sampling simply queries the instance whose posterior probability of being positive is nearest 0.5; For problems with three or more class labels, a more general uncertainty sampling variant might query the instance whose prediction is the least confident:

* __margin sampling__: the least confident strategy only considers information about the most probable label. Thus, it effectively “throws away” information about the remaining label distribution. To correct for this, some researchers use a different multi-class uncertainty sampling variant called margin sampling, incorporating the posterior of the second most likely label. Intuitively, instances with large margins are easy, since the classifier has little doubt in differentiating between the two most likely class labels. Instances with small margins are more ambiguous, thus knowing the true label would help the model discriminate more effectively between them.

* __entropy__:


__Query-By-Committee__

__Expected Model Change__

__Expected Error Reduction__

__Variance Reduction__

__Density-Weighted Methods__





