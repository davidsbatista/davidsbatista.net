---
layout: post
title: Performance Metrics for Natural Language Processing
date: 2018-08-19 00:0:00
categories: blog
tags: evaluation_metrics
#comments: true
#disqus_identifier: 20180819
#preview_pic: /assets/images/2018-05-09-NER_metrics.jpeg
description: This blog post describes some evaluation metrics used in NLP, it points out where we should use each one of them and the advantages and disadvantages of each.
---

I wrote this blog post with the intention to review and compare some evaluation metrics typically used in NLP tasks, the goal was to review them in detail.


## __Accuracy__

Accuracy simply measures the number of correct predicted samples over the total number of samples. For instance, if the classifier is 90% correct, it means that out of 100 instance it correctly predicts the class for 90 of them.

This can be misleading if the number of samples per class in your problem is unbalanced. Having a dataset with two classes only, where the first class is 90% of the data, and the second completes the remaining 10%. If the classifier predicts every sample as belonging to the first class, the accuracy reported will be of 90% but this classifier is in practice useless.

With imbalanced classes, it’s easy to get a high accuracy without actually making useful predictions. So, __accuracy__ as an evaluation metrics makes sense only if the class labels are uniformly distributed.

Looking at __precision__ and __recall__ helps to analyze how well the classifier is predicting each class.


## __Precision and Recall__

Precision and Recall are two metrics computed for each class. They can be easily explained through an example, imagine that we want to evaluate how well does a robot selects good apples from rotten apples There are $m$ good apples and $n$ rotten apples in a basket. A robot looks into the basket and picks out all the good apples, leaving the rotten apples behind, but is not perfect and could sometimes mistake a rotten apple for a good apple orange.

When the robot finishes, regarding the good apples, precision and recall means:

- __Precision__: number of good apples picked out of all the apples picked out;
- __Recall__:    number of good apples picked out of all the apples in the basket

__Precision__ is about exactness, classifying only one instance correctly yields 100% precision, but a very low recall, it tells us how well the system identifies samples from a given class. __Recall__ is about completeness, classifying all instances as positive yields 100% recall, but a very low precision, it tells how well the system does and identify all the samples from a given class.

Typically these metrics are combined together in a metric called $F_{1}$ (i.e., harmonic mean of precision and recall), which eases comparison of different systems, and problems with many classes.

## __Sensitivity and Specificity__

Sensitivity= true positives/(true positive + false negative)

Specificity=true negatives/(true negative + false positives)


## ROC Curves and Precision-Recall Curves

ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.

Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.

ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets.


A common way to compare models that predict probabilities for two-class problems us to use a ROC curve.


What Are ROC Curves?
A useful tool when predicting the probability of a binary outcome is the Receiver Operating Characteristic curve, or ROC curve.

It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) for a number of different candidate threshold values between 0.0 and 1.0. Put another way, it plots the false alarm rate versus the hit rate.


<!--
Two diagnostic tools that help in the interpretation of probabilistic forecast for binary (two-class) classification predictive modeling problems are ROC Curves and Precision-Recall curves.

https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks

https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

https://www.wikiwand.com/en/Precision_and_recall
-->

<!--
What is the difference between a ROC curve and a precision-recall curve? When should I use each?
https://www.quora.com/What-is-the-difference-between-a-ROC-curve-and-a-precision-recall-curve-When-should-I-use-each

ROC ?
AUC ?
metrics for binary problems?
metrics for multi-class problems?
metrics for multi-label problems?
metrics for multi-class and multi-label problems?

https://tryolabs.com/blog/2013/03/25/why-accuracy-alone-bad-measure-classification-tasks-and-what-we-can-do-about-it/

-->

<!--
http://gabrielelanaro.github.io/blog/2016/02/03/multiclass-evaluation-measures.html
-->