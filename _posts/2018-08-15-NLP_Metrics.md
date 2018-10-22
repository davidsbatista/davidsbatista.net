---
layout: post
title: Evaluation Metrics for NLP (and other domains)
date: 2018-08-19 00:0:00
categories: blog
tags: evaluation_metrics imbalanced_data
#comments: true
#disqus_identifier: 20180819
#preview_pic: /assets/images/2018-05-09-NER_metrics.jpeg
description: This blog post describes some evaluation metrics used in NLP, it points out where we should use each one of them and the advantages and disadvantages of each.
---

I wrote this blog post with the intention to review and compare some evaluation metrics typically used in NLP tasks, the goal was to review them in detail.

# __Introduction__

When making a prediction for a binary or two-class classification problem, there are two types of errors that we could make.

- __False Positive (FP)__: predict an event when there was no event.
- __False Negative (FN)__: predict no event when in fact there was an event.
- __True Positive (TP)__: predict an event when there was an event.
- __True Negative (TN)__: predict no event when in fact there was no event.

---

## __Accuracy__

Accuracy simply measures the number of correct predicted samples over the total number of samples. For instance, if the classifier is 90% correct, it means that out of 100 instance it correctly predicts the class for 90 of them.

$$ accuracy = \frac{nr. correct \; predictions}{nr. total \; predictions} = \frac{TP+TN}{TP+TN+FP+FN}$$

This can be misleading if the number of samples per class in your problem is unbalanced. Having a dataset with two classes only, where the first class is 90% of the data, and the second completes the remaining 10%. If the classifier predicts every sample as belonging to the first class, the accuracy reported will be of 90% but this classifier is in practice useless.

With imbalanced classes, it’s easy to get a high accuracy without actually making useful predictions. So, __accuracy__ as an evaluation metrics makes sense only if the class labels are uniformly distributed.

---

## __Precision and Recall__

Precision and Recall are two metrics computed for each class. They can be easily explained through an example, imagine that we want to evaluate how well does a robot selects good apples from rotten apples There are $m$ good apples and $n$ rotten apples in a basket. A robot looks into the basket and picks out all the good apples, leaving the rotten apples behind, but is not perfect and could sometimes mistake a rotten apple for a good apple orange.

When the robot finishes, regarding the good apples, precision and recall means:

- __Precision__: number of good apples picked out of all the apples picked out;
- __Recall__:    number of good apples picked out of all the apples in the basket;

__Precision__ is about exactness, classifying only one instance correctly yields 100% precision, but a very low recall, it tells us how well the system identifies samples from a given class.

__Recall__ is about completeness, classifying all instances as positive yields 100% recall, but a very low precision, it tells how well the system does and identify all the samples from a given class.

We will see further ahead how to get the best out of these two metrics, using Precision-Recall curves.

Typically these two metrics are combined together in a metric called $F_{1}$ (i.e., harmonic mean of precision and recall), which eases comparison of different systems, and problems with many classes. They are defined as:

$$ Precision = \frac{TP}{TP+FP}$$  

$$ Recall = \frac{TP}{TP+FN}$$

$$ F_{1} = 2 \times\frac{Precision \times Recall}{Precision + Recall}$$


Note that you need to measure this for every possible class in your dataset. So, __Precision__ and __Recall__ metrics are appropriate when you are dealing with imbalanced datasets.

---

## __Sensitivity and Specificity__

These two metrics are somehow related to __Precision__ and __Recall__, and although not often, I saw them being used a couple of times in NLP-related problems:

$$ Sensitivity = \frac{TP}{TP+FP}$$

$$ Specificity = \frac{TN}{TN+FP}$$

__Sensitivity__: is the same as recall, defined above.

__Specificity__: also called the true negative rate, measures the proportion of actual negatives that are correctly identified as such.

<!--
Equivalently, in medical tests sensitivity is the extent to which actual positives are not overlooked (so false negatives are few), and specificity is the extent to which actual negatives are classified as such (so false positives are few). Thus a highly sensitive test rarely overlooks an actual positive (for example, showing "nothing bad" despite something bad existing); a highly specific test rarely registers a positive classification for anything that is not the target of testing (for example, finding one bacterial species and mistaking it for another closely related one that is the true target); and a test that is highly sensitive and highly specific does both, so it "rarely overlooks a thing that it is looking for" and it "rarely mistakes anything else for that thing." Because most medical tests do not have sensitivity and specificity values above 99%, "rarely" does not equate to certainty. But for practical reasons, tests with sensitivity and specificity values above 90% have high credibility, albeit usually no certainty, in differential diagnosis.

Sensitivity therefore quantifies the avoiding of false negatives, and specificity does the same for false positives. For any test, there is usually a trade-off between the measures – for instance, in airport security since testing of passengers is for potential threats to safety, scanners may be set to trigger alarms on low-risk items like belt buckles and keys (low specificity), in order to increase the probability of identifying dangerous objects and minimize the risk of missing objects that do pose a threat (high sensitivity).
-->

---

<br>

## __Receiver Operating Characteristic (ROC) Curves__

While defining the metrics above, I assumed that we are directly given the predictions of each class. But it might be the case that we have the probability for each class instead, and it can allows to calibrate the threshold on how to interpret the probabilities.

By calibrating a threshold, a balance __precision__ and __recall__ or between __sensitivity__ and __specificity__ can be set.

The curve is essentially a plot where of false positive rate (x-axis) versus the true positive rate (y-axis) for a number of different candidate threshold values between 0.0 and 1.0. An operator may plot the ROC curve and choose a threshold that gives a desirable balance between the false positives and false negatives.


x-axis
======
The false positive rate is also referred to as the inverted specificity where specificity is the total number of true negatives divided by the sum of the number of true negatives and false positives.

False Positive Rate = False Positives / (False Positives + True Negatives)


y-axis (Precision)
==================
The true positive rate is calculated as the number of true positives divided by the sum of the number of true positives and the number of false negatives. It describes how good the model is at predicting the positive class when the actual outcome is positive.

True Positive Rate = True Positives / (True Positives + False Negatives)

<!--
A skilful model will assign a higher probability to a randomly chosen real positive occurrence than a negative occurrence on average. This is what we mean when we say that the model has skill. Generally, skilful models are represented by curves that bow up to the top left of the plot.

A model with no skill is represented at the point [0.5, 0.5]. A model with no skill at each threshold is represented by a diagonal line from the bottom left of the plot to the top right and has an AUC of 0.0.

A model with perfect skill is represented at a point [0.0 ,1.0]. A model with perfect skill is represented by a line that travels from the bottom left of the plot to the top left and then across the top to the top right.

An operator may plot the ROC curve for the final model and choose a threshold that gives a desirable balance between the false positives and false negatives.
-->



## __Precision-Recall Curve__

As shown before when one has imbalanced classes, precision and recall are better metrics than accuracy, in the same way, for imbalanced datasets a precision-recall curve is more suitable as ROC curve.

<!--
Key to the calculation of precision and recall is that the calculations do not make use of the true negatives. It is only concerned with the correct prediction of the minority class, class 1.

A precision-recall curve is a plot of the precision (y-axis) and the recall (x-axis) for different thresholds, much like the ROC curve.

The no-skill line is defined by the total number of positive cases divide by the total number of positive and negative cases. For a dataset with an equal number of positive and negative cases, this is a straight line at 0.5. Points above this line show skill.

A model with perfect skill is depicted as a point at [1.0,1.0]. A skilful model is represented by a curve that bows towards [1.0,1.0] above the flat line of no skill.

There are also composite scores that attempt to summarize the precision and recall; three examples include:

F score or F1 score: that calculates the harmonic mean of the precision and recall (harmonic mean because the precision and recall are ratios).
Average precision: that summarizes the weighted increase in precision with each change in recall for the thresholds in the precision-recall curve.
Area Under Curve: like the AUC, summarizes the integral or an approximation of the area under the precision-recall curve.
In terms of model selection, F1 summarizes model skill for a specific probability threshold, whereas average precision and area under curve summarize the skill of a model across thresholds, like ROC AUC.

This makes precision-recall and a plot of precision vs. recall and summary measures useful tools for binary classification problems that have an imbalance in the observations for each class.
-->



__NOTE__: In both cases the area under the curve (AUC) can be used as a summary of the model skill.









## __Summary__

If you have an imbalanced dataset __accuracy__ can give you false assumptions regarding the classifier's performance, it's better to rely on __precision__ and __recall__, in the same way a Precision-Recall curve is better to calibrate the probability threshold in an imbalanced class scenario as a ROC curve.

- __ROC Curves__: summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.

- __Precision-Recall curves__: summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.

ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets.


Accuracy              Overall performance of model
Precision             How accurate the positive predictions are
Recall/Sensitivity	  Coverage of actual positive sample
Specificity	          Coverage of actual negative sample
F1 score	            Hybrid metric useful for unbalanced classes


### References

- [https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)

https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python

<!--
https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/
-->

