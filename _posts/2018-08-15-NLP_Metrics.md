---
layout: post
title: Evaluation Metrics, ROC-Curves and imbalanced datasets
date: 2018-08-19 00:0:00
categories: blog
tags: evaluation_metrics imbalanced_data classification
#comments: true
#disqus_identifier: 20180819
preview_pic: /assets/images/2018-08-19-ROC-Curve.png
description: This blog post describes some evaluation metrics used in NLP, it points out where we should use each one of them and the advantages and disadvantages of each.
---

I wrote this blog post with the intention to review and compare some evaluation metrics typically used for classification tasks, and how they should be used depending on the the dataset. I also show how one can tune the probability thresholds for the particularly metrics.

You can find the complete code associated with this blog post on this repository:

* __[https://github.com/davidsbatista/machine-learning-notebooks/](https://github.com/davidsbatista/machine-learning-notebooks/blob/master/ROC-Curve-vs-Precision-Recall-Curve.ipynb)__



# __Introduction__

When making a prediction for a two-class classification problem, the following types of errors can be made by a classifier:

- __False Positive (FP)__: predict an event when there was no event.
- __False Negative (FN)__: predict no event when in fact there was an event.
- __True Positive (TP)__: predict an event when there was an event.
- __True Negative (TN)__: predict no event when in fact there was no event.

---

## __Accuracy__

Accuracy simply measures the number of correct predicted samples over the total number of samples. For instance, if the classifier is 90% correct, it means that out of 100 instance it correctly predicts the class for 90 of them.

$$ \textrm{accuracy} = \frac{\textrm{nr. correct predictions}}{\textrm{nr. total predictions}} = \frac{\textrm{TP+TN}}{\textrm{TP+TN+FP+FN}}$$

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

$$ \textrm{Precision} = \frac{\textrm{TP}}{\textrm{TP+FP}}$$  

$$ \textrm{Recall} = \frac{\textrm{TP}}{\textrm{TP+FN}}$$

$$ F_{1} = 2 \times\frac{\textrm{Precision} \times \textrm{Recall}}{\textrm{Precision} + \textrm{Recall}}$$


Note that you need to measure this for every possible class in your dataset. So, __Precision__ and __Recall__ metrics are appropriate when you are dealing with imbalanced datasets.

---

## __Sensitivity and Specificity__

These two metrics are somehow related to __Precision__ and __Recall__, and although not often, I saw them being used a couple of times in NLP-related problems:

$$ \textrm{Sensitivity} = \frac{\textrm{TP}}{\textrm{TP+FP}}$$

$$ \textrm{Specificity} = \frac{\textrm{TN}}{\textrm{TN+FP}}$$

__Sensitivity__: is the same as recall, defined above, can be tought of as the extent to which actual positives are not overlooked, so false negatives are few.

__Specificity__: also called the true negative rate, measures the proportion of actual negatives that are correctly identified as such, i.e., is the extent to which actual negatives are classified as such (so false positives are few).

Sensitivity therefore quantifies the avoiding of false negatives, and specificity does the same for false positives.

There some scenarios where focusing on one of these two might be important, e.g:

- Sensitivity: the percentage of sick people who are correctly identified as having the condition.
- Specificity: the percentage of healthy people who are correctly identified as not having the condition.

---

<br>

## __Receiver Operating Characteristic (ROC) Curves__

While defining the metrics above, I assumed that we are directly given the predictions of each class. But it might be the case that we have the probability for each class instead, which then allows to calibrate the threshold on how to interpret the probabilities. Does it belong to positive class if it's greater than 0.5 or 0.3 ?

The curve is a plot of _false positive rate_ (x-axis) versus the _true positive rate_ (y-axis) for a number of different candidate threshold values between 0.0 and 1.0. An operator may plot the ROC curve and choose a threshold that gives a desirable balance between the false positives and false negatives.


* __x-axis__: the false positive rate is also referred to as the inverted specificity where specificity is the total number of true negatives divided by the sum of the number of true negatives and false positives.

$$\textrm{False Positive Rate} = \frac{\textrm{FP}}{\textrm{FP+TN}}$$

* __y-axis__: the true positive rate is calculated as the number of true positives divided by the sum of the number of true positives and the number of false negatives. It describes how good the model is at predicting the positive class when the actual outcome is positive.

$$\textrm{True Positive Rate} = \frac{\textrm{TP}}{\textrm{TP+FN}}$$


_NOTE_: remember that both the False Positive Rate and the True Positive Rate are calculated for different probability thresholds.


## __Precision-Recall Curve__

As shown before when one has imbalanced classes, precision and recall are better metrics than accuracy, in the same way, for imbalanced datasets a Precision-Recall curve is more suitable than a ROC curve.

A Precision-Recall curve is a plot of the __Precision__ (y-axis) and the __Recall__ (x-axis) for different thresholds, much like the ROC curve. Note that in computing precision and recall there is never a use of the true negatives, these measures only consider correct predictions

---
<br>

## __Pratical Example__


### Let's first generate a 2 class imbalanced dataset
```python
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.95,0.05], random_state=42)
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2)
```

### Train a model for classification

```python
model = LogisticRegression()
model.fit(trainX, trainy)
predictions = model.predict(testX)
```

### Comparing the Accuracy vs. Precision-Recall with imbalanced data


```python
accuracy = accuracy_score(testy, predictions)
print('Accuracy: %.3f' % accuracy)
```

    Accuracy: 0.957



```python
print(classification_report(testy, predictions))
```

                 precision    recall  f1-score   support

              0       0.96      0.99      0.98      1884
              1       0.73      0.41      0.53       116

    avg / total       0.95      0.96      0.95      2000



### ROC Curve vs. Precision-Recall Curve with imbalenced data


```python
probs = model.predict_proba(testX)
probs = probs[:, 1]
```


```python
fpr, tpr, thresholds = roc_curve(testy, probs)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.show()
auc_score = roc_auc_score(testy, probs)
print('AUC: %.3f' % auc_score)
```


![png](/assets/images/2018-08-19-ROC-Curve.png)


    AUC: 0.920


# Precision-Recall curve


```python
precision, recall, thresholds = precision_recall_curve(testy, probs)
auc_score = auc(recall, precision)
```


```python
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
pyplot.plot(recall, precision, marker='.')
pyplot.show()
print('AUC: %.3f' % auc_score)
```


![png](/assets/images/2018-08-19-Precision-Recall-Curve.png)


    AUC: 0.577





---
<br>

## __Summary__

If you have an imbalanced dataset __accuracy__ can give you false assumptions regarding the classifier's performance, it's better to rely on __precision__ and __recall__, in the same way a Precision-Recall curve is better to calibrate the probability threshold in an imbalanced class scenario as a ROC curve.

- __ROC Curves__: summarise the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.

- __Precision-Recall curves__: summarise the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.

ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets. In both cases the area under the curve (AUC) can be used as a summary of the model performance.


<center>
<table class="blueTable">
<thead>
<tr>
<th>Metric</th>
<th>Formula</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Accuracy</td>
<td>$$\frac{\textrm{TP+TN}}{\textrm{TP+TN+FP+FN}}$$</td>
<td>Overall performance of model</td>
</tr>
<tr>
<td>Precision</td>
<td>$$\frac{\textrm{TP}}{\textrm{TP+FP}}$$  </td>
<td>How accurate the positive predictions are</td>
</tr>
<tr>
<td>Recall/Sensitivity</td>
<td>$$\frac{\textrm{TP}}{\textrm{TP+FN}}$$</td>
<td>Coverage of actual positive sample</td>
</tr>
<tr>
<td>Specificity</td>
<td>$$\frac{\textrm{TN}}{\textrm{TN+FP}}$$</td>
<td>Coverage of actual negative sample</td>
</tr>
<tr>
<td>F1-score</td>
<td>$$2 \times\frac{\textrm{Precision} \times \textrm{Precision}}{\textrm{Precision} + \textrm{Precision}}$$</td>
<td>Harmonic mean of Precision and Recall</td>
</tr>
</tbody>
</table>
</center>


### __References__

- [CS 229 - Machine Learning (tips and tricks cheatsheet)](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)

- [ROC Curves and Precision-Recall Curves for Classification](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python)

- [Sensitivity and Specificity (Wikipedia)](https://www.wikiwand.com/en/Sensitivity_and_specificity)

- [The Precision-Recall Plot Is More Informative
than the ROC Plot When Evaluating Binary
Classifiers on Imbalanced Datasets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)

<!--
https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/
-->

