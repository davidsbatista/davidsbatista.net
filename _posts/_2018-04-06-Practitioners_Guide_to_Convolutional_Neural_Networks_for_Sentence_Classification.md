---
layout: post
title: Guide to Convolutional Neural Networks for Sentence Classification
date: 2018-04-06 00:00:00
categories: [blog]
tags: convolutional-neural-networks document-classification deep-learning
comments: false
# disqus_identifier: #20180406
preview_pic: /assets/images/2018-03-31-cnn-sentence-classification.png
description: Guide to Convolutional Neural Networks for Sentence Classification
---

This post explores a paper which studies the effect of different parameters of a Convolutional Neural Network architecture affect the performance in the task of sentence classification.


# __Convolutional Neural Networks for NLP__

<!--
In the case of NLP tasks, i.e., when applied to text instead of images, we have a 1 dimensional array representing the text. Here the architecture of the ConvNets is changed to 1D convolutional-and-pooling operations.
-->

<!--
https://www.aclweb.org/anthology/I/I17/I17-1026.pdf

http://www.aclweb.org/anthology/C14-1008


https://uwspace.uwaterloo.ca/bitstream/handle/10012/9592/Chen_Yahui.pdf?sequence=3&isAllowed=y

Papers experiments
==================
A Convolutional Neural Network for Modelling Sentences (Nal Kalchbrenner)
http://www.aclweb.org/anthology/P14-1062

http://riejohnson.com/paper/dpcnn-acl17.pdf

A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification
https://www.aclweb.org/anthology/I/I17/I17-1026.pdf


Start with a baseline configuration and study:

### Effect of input word vectors:
* Consider starting with the basic configuration described in Table 1 and using non-static word2vec or GloVe.

### Effect of filter region size

The filter region size can have a large effect on performance, and should be tuned. Line-search over the single filter region size to find the ‘best’ single region size. A rea- sonable range might be 1∼10. However, for datasets with very long sentences like CR, it may be worth exploring larger filter region sizes. Once this ‘best’ region size is iden- tified, it may be worth exploring combining multiple filters using regions sizes near this single best size, given that empirically multi- ple ‘good’ region sizes always outperformed using only the single best region size.


### Effect of number of feature maps for each filter region size

### Effect of activation function

### Effect of pooling strategy

### Effect of regularization

Neural Networks for NLP
http://www.phontron.com/class/nn4nlp2017/schedule.html


I think of model evaluation in four different categories:

Underfitting – Validation and training error high
Overfitting – Validation error is high, training error low
Good fit – Validation error low, slightly higher than the training error
Unknown fit - Validation error low, training error 'high'

Implementations
===============
https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/

-->


## __References__

<!--
* ["Convolutional Neural Networks for Sentence Classification" Y. Kim 2014 in Conference on Empirical Methods in Natural Language Processing (EMNLP'14)](http://www.aclweb.org/anthology/D14-1181)
-->