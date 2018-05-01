---
layout: post
title: Sentence Classification with Convolutional Neural Networks
date: 2018-03-31 00:0:00
categories: [blog]
tags: convolutional-neural-networks document-classification deep-learning
comments: false
# disqus_identifier: #20180223
preview_pic: /assets/images/2018-03-31-cnn-sentence-classification.png
description: Sentence Classification with Convolutional Neural Networks
---

Convolutional Neural Networks (ConvNets) have in the past years shown break-through results in some NLP tasks, one particular task is sentence classification, i.e., classifying short phrases (i.e., around 20~50 tokens), into a set of pre-defined categories. Examples of such sentences are reviews classifications (e.g.: very positive, positive, neutral, negative, very negative), or classifying a sentence as being subjective or objective. In this post I will explain how ConvNets can be applied to classifying short-sentences and how to easily implemented them in Keras.


# __Convolutional Neural Networks__

ConvNets were initially developed in the neural network image processing community where they achieved break-through results in recognising an object from a pre-defined category (e.g., cat, bicycle, etc.).

A Convolutional Neural Network typically involves two operations, which can be though of as feature extractors: __convolution__ and __pooling__.

The output of this sequence of operations is then typically connected to a fully connected layer which is in principle the same as the traditional multi-layer perceptron neural network (MLP).

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-03-31-mylenet_cnn.png">
  <figcaption>The Convolutional Neural Network architecture applied to images. <br> (Image adapted from http://deeplearning.net/)</figcaption>
</figure>

<br>

<!--

kernel
feature_maps : The number of feature maps directly controls capacity and depends on the number of available examples and task complexity.
strides

-->


### __Convolutions__

__TODO__: black and white image with just 1 and 0

We can think about the input image as a matrix representing each pixel, and a value between
0 and 255 representing the brightness intensity. Let's assume it's a black and white image with just one [channel](https://www.wikiwand.com/en/Channel_(digital_image)) representing the grayscale.

One way to understand the convolution operation is to imagine placing the __convolution filter__ or __kernel__ on the top of the input image, positioned in a way so that the __kernel__ and the image upper left corners coincide, and then multiplying the values of the input image matrix with the corresponding values in the __convolution filter__.

All of the multiplied values are then added together resulting in a single scalar, which is placed in the first position of a result matrix.

The __kernel__ is then moved $$x$$ pixels to the right, where $$x$$ is denoted __stride length__ and is a parameter of the ConvNet structure. The process is then repeated, and the next value in the result matrix is computed and filled.

This process is then repeated, first covering an entire row, then shifting down the columns with the same __stride length__ until all entries in the input image have been covered.

The output of this process is a matrix with all it's entries filled, called the __convoluted feature__.

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-03-31_dpln_0412_cnn.png">
  <figcaption>Example of a convolution operation. <br> (Image adapted from <i>"Deep Learning"</i> by Adam Gibson, Josh Patterson)</figcaption>
</figure>

<br>





### __Pooling__


<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-03-31_cnn_pooling.jpg">
  <figcaption>Example of a pooling operation. <br> (Image adapted from ...)</figcaption>
</figure>

<!-- https://medium.com/@Aj.Cheng/convolutional-neural-network-d9f69e473feb -->


### __Prediction__


# __Convolutional Neural Networks for NLP__

In the case of NLP tasks, were are going to apply a 1D convolutional-and-pooling architecture.

<!--
### Ngram Detectors
-->

# __Summary__


<!--
Papers experiments
==================
A Convolutional Neural Network for Modelling Sentences (Nal Kalchbrenner)
http://www.aclweb.org/anthology/P14-1062

http://riejohnson.com/paper/dpcnn-acl17.pdf

Kim (mais completo:)
https://arxiv.org/pdf/1408.5882.pdf

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


## References

<!--

Livro do Yoav Goldberg

Wikipedia article of cnn

http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

Understanding the Mathematics of the Convolution Layer
http://deeplearningthesis.com/2018/02/08/understanding_the_mathematics_of_the_convolution_layer.html

http://colah.github.io/posts/2014-07-Understanding-Convolutions/

http://deeplearningthesis.com/2018/02/08/understanding_the_mathematics_of_the_convolution_layer.html

https://www.safaribooksonline.com/library/view/deep-learning/9781491924570/ch04.html

-->