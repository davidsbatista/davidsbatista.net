---
layout: post
title: Convolutional Neural Networks for Text Classification
date: 2018-03-31 00:0:00
categories: [blog]
tags: convolutional-neural-networks document-classification deep-learning
comments: false
# disqus_identifier: #20180331
preview_pic: /assets/images/2018-03-31-Kim_cnn-sentence-classification.png
description: Convolutional Neural Networks for Sentence Classification
---

Convolutional Neural Networks (ConvNets) have in the past years shown break-through results in some NLP tasks, one particular task is sentence classification, i.e., classifying short phrases (i.e., around 20~50 tokens), into a set of pre-defined categories. In this post I will explain how ConvNets can be applied to classifying short-sentences and how to easily implemented them in Keras.

<!--
Examples of such sentences are reviews classifications (e.g.: very positive, positive, neutral, negative, very negative), or classifying a sentence as being subjective or objective.
-->

# __Convolutional Neural Networks__

ConvNets were initially developed in the neural network image processing community where they achieved break-through results in recognising an object from a pre-defined category (e.g., cat, bicycle, etc.).

A Convolutional Neural Network typically involves two operations, which can be though of as feature extractors: __convolution__ and __pooling__.

The output of this sequence of operations is then typically connected to a fully connected layer which is in principle the same as the traditional multi-layer perceptron neural network (MLP).

__TODO__: imagem onde se vejam os passos todos

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-03-31-mylenet_cnn.png">
  <figcaption>The Convolutional Neural Network architecture applied to images. <br> (Image adapted from http://deeplearning.net/)</figcaption>
</figure>

<br>


### __Convolutions__

__TODO__: black and white image with just 1 and 0

We can think about the input image as a matrix representing each pixel, and a value between
0 and 255 representing the brightness intensity. Let's assume it's a black and white image with just one [channel](https://www.wikiwand.com/en/Channel_(digital_image)) representing the grayscale.

One way to understand the convolution operation is to imagine placing the __convolution filter__ or __kernel__ on the top of the input image, positioned in a way so that the __kernel__ and the image upper left corners coincide, and then multiplying the values of the input image matrix with the corresponding values in the __convolution filter__.

All of the multiplied values are then added together resulting in a single scalar, which is placed in the first position of a result matrix.

The __kernel__ is then moved $$x$$ pixels to the right, where $$x$$ is denoted __stride length__ and is a parameter of the ConvNet structure. The process of multiplication is then repeated, so that the next value in the result matrix is computed and filled.

This process is then repeated, by first covering an entire row, and then shifting down the columns by the same __stride length__, until all the entries in the input image have been covered.

The output of this process is a matrix with all it's entries filled, called the __convoluted feature__ or __input feature map__.

Finally an input can be convolved with multiple convolution kernels at once, creating one output for each kernel.

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-03-31_dpln_0412_cnn.png">
  <figcaption>Example of a convolution operation. <br> (Image adapted from <i>"Deep Learning"</i> by Adam Gibson, Josh Patterson)</figcaption>
</figure>

<br>

<!--
Rectified Linear Units (ReLU)
ReLU is something usually encountered when discussing CNNs, it scared us a bit in the beginning since we didn’t quite understand what it meant, but it is actually not especially hard.

It is a layer that usually comes directly after each convolution layer, in order to introduce nonlinearity in the system, since the calculations in the convolution layers are linear (just element wise multiplications and summations). Nonlinear functions such as tanh and sigmoids were used for this in the past, but ReLU layers were found to work better since they train a lot faster without making significant difference to the accuracy. The ReLU layer simply applies the function

f(x) = max(0,x)

to all the values in the input volume, which basically mean that it changes all negative values to 0. The different activation functions looks like this
 -->



### __Pooling__

Next comes the  __pooling__ or __downsampling__ layer, which consists of applying some operation over regions/patches in the __input feature map__ map and extracting some representative value for each of the analysed regions/patches.

This process is somehow similar to the convolution described before, but instead of transforming local patches via a learned linear transformation (i.e., the __convolution filter__), they’re transformed via a hardcoded operation.

 Two of the most common pooling operations are max- and average-pooling. Max-pooling selects the maximum of the values in the __input feature map__ region of each step and average-pooling the average value of the values in the region. The output in each step is therefore a single scalar, resulting in significant size reduction in output size.

 <figure>
   <img style="width: 75%; height: 75%" src="/assets/images/2018-03-31_cnn_pooling.jpg">
   <figcaption>Example of a pooling operation with stride length of 2. <br> (Image adapted from ...)</figcaption>
 </figure>

 <!-- image from: https://medium.com/@Aj.Cheng/convolutional-neural-network-d9f69e473feb -->

Why do we downsample the feature maps and simply just don't remove the pooling layers and keep possibly large feature maps? François Chollet in _"Deep Learning with Python"_ summarises it in this sentence:

"The reason to use downsampling is to reduce the number of feature-map coefficients to process, as well as to induce spatial-filter hierarchies by making successive convolution layers look at increasingly large windows (in terms of the fraction of the original input they cover)."

### __Fully Connected__

The two processes described before i.e.: convolutions and pooling, can been thought of as a feature extractors, then we pass this features further to the network, for instance, a multi-layer perceptron to be trained for classification.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2018-03-31-mlp.png">
  <figcaption>Example of multi-layer perceptron network used to train for classification. <br> (Image adapted from ...)</figcaption>
</figure>


# __Convolutional Neural Networks for NLP__

In the case of NLP tasks, i.e., when applied to text instead of images, we have a 1 dimensional array representing the text. Here the architecture of the ConvNets is changed to 1D convolutional-and-pooling operations.


 Natural Language Processing (almost) from Scratch
<!--
http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf
-->

<!--
código original em Theano
https://github.com/yoonkim/CNN_sentence
-->

## __Sentence Classification__

### __Datasets__

### __Experiments and Results__

<!-- apenas a experienca do Kim, simples, notebook, partes do código, final link pó notebook -->

# __Summary__

* __convolution__ :
* __convolution filter__ or __kernel__:
* __pooling__:
* __feature_maps__ : the number of feature maps directly controls capacity and depends on the number of available examples and task complexity.
* __strides__ :

## __References__

* ["Convolutional Neural Networks for Sentence Classification" Y. Kim 2014 in Conference on Empirical Methods in Natural Language Processing (EMNLP'14)](http://www.aclweb.org/anthology/D14-1181)

* ["Deep Learning" by Adam Gibson, Josh Patterson (O'Reilly Media, Inc. 2017)](https://www.oreilly.com/library/view/deep-learning/9781491924570/)

* ["Neural Network Methods for Natural Language Processing" by Yoav Goldberg (Morgan & Claypool Publishers 2017)  2017)](http://www.cs.biu.ac.il/~yogo/)

* ["Deep Learning with Python" by François Chollet (Manning Publications Co. 2017)](https://github.com/fchollet)
