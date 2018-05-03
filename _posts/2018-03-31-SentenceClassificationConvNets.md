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

__TODO__: imagem onde se vejam os passos todos

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
ReLU is something usually encountered when discussing CNNs, it scared us a bit in the beginning since we didn’t quite understand what it meant, but it is actually not especially hard. It is a layer that usually comes directly after each convolution layer, in order to introduce nonlinearity in the system, since the calculations in the convolution layers are linear (just element wise multiplications and summations). Nonlinear functions such as tanh and sigmoids were used for this in the past, but ReLU layers were found to work better since they train a lot faster without making significant difference to the accuracy. The ReLU layer simply applies the function
f
(
x
)
=
m
a
x
(
0
,
x
)
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

 <!-- https://medium.com/@Aj.Cheng/convolutional-neural-network-d9f69e473feb -->


<!--
“Why downsample feature maps this way? Why not remove the max-pooling layers and keep fairly large feature maps all the way up?”

“In short, the reason to use downsampling is to reduce the number of feature-map coefficients to process, as well as to induce spatial-filter hierarchies by making successive convolution layers look at increasingly large windows (in terms of the fraction of the original input they cover).”

“Note that max pooling isn’t the only way you can achieve such downsampling. As you already know, you can also use strides in the prior convolution layer. And you can use average pooling instead of max pooling, where each local input patch is transformed by taking the average value of each channel over the patch, rather than the max. But max pooling tends to work better than these alternative solutions. In a nutshell, the reason is that features tend to encode the spatial presence of some pattern or concept over the different tiles of the feature map (hence, the term feature map), and it’s more informative to look at the maximal presence of different features than at their average presence. So the most reasonable subsampling strategy is to first produce dense maps of features (via unstrided convolutions) and then look at the maximal activation of the features over small patches, rather than looking at sparser windows of the inputs (via strided convolutions) or averaging input patches, which could cause you to miss or dilute feature-presence information.”
-->








### __Fully Connected__



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

- Livro do Yoav Goldberg

- Livro do Keras

- Wikipedia article of cnn

http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

Understanding the Mathematics of the Convolution Layer
http://deeplearningthesis.com/2018/02/08/understanding_the_mathematics_of_the_convolution_layer.html

http://colah.github.io/posts/2014-07-Understanding-Convolutions/

https://www.safaribooksonline.com/library/view/deep-learning/9781491924570/ch04.html

-->