---
layout: post
title: Convolutional Neural Networks for Text Classification
date: 2018-03-31 00:0:00
categories: [blog]
tags: convolutional-neural-networks document-classification
comments: true
disqus_identifier: 20180331
preview_pic: /assets/images/2018-03-31-Kim_cnn-sentence-classification.png
description: Convolutional Neural Networks for Sentence Classification
---

Convolutional Neural Networks (ConvNets) have in the past years shown break-through results in some NLP tasks, one particular task is sentence classification, i.e., classifying short phrases (i.e., around 20~50 tokens), into a set of pre-defined categories. In this post I will explain how ConvNets can be applied to classifying short-sentences and how to easily implemented them in Keras.

You can find the complete code associated with this blog post on this repository:

[https://github.com/davidsbatista/ConvNets-for-sentence-classification](https://github.com/davidsbatista/ConvNets-for-sentence-classification)

# __Convolutional Neural Networks__

ConvNets were initially developed in the neural network image processing community where they achieved break-through results in recognising an object from a pre-defined category (e.g., cat, bicycle, etc.).

A Convolutional Neural Network typically involves two operations, which can be though of as feature extractors: __convolution__ and __pooling__.

The output of this sequence of operations is then typically connected to a fully connected layer which is in principle the same as the traditional multi-layer perceptron neural network (MLP).

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-03-31-mylenet_cnn.png">
  <figcaption>The Convolutional Neural Network architecture applied to image classification. <br> (Image adapted from http://deeplearning.net/)</figcaption>
</figure>

<br>


### __Convolutions__

We can think about the input image as a matrix, where each entry represents each pixel, and a value between 0 and 255 representing the brightness intensity. Let's assume it's a black and white image with just one [__channel__](https://www.wikiwand.com/en/Channel_(digital_image)) representing the grayscale. If you would be processing a colour image, and taking into account the colours one would have 3 channels, following the [__RGB colour mode__](https://www.wikiwand.com/en/RGB_color_model).

One way to understand the convolution operation is to imagine placing the __convolution filter__ or __kernel__ on the top of the input image, positioned in a way so that the __kernel__ and the image upper left corners coincide, and then multiplying the values of the input image matrix with the corresponding values in the __convolution filter__.

All of the multiplied values are then added together resulting in a single scalar, which is placed in the first position of a result matrix.

The __kernel__ is then moved $$x$$ pixels to the right, where $$x$$ is denoted __stride length__ and is a parameter of the ConvNet structure. The process of multiplication is then repeated, so that the next value in the result matrix is computed and filled.

This process is then repeated, by first covering an entire row, and then shifting down the columns by the same __stride length__, until all the entries in the input image have been covered.

The output of this process is a matrix with all it's entries filled, called the __convoluted feature__ or __input feature map__.

An input image can be convolved with multiple convolution kernels at once, creating one output for each kernel.

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-03-31_dpln_0412_cnn.png">
  <figcaption>Example of a convolution operation. <br> (Image adapted from <i>"Deep Learning"</i> by Adam Gibson, Josh Patterson)</figcaption>
</figure>

<br>


### __Pooling__

Next comes the  __pooling__ or __downsampling__ layer, which consists of applying some operation over regions/patches in the __input feature map__ and extracting some representative value for each of the analysed regions/patches.

This process is somehow similar to the convolution described before, but instead of transforming local patches via a learned linear transformation (i.e., the __convolution filter__), they’re transformed via a hardcoded operation.

 Two of the most common pooling operations are max- and average-pooling. __Max-pooling__ selects the maximum of the values in the __input feature map__ region of each step and __average-pooling__ the average value of the values in the region. The output in each step is therefore a single scalar, resulting in significant size reduction in output size.

 <figure>
   <img style="width: 75%; height: 75%" src="/assets/images/2018-03-31_cnn_pooling.jpg">
   <figcaption>Example of a pooling operation with stride length of 2. <br> (Image adapted from <a href="https://medium.com/@Aj.Cheng/convolutional-neural-network-d9f69e473feb">AJ Cheng blog</a>)</figcaption>
 </figure>


Why do we downsample the feature maps and simply just don't remove the pooling layers and keep possibly large feature maps? François Chollet in _"Deep Learning with Python"_ summarises it well in this sentence:

_"The reason to use downsampling is to reduce the number of feature-map coefficients to process, as well as to induce spatial-filter hierarchies by making successive convolution layers look at increasingly large windows (in terms of the fraction of the original input they cover)."_

### __Fully Connected__

The two processes described before i.e.: convolutions and pooling, can been thought of as a feature extractors, then we pass this features, usually as a reshaped vector of one row, further to the network, for instance, a multi-layer perceptron to be trained for classification.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2018-03-31-mlp.png">
  <figcaption>Example of multi-layer perceptron network used to train for classification.</figcaption>
</figure>


This was a briefly description of the ConvNet architecture when applied to image processing, let's now see how we can adapt this architecture to Natural Language Processing tasks.

<br>

---

<br>

# __Convolutional Neural Networks for NLP__

In the case of NLP tasks, i.e., when applied to text instead of images, we have a 1 dimensional array representing the text. Here the architecture of the ConvNets is changed to 1D convolutional-and-pooling operations.

One of the most typically tasks in NLP where ConvNet are used is sentence classification, that is, classifying a sentence into a set of pre-determined categories by considering $$n$$-grams, i.e. it's words or sequence of words, or also characters or sequence of characters.


### __1-D Convolutions over text__

Given a sequence of words $$w_{1:n} = w_{1}, \ldots, w_{n}$$, where each is associated with an embedding vector of dimension $$d$$. A 1D convolution of width-$$k$$ is the result of moving a sliding-window of size $$k$$ over the sentence, and applying the same __convolution filter__ or __kernel__ to each window in the sequence, i.e., a dot-product between the concatenation of the embedding vectors in a given window and a weight vector $$u$$, which is then often followed by a non-linear activation function $$g$$.

Considering a window of words $$w_{i}, \ldots, w_{i+k}$$ the concatenated vector of the $$i$$th window is then:

<center>
$$x_{i} = [w_{i}, w_{i+1}, \ldots, w_{i+k}] \in R^{\ k\ \times\  d}$$
</center>

The __convolution filter__ is applied to each window, resulting in scalar values $$r_{i}$$, each for the $$i$$th window:

<center>
$$r_{i} = g(x_{i} \cdot u) \in R$$
</center>

In practice one typically applies more filters, $$u_{1}, \ldots, u_{l}$$, which can then be represented as a vector multiplied by a matrix $$U$$ and with an addition of a bias term $$b$$:

<center>
$$\text{r}_{i} = g(x_{i} \cdot U + b)$$

<br><br>

with $$\text{r}_{i} \in R^{l},\ \ \ x_{i} \in R^{\ k\ \times\  d},\ \ \  U \in R^{\ k\ \cdot\  d \ \times l}\ \ \  \text{and}\ \ \  b \in R^{l}$$
</center>

An example of a sentence convolution in a vector-concatenation notation:

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2018-03-31-sentence_convolution-example.png">
  <figcaption>Example of a sentence convolution with $$k$$=2 and dimensional output $$l$$=3. <br> (Image adapted from <a href="http://u.cs.biu.ac.il/~yogo/">Yoav Goldberg</a> book "Neural Network Methods for NLP")</figcaption>
</figure>


#### __Channels__

In the introduction above I assumed we were processing a black and white image, and therefore we have one matrix representing the grayscale intensity of each pixel. With the [__RGB colour mode__](https://www.wikiwand.com/en/RGB_color_model) each pixel would be a combination of three intensity values instead, one for each of Red, Green and Blue components, and such representation would be stored in three different matrices, providing different characteristics or view of the image, referred to as a [__Channel__](https://www.wikiwand.com/en/Channel_(digital_image)). It's common to apply a different set of filters to each channel, and then combine the three resulting vectors into a single vector.

We can also apply the multiple channels paradigm in text processing as well. For example, for a given phrase or window of text, one channel could be the sequence of words, another channel the sequence of corresponding POS tags, and a third one the shape of the words:

<center>
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-hgcj{font-weight:bold;text-align:center}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th>Word:</th>
    <th class="tg-hgcj">The</th>
    <th class="tg-hgcj">plane</th>
    <th class="tg-hgcj">lands</th>
    <th class="tg-hgcj">in</th>
    <th class="tg-amwm">Lisbon</th>
  </tr>
  <tr>
    <th>PoS-tag:</th>
    <td class="tg-hgcj">DET</td>
    <td class="tg-hgcj">NOUN</td>
    <td class="tg-hgcj">VERB</td>
    <td class="tg-hgcj">PROP</td>
    <td class="tg-amwm">NOUN</td>
  </tr>
  <tr>
    <th>Shape:</th>
    <td class="tg-hgcj">Xxx</td>
    <td class="tg-hgcj">xxxx</td>
    <td class="tg-hgcj">xxxx</td>
    <td class="tg-hgcj">xx</td>
    <td class="tg-amwm">Xxxxxx</td>
  </tr>
</table>
</center>

Applying the convolution over the words will result in $$m$$ vectors $$w$$, applying it over the PoS-tags will result also in $$m$$ vectors, and the same for the shapes, again $$m$$ vectors. These three different channels can then be combined either by summation:

<center>
 $$p_i = words_{1:m} + pos_{1:m} + shapes_{1:m}$$
</center>

or by concatenation:

<center>
 $$p_i = [words_{1:m}:pos_{1:m}:shapes_{1:m}]$$.
</center>

__NOTE__: each channel can still have different convolutions that read the source document using different kernel sizes, for instance, applying different context windows over words, pos-tags or shapes.

### __Pooling__

The pooling operation is used to combine the vectors resulting from different convolution windows into a single $$l$$-dimensional vector. This is done again by taking the _max_ or the _average_ value observed in resulting vector from the convolutions. Ideally this vector will capture the most relevant features of the sentence/document.

This vector is then fed further down in the network - hence, the idea that ConvNet itself is just a feature extractor - most probably to a full connected layer to perform prediction.

---

### __Convolutional Neural Networks for Sentence Classification__

I did a quick experiment, based on the paper by Yoon Kim, implementing the 4 ConvNets models he used to perform sentence classification.

- __CNN-rand__: all words are randomly initialized and then modified during training

- __CNN-static__: pre-trained vectors with all the words— including the unknown ones that are randomly initialized—kept static and only the other parameters of the model are learned

- __CNN-non-static__: same as CNN-static but word vectors are fine-tuned

- __CNN-multichannel__: model with two sets of word vectors. Each set of vectors is treated as a channel and each filter is applied

Let's just first quickly look at how these different models look like in as a computational graph. The first three (i.e., CNN-rand, CNN-static and CNN-non-static) look pretty much the same:

<figure>
  <img style="width: 100%; height: 100%" src="/assets/images/2018-03-31-SentenceClassificationConvNets-no_multi_channel.svg">
  <figcaption></figcaption>
</figure>


The CNN-multichannel model uses two embedding layers, in one channel the embeddings are updated, in the second they remain static. It's exactly the same network as above but duplicated and adding an extra layer do concatenate both results into a single vector:

<figure>
  <img style="width: 100%; height: 100%" src="/assets/images/2018-03-31-SentenceClassificationConvNets-multi_channel.svg">
  <figcaption></figcaption>
</figure>

---

# __Experiments and Results__

I applied the implemented models on same of the datasets that Kim reported, but I could not get exactly the same results, first his results were reported over, I believe a Tensorflow implementation, and then there is the issue of how the datasets are pre-processed, i.e., tokenised, cleaned, etc.; that will always impact the results.

Another issue which puzzles me is that all those experiments only take into consideration the accuracy. Since the class samples are not uniformly distributed across the different classes I think this is the wrong way to evaluate a classifier.

All the code for the models and experiments is available here:

[https://github.com/davidsbatista/ConvNets-for-sentence-classification](https://github.com/davidsbatista/ConvNets-for-sentence-classification)

---

## __Summary__

The CNN is just a feature-extraction architecture, alone itself is not useful, but is the fist building block of a larger network. It needs to be trained together with a classification layer in order to produce some useful results.

As Yoav Goldberg summarises it:

_"The CNN layer’s responsibility is to extract meaningful sub-structures that are useful for the overall prediction task at hand. A convolutional neural network is designed to identify indicative local predictors in a large structure, and to combine them to produce a fixed size vector representation of the structure, capturing the local aspects that are most informative for the prediction task at hand. In the NLP case the convolutional architecture will identify $$n$$-grams that are predictive for the task at hand, without the need to pre-specify an embedding vector for each possible ngram."_

* __convolution__ : an operation which applies a filter to a fixed size window.
* __convolution filter__ or __kernel__: a template matrix which is used in the convolution operation.
* __pooling__: combines the vectors resulting from different convolution windows into a single $$l$$-dimensional vector.
* __feature_maps__ : the number of feature maps directly controls capacity and depends on the number of available examples and task complexity.

## __References__

* [__"Convolutional Neural Networks for Sentence Classification" Y. Kim 2014 in Conference on Empirical Methods in Natural Language Processing (EMNLP'14)__](http://www.aclweb.org/anthology/D14-1181)

* [__"Deep Learning" by Adam Gibson, Josh Patterson (O'Reilly Media, Inc. 2017)__](https://www.oreilly.com/library/view/deep-learning/9781491924570/)

* __["Neural Network Methods for Natural Language Processing"](http://www.morganclaypoolpublishers.com/catalog_Orig/product_info.php?products_id=1056) by [Yoav Goldberg](http://www.cs.biu.ac.il/~yogo/) (Morgan & Claypool Publishers 2017)__

* __["Deep Learning with Python"](https://www.manning.com/books/deep-learning-with-python) by [François Chollet](https://github.com/fchollet) (Manning Publications 2017)__
