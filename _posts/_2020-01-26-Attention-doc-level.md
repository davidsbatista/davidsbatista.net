---
layout: post
title: The Attention Mechanism in Document-Level with Neural Networks
date: 2020-01-25 00:00:00
tags: attention nlp seq2seq
categories: [blog]
comments: false
disqus_identifier: 2020125
preview_pic: /assets/images/2020-01-25-seq2seq_with_attention.png
---

The __Attention__ mechanism is now an established


## __Introduction__


## __Attention in Document-Level Classification tasks__


<!--

https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch/tree/master/Model_Seq2Seq_Attention

https://github.com/prakashpandey9/Text-Classification-Pytorch/tree/master/models

https://towardsdatascience.com/nlp-learning-series-part-3-attention-cnn-and-what-not-for-text-classification-4313930ed566

->

<!--
Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
https://www.aclweb.org/anthology/P16-2034.pdf

Attention-based LSTM for Aspect-level Sentiment Classification
https://www.aclweb.org/anthology/D16-1058.pdf
-->

<!--

Example applied to document/sentence level classification task

# in the 'dot-product' approach:
#
# the lstm outputs for each word (i.e., (sequence of vectors)) is multiplied
# by final the lstm state, and then normalized through a softmax and this normalized weights
# are used to recompute the new last state by avering each the original 150 256-value of the elements in
# the input sequence

# in the 'additive' approach this is learned by a feed forward network, more precisely we want
# this feed-forward network to learn the weight for each word (i.e., the original 150 256-value of
# the elements in the input sequence ))

# [32, 150, 256]  x [32, 256, 1]  same shape as before

-->

## Hierarchical Attention Networks




## __References__

-
<!--
https://ruder.io/deep-learning-nlp-best-practices/index.html#attention

https://dzone.com/articles/self-attention-mechanisms-in-natural-language-proc

self-attention, two-way attention, key-value-predict models and hierarchical attention
https://medium.com/@joealato/attention-in-nlp-734c6fa9d983
-->



