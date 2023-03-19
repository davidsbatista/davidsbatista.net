---
layout: post
title: Sentence Transformers
date: 2023-02-05 00:00:00
tags: Sentence Transformers
categories: [blog]
comments: true
disqus_identifier: 20230205
preview_pic:
---


## Having a sentence from BERT


We feed the input sentence or text into a transformer network like BERT. BERT produces contextualized word embeddings for all input tokens in our text. As we want a fixed-sized output representation (vector u), we need a pooling layer. Different pooling options are available, the most basic one is mean-pooling: We simply average all contextualized word embeddings BERT is giving us. This gives us a fixed 768 dimensional output vector independent how long our input text was.


<!-- 

https://www.sbert.net/docs/training/overview.html

https://www.pinecone.io/learn/sentence-embeddings/ 

https://towardsdatascience.com/an-intuitive-explanation-of-sentence-bert-1984d144a868

-->


## A Siamese Neural Network 

% https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942

A Siamese Neural Network is a class of neural network architectures that contain two or more identical subnetworks, ‘identical’ meaning they have the same configuration with the same parameters and weights. Parameter updating is mirrored across both sub-networks. 

Siamese Neural Network (SNNs) can be used to find the similarity of the inputs by comparing its feature vectors, so these networks are used in many application

SNNs learn a similarity function. Thus, we can train it to see if the two images are the same. Learning from Semantic Similarity: Siamese focuses on learning embeddings (in the deeper layer) that place the same classes/concepts close together. Hence, can learn semantic similarity.

### Losses

__Triplet loss__ is a loss function where a baseline (anchor) input is compared to a positive (truthy) input and a negative (falsy) input. The distance from the baseline (anchor) input to the positive (truthy) input is minimized, and the distance from the baseline (anchor) input to the negative (falsy) input is maximized.

__Contrastive Loss__