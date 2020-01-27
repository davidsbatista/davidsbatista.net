---
layout: post
title: The Attention Mechanism
date: 2020-01-25 00:00:00
tags: attention nlp seq2seq
categories: [blog]
comments: false
disqus_identifier: 2020125
preview_pic: # /assets/images/2019-11-03-pt-embeddings.jpg
---

The __Attention__ mechanism is now an established


## __Introduction__

To the best of my knowledge the first of the attention mechanism
was first presented in _"Neural Machine Translation by Jointly Learning to Align and Translate"_
at ICLR 2015 ([Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf))

In the paper the authors propose to tackle the problem of a fixed-length context
vector in the original seq2seq model for machine translation
([Cho et al., 2014](https://www.aclweb.org/anthology/D14-1179/))

The seq2seq model is composed of:

- an __encoder__ reads the input sequence compressing the information into a context vector.

- an __decoder__ is initialised with the context vector to emit the transformed output.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2020-01-25-seq2seq.jpeg">
  <figcaption>Figure 1: A seq2seq model composed of an encoder and decoder.</figcaption>
</figure>

The fixed size context-vector needs to contain a good summary of the meaning of
the whole source sequence, being one big bottleneck, specially for long sentences.
This was one of the motivations by [Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf),
which proposed a similar architecture with a crucial improvement:

"_The new architecture consists of a bidirectional RNN as an encoder and a decoder that emulates searching
through a source sentence during decoding a translation_"

Emulation a search through the source sentence is done through the so called
attention mechanism

<figure>
  <img style="width: 40%; height: 40%" src="/assets/images/2020-01-25-seq2seq_with_attention.png">
  <figcaption>Figure 2: A seq2seq model composed of an encoder and decoder.</figcaption>
</figure>





Attention uses a context vector to attend to all words








"Effective Approaches to Attention-based Neural Machine Translation"
https://www.aclweb.org/anthology/D15-1166/

---

## __Seq2Sequence Example__

<!--

ToDo: usar o exemplo e fazer uma tradução com português?

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

-->

## __Document-Level Classification Example__

## __Sequence Labelling Example__

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



<!--
## Examples applied to doc-level tasks:

Attention-based LSTM for Aspect-level Sentiment Classification
https://www.aclweb.org/anthology/D16-1058.pdf


Attention-Based Bidirectional Long Short-Term Memory Networks for
Relation Classification
https://www.aclweb.org/anthology/P16-2034.pdf

-->


## __References__

- [1] [An Introductory Survey on Attention Mechanisms in NLP Problems](https://link.springer.com/chapter/10.1007/978-3-030-29513-4_31) ([arXiv.org version](https://arxiv.org/pdf/1811.05544.pdf))([presentation slides](https://www.iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2015:bahdanau-iclr2015.pdf))


- __Figure 1__ taken from https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)

<!--
https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

EMNLP 2015: "effective approaches to attention-based neural machine translation
https://vimeo.com/162101582

-->