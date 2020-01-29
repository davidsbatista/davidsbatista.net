---
layout: post
title: The Attention Mechanism in Natural Language Processing
date: 2020-01-25 00:00:00
tags: attention nlp seq2seq
categories: [blog]
comments: false
disqus_identifier: 2020125
preview_pic: # /assets/images/2019-11-03-pt-embeddings.jpg
---

The __Attention__ mechanism is now an established


## __Introduction__: origins in the seq2seq model

To the best of my knowledge the first of the attention mechanism
was first presented in _"Neural Machine Translation by Jointly Learning to Align and Translate"_
at ICLR 2015 ([Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf))

In the paper the authors propose to tackle the problem of a fixed-length context
vector in the original seq2seq model for machine translation
([Cho et al., 2014](https://www.aclweb.org/anthology/D14-1179/))

### __The 'classical' seq2seq__

The seq2seq model is composed of two main components:

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2020-01-25-seq2seq.jpeg">
  <figcaption>Figure 1: A seq2seq model composed of an encoder and decoder.</figcaption>
</figure>

The __encoder__ reads the input sentence, a sequence of vectors $x = (x_{1}, \dots , x_{T})$,
into a fixed-length vector $c$. Typical approaches are RNN or LSTMs such that:

$$h_{t} = f\ (x_{t}, h_{t−1})$$

$$ c = q\ (h_{1}, \dotsc, h_{T}) $$

where $h_{t}$ is a hidden state at time $t$, and $c$ is a vector generated from
the sequence of the hidden states, and $f$ and $q$ are some nonlinear functions.


The __decoder__ is trained to predict the next word $$y_{t}$$ given the context
vector $$c$$ and all the previously predict words $\\{y_{1}, \dots , y_{t-1}\\}$.
the decoder defines a probability over the translation $${\bf y}$$ by decomposing
the joint probability into the ordered conditionals:

$$p({\bf y}) = \prod\limits_{i=1}^{x} p(y_{t} | {y_{1}, \dots , y_{t-1}}, c)$$

where $\bf y = \\{y_{1}, \dots , y_{t}\\}$. That is the probability of a
translation sequence is calculated by computing the conditional probability
of each word given the previous words. With an LSTM/RNN each conditional probability
is computed as:

$$p(y_{t} | {y_{1}, \dots , y_{t-1}}, c) = g(y_{t−1}, s_{t}, c)$$

where, $g$ is a nonlinear function that outputs the probability of y_{t},
s_{t} is the hidden state, and $c$ the context vector.

The fixed size context-vector needs to contain a good summary of the meaning of
the whole source sentence, being this fixed-size vector one big bottleneck,
specially for long sentences.

### __seq2seq with Attention__

This was one of the motivations by [Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf),
which proposed a similar architecture with a crucial improvement:

"_The new architecture consists of a bidirectional RNN as an encoder and a decoder that
emulates searching through a source sentence during decoding a translation_"

The encoder is now a bidirectional recurrent network with a forward hidden state
and a backward one. A simple concatenation of two represents the encoder state.
The motivation is to include both the preceding and following words in the
annotation of one word.

Another key element in the decoder is that now it's equipped with some sort of search
through the source sentence, done through the __attention mechanism__.

<figure>
  <img style="width: 40%; height: 40%" src="/assets/images/2020-01-25-seq2seq_with_attention.png">
  <figcaption>Figure 2: A seq2seq model composed of an encoder and decoder.</figcaption>
</figure>

They propose to replace the fixed-length context vector by a another context
vector $c_{i}$ which is a sum of the hidden states of the input sequence, weighted
by alignment scores.

$$p(y_{t} | {y_{1}, \dots , y_{t-1}}, c) = g(y_{t−1}, s_{t}, c)$$

where $s_{i}$ is the hidden state for time $i$, computed by:

$$s_{i} = f(s_{i−1}, y_{i−1}, c_{i})$$

the probability is conditioned on a distinct context vector c_{i} for each target
word $y$.

### __Context Vector__

How to compute the context vector $$c_{i}$$ ?

The context vector depends $$c_{i}$$ depends on a sequence of annotations to which
an encoder maps the input sentence. Each annotation contains information about
the whole input sequence with a strong focus on the parts surrounding the $$i_{th}$$
word of the input sequence. These annotations are simple the concatenation
of the two states from the forward and backward RNN/LSTM from the encoder for
each word in the input, h_{j}.

<!--
We obtain an annotation for each word xj by concatenating the forward hidden state
and the backward one from the encoder.
In this way, the annotation hj contains the summaries of both the preceding
words and the following words
-->

- __NOTE__ if you are interesting in this kind of mechanism check how the flair
embeddings generate from character embeddings en embedding for a word, there's
a similar idea there.

The context vector $c_{i}$ is computed as a weighted sum of these
annotations $h_{i}$:


$$c_{i} = \sum_{j=1}^{T_{x}} \alpha_{ij}h_{j}$$


The weight $\alpha_{ij}$ of each annotation $h_{j}$ is computed by:

$$\alpha_{ij} = \text{softmax}(e_{ij}) \ \ \ \ \text{where} \ \ \ \  e_{ij} = a(s_{i-1,h_{j}})$$

$a$ is an alignment model which scores how well the inputs around position $j$ and
the output at position $i$ match. The score is based on the RNN hidden state $s_{i−1}$
(just before emitting $y_{i}$) and the $j_{th}$ annotation $h_{j}$ of the input sentence

$$a(s_{i-1},h_{j}) = \mathbf{v}_a^\top \tanh(\mathbf{W}_{a}\ s_{i-1} + \mathbf{U}_{a}\ {h}_j)$$

where both $\mathbf{v}_a$ and $\mathbf{W}_a$ are weight matrices to be learned in the alignment model.

<!--
Instead of $s_{t}$ (the hidden state at time t), the new model uses now $s_{i}$,

So this new proposal - __attention__ - uses a context vector to attend to all words
in the source sentence.
-->


<!--
Other attention mechanisms/functions

"Effective Approaches to Attention-based Neural Machine Translation"
https://www.aclweb.org/anthology/D15-1166/
-->





---

## __Seq2Sequence Example__

<!--

ToDo: usar o exemplo e fazer uma tradução com português?

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

-->

## __Document-Level Classification Example__

<!--
Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
https://www.aclweb.org/anthology/P16-2034.pdf

Attention-based LSTM for Aspect-level Sentiment Classification
https://www.aclweb.org/anthology/D16-1058.pdf


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