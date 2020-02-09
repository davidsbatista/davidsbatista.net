---
layout: post
title: The Attention Mechanism in Natural Language Processing - seq2seq
date: 2020-01-25 00:00:00
tags: attention nlp seq2seq machine-translation neural-networks recurrent-neural-networks LSTM GRU RNN
categories: [blog]
comments: false
disqus_identifier: 2020125
preview_pic: /assets/images/2020-01-25-seq2seq_with_attention.png
---

The __Attention__ mechanism is now an established technique in many NLP tasks.
I've heard about it often, but wanted to go a bit more deep and understand
the details. In this first blog post - since I plan to publish a few more blog posts
regarding the __attention__ subject - I make an introduction by focusing in the
first proposal of attention mechanism, as applied to the task of neural machine
translation.


## __Introduction__

To the best of my knowledge the attention mechanism within the context of NLP
was first presented in _"Neural Machine Translation by Jointly Learning to Align
and Translate"_ at ICLR 2015 ([Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf)).

This was proposed in the context of machine translation, where given a sentence
in one language, the model has to produce a translation for that sentence in
another language.

In the paper, the authors propose to tackle the problem of a fixed-length context
vector in the original __seq2seq__ model for machine translation
([Cho et al., 2014](https://www.aclweb.org/anthology/D14-1179/))

### __The _classical_ Sequence-to-Sequence model__

The model is composed of two main components: an encoder, and a decoder.

<br>

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2020-01-25-seq2seq.jpeg">
  <figcaption>Figure 1: A seq2seq model composed of an encoder and decoder.</figcaption>
</figure>

<br>

The __encoder__ reads the input sentence, a sequence of vectors $x = (x_{1}, \dots , x_{T})$,
into a fixed-length vector $c$. The __encoder__ is a recurrent neural network,
typical approaches are GRU or LSTMs such that:

$$h_{t} = f\ (x_{t}, h_{t−1})$$

$$ c = q\ (h_{1}, \dotsc, h_{T}) $$

where $h_{t}$ is a hidden state at time $t$, and $c$ is a vector generated from
the sequence of the hidden states, and $f$ and $q$ are some nonlinear functions.

At every time-step $t$ the encoder produces a hidden state $h_{t}$, and the
generated context vector is modelled according to all hidden states.

<br>

The __decoder__ is trained to predict the next word $$y_{t}$$ given the context
vector $$c$$ and all the previously predict words $\\{y_{1}, \dots , y_{t-1}\\}$,
it defines a probability over the translation $${\bf y}$$ by decomposing
the joint probability:

$$p({\bf y}) = \prod\limits_{i=1}^{x} p(y_{t} | {y_{1}, \dots , y_{t-1}}, c)$$

where $\bf y = \\{y_{1}, \dots , y_{t}\\}$. In other words, the probability of a
translation sequence is calculated by computing the conditional probability
of each word given the previous words. With an LSTM/GRU each conditional
probability is computed as:

$$p(y_{t} | {y_{1}, \dots , y_{t-1}}, c) = g(y_{t−1}, s_{t}, c)$$

where, $g$ is a nonlinear function that outputs the probability of $y_{t}$,
$s_{t}$ is the value of the hidden state of the current position, and $c$ the
context vector.

In a simple __seq2seq__ model, the last output of the LSTM/GRU is the context
vector, encoding context from the entire sequence. This context vector is then
used as the initial hidden state of the decoder.

At every step of decoding, the decoder is given an input token and (the previous)
hidden state. The initial input token is the start-of-string <SOS> token, and
the first hidden state is the context vector (the encoder’s last hidden state).

So, the fixed size context-vector needs to contain a good summary of the meaning
of the whole source sentence, being this one big bottleneck, specially for long
sentences.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2020-01-25-seq2seq_long_sentences.png">
  <figcaption>Figure 2: A seq2seq model performance by sentence length.</figcaption>
</figure>

<br>


### __Sequence-to-Sequence model with Attention__

The fixed size context-vector bottleneck was one of the main motivations by
[Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf), which proposed a
similar architecture but with a crucial improvement:

"_The new architecture consists of a bidirectional RNN as an encoder and a
decoder that emulates searching through a source sentence during decoding
a translation_"

The encoder is now a bidirectional recurrent network with a forward and backward
hidden states. A simple concatenation of the two hidden states represents the
encoder state at any given position in the sentence. The motivation is to
include both the preceding and following words in the representation/annotation
of an input word.

The other key element, and the most important one, is that the decoder is now
equipped with some sort of search, allowing it to look at the whole source
sentence when it needs to produce an output word, the __attention mechanism__.

<figure>
  <img style="width: 40%; height: 40%" src="/assets/images/2020-01-25-seq2seq_with_attention.png">
  <figcaption>Figure 2: The attention mechanism in a seq2seq model. Taken from Bahdanau et al. 2015.</figcaption>
</figure>

The Figure 2 above gives a good overview of this new mechanism. To produce the
output word at time $y_{t}$ the decoder uses the last hidden state from the
decoder - one can thing about this as some sort of representation of the already
produced words - and a dynamically computed context vector based on the input
sequence.

The authors proposed to replace the fixed-length context vector by a another
context vector $c_{i}$ which is a sum of the hidden states of the input sequence,
weighted by alignment scores.

Note that now the probability of each output word is conditioned on a distinct
context vector $c_{i}$ for each target word $y$.

The new decoder is then defined as:

$$p(y_{t} | {y_{1}, \dots , y_{t-1}}, c) = g(y_{t−1}, s_{i}, c)$$

where $s_{i}$ is the hidden state for time $i$, computed by:

$$s_{i} = f(s_{i−1}, y_{i−1}, c_{i})$$

that is, a new hidden state for $i$ depends on the previous hidden state,
the representation of the word generated by the previous state and the
context vector for position $i$. The remaining question now is, how to compute
the context vector $$c_{i}$$ ?


### __Context Vector__

The context vector $c_{i}$ is a sum of the hidden states of the input sequence,
weighted by alignment scores. Each word in the input sequence is represented by
a concatenation of the two (i.e., forward and backward) RNNs hidden states,
let's call them annotations.

Each annotation contains information about the whole input sequence with a
strong focus on the parts surrounding the $$i_{th}$$ word in the input sequence.

The context vector $c_{i}$ is computed as a weighted sum of these annotations:

$$c_{i} = \sum_{j=1}^{T_{x}} \alpha_{ij}h_{j}$$

The weight $\alpha_{ij}$ of each annotation $h_{j}$ is computed by:

$$\alpha_{ij} = \text{softmax}(e_{ij})$$

where:

$$e_{ij} = a(s_{i-1,h_{j}})$$

$a$ is an alignment model which scores how well the inputs around position $j$ and
the output at position $i$ match. The score is based on the RNN hidden state $s_{i−1}$
(just before emitting $y_{i}$) and the $j_{th}$ annotation $h_{j}$ of the input sentence

$$a(s_{i-1},h_{j}) = \mathbf{v}_a^\top \tanh(\mathbf{W}_{a}\ s_{i-1} + \mathbf{U}_{a}\ {h}_j)$$

where both $\mathbf{v}_a$ and $\mathbf{W}_a$ are weight matrices to be learned in the alignment model.


The alignment model in the paper is described as feed forward neural network
whose weight matrices $\mathbf{v}_a$ and $\mathbf{W}_a$ are learned jointly
together with the whole graph/network.

The authors note:

"_The probability_ $\alpha_{ij}h_{j}$ _reflects the importance of the annotation_
$h_{j}$ _with respect to the previous hidden state_ $s_{i−1}$ _in deciding the
next state_ $s_{i}$ _and generating_ $y_{i}$. _Intuitively, this implements a
mechanism of attention in the decoder._

### __Resume__

It's now useful to review again visually the attention mechanism and compare it
against the fixed-length context vector. The pictures below were made by
[Nelson Zhao](https://github.com/NELSONZHAO) and hopefully will help understand
clearly the difference between the two encoder-decoder approaches.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2020-01-25-attention_seq2seq_context.jpg">
  <figcaption>Figure 3: Encoder-Decoder with fixed-context vector <br> (https://zhuanlan.zhihu.com/p/37290775)</figcaption>
</figure>


<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2020-01-25-attention_seq2seq_context_with_attention.jpg">
  <figcaption>Figure 4: Ecnoder-Decoder with attention mechanism <br> (https://zhuanlan.zhihu.com/p/37290775)</figcaption>
</figure>


### __Extensions to the classical attention mechanism__

[Luong et al.](https://www.aclweb.org/anthology/D15-1166/) proposed and compared
other mechanisms of attentions, more specifically, alternative functions to
compute the alignment score:


<figure>
  <img style="width: 55%; height: 55%" src="/assets/images/2020-01-25-alignment_scores.png">
  <figcaption>Figure 5:Alternative alignment scoring functions.</figcaption>
</figure>

NOTE: the _concat_ is the same as in [Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf).
But, most importantly, instead of a weighted average over all the source hidden
states, they proposed a mechanism of local attention which focus only on a small
subset of the source positions per target word instead of attending to all words
on the source for each target word.

## __Summary__

This was a short introduction on the first "classical" attention mechanism, in
the meantime others were published, such as __self-attention__ or
__key-value-attention__, which I plan to write about in the future.

The __attention__ mechanism was then applied to other natural language processing
tasks based on neural networks such as RNN/CNN, such as document-level
classification or sequence labelling, which I plan to write a bit more about in
forthcoming blog posts.


<br>

## __References__

- [1] [An Introductory Survey on Attention Mechanisms in NLP Problems](https://link.springer.com/chapter/10.1007/978-3-030-29513-4_31) ([arXiv.org version](https://arxiv.org/pdf/1811.05544.pdf))

- [2] [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) ([slides](https://www.iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2015:bahdanau-iclr2015.pdf))

- [3] [Effective Approaches to Attention-based Neural Machine Translation](https://www.aclweb.org/anthology/D15-1166/) ([slides](https://nlp.stanford.edu/~lmthang/data/papers/emnlp15_attn.pptx))

- ["Attention, Attention" in Lil'Log](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

- __Figure 1__ taken from [towardsdatascience](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)

- __Figure 2 and 3__ taken from [Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf)

- __Figure 4 and 5__ taken from Nelson Zhao [blog's post](https://zhuanlan.zhihu.com/p/37290775)

- __Figure 6__ taken from [Luong et al.](https://www.aclweb.org/anthology/D15-1166/)





