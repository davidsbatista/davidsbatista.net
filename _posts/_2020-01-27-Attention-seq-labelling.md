---
layout: post
title: The Attention Mechanism in Natural Language Processing - seq2seq
date: 2020-01-25 00:00:00
tags: attention nlp seq2seq
categories: [blog]
comments: false
disqus_identifier: 2020125
preview_pic: /assets/images/2020-01-25-seq2seq_with_attention.png
---

The __Attention__ mechanism is now an established


## __Introduction__

To the best of my knowledge the attention mechanism was first presented in
_"Neural Machine Translation by Jointly Learning to Align and Translate"_
at ICLR 2015 ([Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf)).

This was proposed in the context of machine translation, where given a sentence
in one language, the model has to produce a translation for that sentence in
another language.

In the paper, the authors propose to tackle the problem of a fixed-length context
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

At every time-step $t$ the encoder produces a hidden state $h_{t}$, and the
generated context vector is modelled according to all hidden states.



The __decoder__ is trained to predict the next word $$y_{t}$$ given the context
vector $$c$$ and all the previously predict words $\\{y_{1}, \dots , y_{t-1}\\}$,
it defines a probability over the translation $${\bf y}$$ by decomposing
the joint probability:

$$p({\bf y}) = \prod\limits_{i=1}^{x} p(y_{t} | {y_{1}, \dots , y_{t-1}}, c)$$

where $\bf y = \\{y_{1}, \dots , y_{t}\\}$. In other words, the probability of a
translation sequence is calculated by computing the conditional probability
of each word given the previous words. With an LSTM/RNN each conditional probability
is computed as:

$$p(y_{t} | {y_{1}, \dots , y_{t-1}}, c) = g(y_{t−1}, s_{t}, c)$$

where, $g$ is a nonlinear function that outputs the probability of $y_{t}$,
$s_{t}$ is the value of the hidden state of the current position, and $c$ the
context vector.

In a simple seq2seq model, the last output of the LSTM/RNN is the context vector,
encoding context from the entire sequence. This context vector is used as the
initial hidden state of the decoder.

At every step of decoding, the decoder is given an input token and (the previous)
hidden state. The initial input token is the start-of-string <SOS> token, and
the first hidden state is the context vector (the encoder’s last hidden state).

So, the fixed size context-vector needs to contain a good summary of the meaning
of the whole source sentence, being this one big bottleneck, specially for long
sentences.




### __seq2seq with Attention__

The fixed size context-vector bottleneck was one of the main motivations by
[Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf), which proposed a
similar architecture but with a crucial improvement:

"_The new architecture consists of a bidirectional RNN as an encoder and a
decoder that emulates searching through a source sentence during decoding
a translation_"

The encoder is now a bidirectional recurrent network with a forward and backward
hidden states. A simple concatenation of the two represents the encoder state
at any given position in the sentence. The motivation is to include both the
preceding and following words in the annotation of one word.

The other key element, and the most important on, is that the decoder is now
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
  <figcaption>Figure 5: </figcaption>
</figure>

NOTE: the _concat_ is the same as in [Bahdanau et al. 2015](https://arxiv.org/pdf/1409.0473.pdf).
But, most importantly, instead of a weighted average over all the source hidden
states, they proposed a mechanism of local attention which focus only on a small
subset of the source positions per target word instead of attending to all words
on the source for each target word.

---

### __Seq2Sequence Example__

<!--
ToDo: usar o exemplo e fazer uma tradução com português?
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
-->






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

## hierarchical Attention Networks
















## __Attention in Sequence Labelling Example__

<!--


https://arxiv.org/pdf/1810.13097.pdf

https://www.aclweb.org/anthology/C16-1030.pdf

https://www.groundai.com/project/a-survey-on-deep-learning-for-named-entity-recognition/1

https://chywang.github.io/papers/apweb2018.pdf

https://ieeexplore.ieee.org/abstract/document/8372007
https://booksc.xyz/book/70658753/270c16

-->























## __References__

- [1] [An Introductory Survey on Attention Mechanisms in NLP Problems](https://link.springer.com/chapter/10.1007/978-3-030-29513-4_31) ([arXiv.org version](https://arxiv.org/pdf/1811.05544.pdf))

- [2] [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) ([slides](https://www.iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2015:bahdanau-iclr2015.pdf))

- [3] [Effective Approaches to Attention-based Neural Machine Translation](https://www.aclweb.org/anthology/D15-1166/) ([slides](https://nlp.stanford.edu/~lmthang/data/papers/emnlp15_attn.pptx))

- ["Attention, Attention" in Lil'Log](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

- __Figure 1__ taken from https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)

- __Figure 2__ taken from https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)

- __Figure 3__ taken from https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)

- __Figure 4 and 5__ taken from Nelson Zhao [blog's post](https://zhuanlan.zhihu.com/p/37290775)

<!--
https://ruder.io/deep-learning-nlp-best-practices/index.html#attention

self-attention, two-way attention, key-value-predict models and hierarchical attention
https://medium.com/@joealato/attention-in-nlp-734c6fa9d983
-->



