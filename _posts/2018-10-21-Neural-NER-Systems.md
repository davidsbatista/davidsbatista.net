---
layout: post
title: Named-Entity Recognition based on Neural Networks
date: 2018-10-22 00:0:00
categories: blog
tags: NER neural-networks sequence-prediction viterbi conditional-random-fields deep-learning
#comments: true
#disqus_identifier: 20181022
#preview_pic: /assets/images/2018-05-09-NER_metrics.jpeg
description: This blog post review some of the recent proposed methods to perform named-entity recognition using neural networks.
---

Recently (i.e., at the time of this writing I mean since 2015~2016 onwards) new methods to perform named-entity recognition (NER) based on Neural Networks start to be proposed/published, I will try in this blog post to do a quick recap of these new methods, understanding their architectures and pointing out what each technique brought new or different to the already knew methods.


## __Introduction__

Sequence tagging tasks (e.g.: part-of-speech tagging (POS), chunking, named entity recognition (NER)) has been a classic NLP task.

## __Linear Sequence Models__

- Independent assumptions with window features

- Hidden Markov Models

- Maximum Entropy Markov Models

- Conditional Random Fields

<!--

http://www.aclweb.org/anthology/W96-0213

The work of maximum entropy classifier (Ratnaparkhi, 1996) and Maximum en- tropy Markov models (MEMMs) (McCallum et al., 2000) fall in this category.
-->

<!--
Explicar resumidamente LSTMs neste contexto

a RNN introduces the connection between the previous hidden state and current hidden
state (and thus the recurrent layer weight parameters). This recurrent layer
is designed to store history information.


- An input layer:
  - has the same dimensionality as feature size.

- An output layer:
  - represents a probability distribution over labels at time t
  - It has the same dimensionality as size of labels.

Long Short- Term Memory networks are the same as RNNs, except that the hidden layer updates are replaced by purpose-built memory cells. As a result, they may be better at finding and exploiting long range dependencies in the data.

-->


## __Neural Sequence Labelling Models__

previous: - Conv-CRF, Collbert

["Bidirectional LSTM-CRF Models for Sequence Tagging" (2015)](https://arxiv.org/pdf/1508.01991v1.pdf)

first to apply a bidirectional LSTM CRF (denoted as BI-LSTM-CRF) model to NLP
benchmark sequence tagging data sets

In this paper, we propose a variety of neural network based models to sequence tagging task:
- CRF: lots of features, described in paper
- LSTM networks
- bidirectional LSTM networks (BI-LSTM)
- LSTM networks with a CRF layer (LSTM-CRF)
- bidirectional LSTM networks with a CRF layer (BI- LSTM-CRF).





We combine a LSTM network and a CRF network to form a LSTM-CRF model, which is shown in Fig. 6.

- no char-level Embeddings
- combined neural networks with hand-crafted features








---

["Named Entity Recognition with Bidirectional LSTM-CNNs" (2016)](https://www.aclweb.org/anthology/Q16-1026)

- do not uses CRF on top
- external knowledge: char-type, capitalization and lexical features, NER
  specific processing, replacing all sequence of digits 0-9 by 0

---


["Neural Architectures for Named Entity Recognition" (2016)](http://www.aclweb.org/anthology/N16-1030)

code:
* [https://github.com/Hironsan/anago](https://github.com/Hironsan/anago)
- [https://github.com/achernodub/bilstm-cnn-crf-tagger](https://github.com/achernodub/bilstm-cnn-crf-tagger)
- [https://github.com/glample/tagger](https://github.com/glample/tagger)

- use LSTM for char embeddings instead of CNN

---

["End-to-end Sequence Labelling via Bi-directional LSTM-CNNs-CRF" (2016)](http://www.aclweb.org/anthology/P16-1101)

code: [https://github.com/achernodub/bilstm-cnn-crf-tagger](https://github.com/achernodub/bilstm-cnn-crf-tagger)

1) For each word, char-level representation is computed by the CNN with char embeddings as input
2) Concatenate char-level representation of word with the word embedding (pre-trained)
3) Fed into a bi-LSTM
4) Fed output vectors of bi-LSTM to CRF layer


---


## Embeddings

"Deep contextualized word representations" (2018)
paper: http://aclweb.org/anthology/N18-1202
code:  https://github.com/Hironsan/anago


"Enriching Word Vectors with Subword Information"
paper: http://aclweb.org/anthology/Q17-1010
code:  https://github.com/facebookresearch/fastText


https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/