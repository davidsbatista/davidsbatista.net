---
layout: post
title: Language Models and Contextualised Word Embeddings
date: 2018-12-06 00:00:00
tags: word-embeddings word2vec fasttext glove ELMo BERT language-models character-embeddings character-language-models
categories: [blog]
#comments: true
#disqus_identifier: 20181206
preview_pic: /assets/images/2018-12-06-word-embeddings.jpg
---

Since the work of [Mikolov et al., 2013](https://arxiv.org/pdf/1301.3781.pdf) was published and the software package _word2vec_ was made public available a new era in NLP started on which word embeddings, also referred to as word vectors, play a crucial role. Word embeddings can capture many different properties of a word and become the de-facto standard to replace feature engineering in NLP tasks.

Since that milestone many new embeddings methods were proposed some which go down to the character level, and others that take into consideration even language models. I will try in this blog post to review some of these methods, but focusing on the most recent word embeddings which are based on language models and take into consideration the context of a word.

## __Introduction__

This blog post consists of two parts, the first one, which is mainly pointers, simply refers to the __classic word embeddings__ techniques, which can also be seen as classic word embeddings, they can also be seen as _static word embeddings_ since the same word will always have the same representation regardless of the context where it occurs. I quickly introduce three embeddings techniques:

- __Skip-Gram (aka Word2Vec)__
- __Glove__
- __fastText__

The second part, introduces 3 news word embeddings techniques that take into consideration the context of the word, and can be seen as __dynamic word embeddings__ techniques, most of these techniques make use of some language model to help modeling the representation of a word. I try to describe three contextual embeddings techniques:

- __ELMO__
- __FlairEmbeddings__
- __BRET__

## __Classic Word Embeddings__

#### [Efficient Estimation of Word Representations in Vector Space (2013)](https://arxiv.org/pdf/1301.3781.pdf)

Introduced by (Mikolov et al., 2013) was the first popular embeddings method for NLP tasks. The paper itself is hard to understand, and many details are left over, but essentially the model is a neural network with a single hidden layer, and the embeddings are actually the weights of the hidden layer in the neural network.

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-12-06-skip_gram_net_arch.png">
  <figcaption><br>Image taken from http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/</figcaption>
</figure>


An important aspect is how to train this network in an efficient way, and then is when negative sampling comes into play.

I will not go into detail regarding this one, as the number of tutorials, implementations and resources regarding this technique is abundant in the net, and I will just rather leave some pointers.

#### __Links__
- [McCormick, C. (2016, April 19). Word2Vec Tutorial - The Skip-Gram Model.](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [McCormick, C. (2017, January 11). Word2Vec Tutorial Part 2 - Negative Sampling.](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
- [word2vec Parameter Learning Explained, Xin Rong](https://arxiv.org/pdf/1411.2738.pdf)
- [https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/)
- [Stanford NLP with Deep Learning: Lecture 2 - Word Vector Representations: word2vec](https://www.youtube.com/watch?v=ERibwqs9p38)

---
<br>


### [GloVe: Global Vectors for Word Representation (2014)](https://www.aclweb.org/anthology/D14-1162)

I will also give a brief overview of this work since there is also abundant resources on-line. It was published shortly after the _skip-gram_ technique and essentially it starts to make an observation that shallow window-based methods suffer from the disadvantage that they do not operate directly on the co-occurrence statistics of the corpus. Window-based models, like skip-gram, scan context windows across the entire corpus and fail to take advantage of the vast amount of repetition in the data.

Count models, like GloVe, learn the vectors by essentially doing some sort of dimensionality reduction on the co-occurrence counts matrix. They start by constructing a matrix with counts of word co-occurrence information, each row tells how often does a word occur with every other word in some defined context-size in a large corpus. This matrix is then factorize, resulting in a lower dimension matrix, where each row is some vector representation for each word.

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-12-06-glove-matrix-factorisation-5.jpg">
  <figcaption><br>Image taken from http://building-babylon.net/2015/07/29/glove-global-vectors-for-word-representations/</figcaption>
</figure>

The dimensionality reduction is typically done by minimizing a some kind of 'reconstruction loss' that finds lower-dimension representations of the original matrix and which can explain most of the variance in the original high-dimensional matrix.

#### __Links__
- [GloVe project at Stanford](https://nlp.stanford.edu/projects/glove/)
- [Building Babylon: Global Vectors for Word Representations](http://building-babylon.net/2015/07/29/glove-global-vectors-for-word-representations/)
- [Good summarization on text2vec.org](http://text2vec.org/glove.html)
- [Stanford NLP with Deep Learning: Lecture 3 GloVe - Global Vectors for Word Representation](https://www.youtube.com/watch?v=ASn7ExxLZws)
- [Paper Dissected: 'Glove: Global Vectors for Word Representation' Explained](http://mlexplained.com/2018/04/29/paper-dissected-glove-global-vectors-for-word-representation-explained)

---
<br>



### [Enriching Word Vectors with Subword Information (2017)](http://aclweb.org/anthology/Q17-1010)

One drawback of the two approaches presented before is the fact that they don't handle out-of-vocabulary.

The work of [Bojanowski et al, 2017](http://aclweb.org/anthology/Q17-1010) introduced the concept of subword-level embeddings, based on the skip-gram model, but where each word is represented as a bag of character $$n$$-grams.

<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2018-12-06-fasttext-logo-color-web.png">
  <figcaption><br>Image taken from https://fasttext.cc/</figcaption>
</figure>

A vector representation is associated to each character $$n$$-gram, and words are represented as the sum of these representations. This allows the model to compute word representations for words that did not appear in the training data.

Each word $w$ is represented as a bag of character $n$-gram, plus a special boundary symbols _\<_ and _\>_ at the beginning and end of words, plus the word $w$ itself in the set of its $n$-grams.

Taking the word _where_ and $n = 3$ as an example, it will be represented by the character $n$-grams: \< wh, whe, her, ere, re \> and the special sequence \< where \>.

#### __Links__
- [https://github.com/facebookresearch/fastText](https://github.com/facebookresearch/fastText)
- [Library for efficient text classification and representation learning](https://fasttext.cc/)

---
<br>


#### __Static Word Embeddings fail to capture polysemy__

The models presented before have a fundamental problem which is they generate the same embedding for the same word in different contexts, for example, given the word _bank_ although it will have the same representation it can have different meanings:

- "I deposited 100 EUR in the __bank__."
- "She was enjoying the sunset o the left __bank__ of the river."

In the methods presented before, the word representation for __bank__ would always be the same regardless if it appears in the context of geography or economics. In the next part of the post we will see how new embedding techniques capture polysemy.

---

<br>

## __Contextualised Word-Embeddings__

Contextualised words embeddings aim at capturing word semantics in different contexts to address the issue of polysemous and the context-dependent nature of words.

#### __Language Models__

Language models compute the probability distribution of the next word in a sequence given the sequence of previous words. LSTMs become a popular neural network architecture to learn this probabilities. The figure below shows how an LSTM can be trained to learn a language model.

<figure>
  <img style="width: 65%; height: 65%" src="/assets/images/2018-12-06-general_word_language_model.jpg">
  <figcaption><br> Image taken from http://torch.ch/blog/2016/07/25/nce.html</figcaption>
</figure>

A sequence of words is fed into an LSTM word by word, the previous word along with the internal state of the LSTM are used to predict the next possible word.

But it's also possible to go one level below and build a character-level language model. [Andrej Karpathy blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) about char-level language model shows some interesting examples.

This is short a very short intro on language models, but they are the backbone of the upcoming techniques/papers that complete this blog post.

---

<br>


### [__ELMo: Deep contextualized word representations (2018)__](https://aclweb.org/anthology/N18-1202)

The main idea of the Embeddings from Language Models (ELMo) can be divided into two main tasks, first we train an LSTM-based language model on some corpus, and then we use the hidden states of the LSTM for each token to generate a vector representation of each word.

#### __Language Model__

The language model is trained by reading the sentences both forward and backward. That is, in essence there are two language models, one that learns to predict the next word given the past words and another that learns to predict the past words given the future words.

Another detail is that the authors, instead of using a single-layer LSTM use a stacked multi-layer LSTM. A single-layer LSTM takes the sequence of words as input, a multi-layer LSTM takes the output sequence of the previous LSTM-layer as input, the authors also mention the use of residual connections between the LSTM layers. In the paper the authors also show that the different layers of the LSTM language model learns different characteristics of language.

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-12-06-ELMo_language_models.png">
  <figcaption><br>Image taken from Shuntaro Yada slides</figcaption>
</figure>

Training $L$-layer LSTM forward and backward language mode generates $$2\ \times \ L$$ different vector representations for each word, $L$ represents the number of stacked LSTMs, each one outputs a vector.

Adding another vector representation of the word, trained on some external resources, or just a random embedding, we end up with $$2\ \times \ L + 1$$ vectors that can be used to compute the context representation of every word.

The parameters for the token representations and the softmax layer are shared by the forward and backward language model, while the LSTMs parameters (hidden state, gate, memory) are separate.

ELMo is a task specific combination of the intermediate layer representations in a bidirectional Language Model (biLM). That is, given a pre-trained biLM and a supervised architecture for a target NLP task, the __end task model learns a linear combination of the layer representations__.

The language model described above is completely task-agnostic, and is trained in an unsupervised manner.

#### __Task-specific word representation__

The second part of the model consists in using the hidden states generated by the LSTM for each token to compute a vector representation of each word, the detail here is that this is done in a specific context, with a given end task.

Concretely, in ELMo, each word representation is computed with a concatenation and a weighted sum:

$$ ELMo_k = \gamma^{task} \sum_{j=0}^{L} s_j^{task} h_{k,j}^{LM} $$

- the scalar parameter $$\gamma^{task}$$ allows the task model to scale the entire ELMo vector
- $$s_j^{task}$$ are softmax-normalized weights
- the indices $$k$$ and $$j$$ correspond to the index of the word and the index of the layer from which the hidden state is being extracted from.

For example, $$h_{k,j}$$ is the output of the $$j$$-th LSTM for the word $$k$$, $$s_j$$ is the weight of $$h_{k,j}$$ in computing the representation for $$k$$.

<figure>
  <img style="width: 85%; height: 85%" src="/assets/images/2018-12-06-ELMo_task_specific.png">
  <figcaption><br>Image taken from Shuntaro Yada slides</figcaption>
</figure>

In practice ELMo embeddings could replace existing word embeddings, the authors however recommend to concatenate ELMos with context-independent word embeddings such as GloVe or fastText before inputting them into the task-specific model.

ELMo is flexible in the sense that it can be used with any model barely changing it, meaning it can work with existing systems or architectures.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2018-12-06-ELMo_overview.png">
  <figcaption><br>Image taken from Shuntaro Yada slides</figcaption>
</figure>

In resume, ELMos train a multi-layer, bi-directional, LSTM-based language model, and extract the hidden state of each layer for the input sequence of words. Then, they compute a weighted sum of those hidden states to obtain an embedding for each word. The weight of each hidden state is task-dependent and is learned during training of the end-task.

#### __Links__

- [ELMo code at AllenNLP github](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)
- [AllenNLP Models](https://allennlp.org/models)
- [Video of the presentation of paper by Matthew Peters @ NAACL-HLT 2018](https://vimeo.com/277672840)
- Images were taken/adapted from [Shuntaro Yada](https://shuntaroy.com/) excellent [slides](https://www.slideshare.net/shuntaroy/a-review-of-deep-contextualized-word-representations-peters-2018)

---

<br>


### [__Contextual String Embeddings for Sequence Labelling__ (2018)](https://aclweb.org/anthology/C18-1139)

The authors propose a contextualized character-level word embedding which captures word meaning in context and therefore produce different embeddings for polysemous words depending on their context. It model words and context as sequences of characters, which aids in handling rare and misspelled words and captures subword structures such as prefixes and endings.


#### __Character-level Language Model__

Characters are the atomic units of language model, allowing text to be treated as a sequence of characters passed to an LSTM which at each point in the sequence is trained to predict the next character.

The authors train a forward and a backward model character language model. Essentially the character-level language model is just 'tuning' the hidden states of the LSTM based on reading lots of sequences of characters.

The LSTM internal states will try to capture the probability distribution of characters given the previous characters (i.e., forward language model) and the upcoming characters (i.e., backward language model).


#### __Extracting Word Representations__

From this forward-backward LM, the authors concatenate the following hidden character states for each word:

- _from the fLM, we extract the output hidden state after the last character in the word. Since the fLM is trained to predict likely continuations of the sentence after this character, the hidden state encodes semantic-syntactic information of the sentence up to this point, including the word itself._

- _from the bLM, we extract the output hidden state before the word’s first character from the bLM to capture semantic-syntactic information from the end of the sentence to this character._

Both output hidden states are concatenated to form the final embedding and capture the semantic-syntactic information of the word itself as well as its surrounding context.

The image below illustrates how the embedding for the word _Washington_ is generated, based on both character-level language models.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2018-12-06-character_lm_for_word_embedding.png">
  <figcaption><br> Image taken from "Contextual String Embeddings for Sequence Labelling (2018)"</figcaption>
</figure>



The embeddings can then be used for other downstream tasks such as named-entity recognition. The embeddings generated from the character-level language models can also (and are in practice) concatenated with word embeddings such as GloVe or fastText.

<figure>
  <img style="width: 70%; height: 70%" src="/assets/images/2018-12-06-character_lm_for_sequence_labelling.png">
  <figcaption><br> Image taken from "Contextual String Embeddings for Sequence Labelling (2018)"</figcaption>
</figure>


In essence, this model first learns two character-based language models (i.e., forward and backward) using LSTMs. Then, an embedding for a given word is computed by feeding a word - character by character - into each of the language-models and keeping the two last states (i.e., last character and first character) as two word vectors, these are then concatenated.

In the experiments described on the paper the authors concatenated the word vector generated before with yet another word vector from fastText an then apply a [Neural NER architecture](../../../../../blog/2018/10/22/Neural-NER-Systems/) for several sequence labelling tasks, e.g.: NER, chunking, PoS-tagging.

#### __Links__

- [https://github.com/zalandoresearch/flair](https://github.com/zalandoresearch/flair)
- [Slides from Berlin Machine Learning Meetup](http://alanakbik.github.io/talks/ML_Meetup_2018.pdf)

<br>

---




<br>

### [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/pdf/1810.04805.pdf)

BERT, or Bidirectional Encoder Representations from Transformers, is essentially a new method of training language models.

Pre-trained word representations, as seen in this blog post, can be __context-free__ (i.e., word2vec, GloVe, fastText), meaning that a single word representation is generated for each word in the vocabulary, or can also be __contextual__ (i.e., ELMo and Flair), on which the word representation depends on the context where that word occurs, meaning that the same word in different contexts can have different representations.

Contextual representations can further be __unidirectional__ or __bidirectional__. Note, even if a language model is trained forward or backward, is still considered unidirectional since the prediction of future words (or characters) is only based on past seen data.

In the sentence: _"The cat sits on the mat"_, the unidirectional representation of _"sits"_ is only based on _"The cat"_ but not on _"on the mat"_. Previous works train two representations for each word (or character), one left-to-right and one right-to-left, and then concatenate them together to a have a single representation for whatever downstream task.

BERT represents _"sits"_ using both its left and right context — _"The cat xxx on the mat"_ based on a simple approach, masking out 15% of the words in the input, run the entire sequence through a multi-layer bidirectional Transformer encoder, and then predict only the masked words.

### __Multi-layer bidirectional Transformer encoder__

The Multi-layer bidirectional Transformer aka Transformer was first introduced in the [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) paper. It follows the encoder-decoder architecture of machine translation models, but  it replaces the RNNs by a different network architecture.

The Transformer tries to learn the dependencies, typically encoded by the hidden states of a RNN, using just an Attention Mechanism. RNNs handle dependencies by being stateful, i.e., the current state encodes the information they needed to decide on how to process subsequent tokens.

This means that RNNs need to keep the state while processing all the words, and this becomes a problem for long-range dependencies between words. The attention mechanism has somehow mitigated this problem but it still remains an obstacle to high-performance machine translation.

The Transformer tries to directly learn these dependencies using the attention mechanism only and it also learns intra-dependencies between the input tokens, and between output tokens. This is done by relying on a key component, the __Multi-Head Attention block__, which has an attention mechanism defined by the authors as the __Scaled Dot-Product Attention__.

<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2018-12-06-transformer_attention.png">
  <figcaption><br> Image taken from "Attention Is All You Need"</figcaption>
</figure>

To improve the expressiveness of the model, instead of computing a single attention pass over the values, the Multi-Head Attention computes multiple attention weighted sums, i.e., it uses several attention layers stacked together with different linear transformations of the same input.

<figure>
  <img style="width: 35%; height: 35%" src="/assets/images/2018-12-06-attention_path_length.png">
  <figcaption><br> Image taken from http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/</figcaption>
</figure>

The main key feature of the Transformer is therefore that instead of encoding dependencies in the hidden state, directly expresses them by attending to various parts of the input.

The Transformer in an encoder and a decoder scenario

<figure>
  <img style="width: 40%; height: 40%" src="/assets/images/2018-12-06-transformer_encoding_decoding_arch.png">
  <figcaption><br> Image taken from "Attention Is All You Need"</figcaption>
</figure>

This is just a very brief explanation of what the Transformer is, please check the original paper and following links for a more detailed description:

- [http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/](http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)
- [https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
- [http://nlp.seas.harvard.edu/2018/04/03/attention.html](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)


## __Masked Language Model__

BERT uses the Transformer encoder to learn a language model.

The input to the Transformer is a sequence of tokens, which are passed to an embeddeding layer and then processed by the Transformer network. The output is a sequence of vectors, in which each vector corresponds to an input token.


<figure>
  <img style="width: 65%; height: 65%" src="/assets/images/2018-12-06-transformer_arch.png">
  <figcaption><br> Image taken from https://www.lyrn.ai/2018/11/07/explained-bert-state-of-the-art-language-model-for-nlp/</figcaption>
</figure>

As explained above this language model is what one could considered a bi-directional model, but some defend that you should be instead called non-directional.

The bi-directional/non-directional property in BERT comes from masking 15% of the words in a sentence, and forcing the model to learn how to use information from the entire sentence to deduce what words are missing.

The original Transformer is adapted so that the loss function only considers the prediction of masked words and ignores the prediction of the non-masked words. The prediction of the output words requires:

- Adding a classification layer on top of the encoder output.
- An embedding matrix, transforming the output vectors into the vocabulary dimension.
- Calculating the probability of each word in the vocabulary with softmax.


<figure>
  <img style="width: 65%; height: 65%" src="/assets/images/2018-12-06-transformer_mll.png">
  <figcaption><br> Image taken from https://www.lyrn.ai/2018/11/07/explained-bert-state-of-the-art-language-model-for-nlp/</figcaption>
</figure>

BRET is also trained in a Next Sentence Prediction (NSP), in which the model receives pairs of sentences as input and has to learn to predict if the second sentence in the pair is the subsequent sentence in the original document or not.

To use BERT for a sequence labelling task, for instance a NER model, this model can be trained by feeding the output vector of each token into a classification layer that predicts the NER label.

#### __Links__

- [Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
- [https://github.com/google-research/bert](https://github.com/google-research/bert)
- [BERT – State of the Art Language Model for NLP (www.lyrn.ai)](https://www.lyrn.ai/2018/11/07/explained-bert-state-of-the-art-language-model-for-nlp/)
- [Reddit: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/)
- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/)

---




<br>

## __Summary__

In a time span of about 10 years Word Embeddings revolutionized the way almost all NLP tasks can be solved, essentially by replacing the feature extraction/engineering by embeddings which are then feed as input to different neural networks architectures.

The most popular models started around 2013 with the word2vec package, but a few years before there were already some results in the famous work of Collobert et, al 2011 [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf) which I did not mentioned above.

Nevertheless these techniques, along with GloVe and fastText, generate static embeddings which are unable to capture polysemy, i.e the same word having different meanings. Typically these techniques generate a matrix that can be plugged in into the current neural network model and is used to perform a look up operation, mapping a word to a vector.

Recently other methods which rely on language models and also provide a mechanism of having embeddings computed dynamically as a sentence or a sequence of tokens is being processed.

<br>

## __References__

- [Efficient Estimation of Word Representations in Vector Space (2013)](https://arxiv.org/pdf/1301.3781.pdf)
- [McCormick, C. (2016, April 19). Word2Vec Tutorial - The Skip-Gram Model.](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [McCormick, C. (2017, January 11). Word2Vec Tutorial Part 2 - Negative Sampling.](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
- [word2vec Parameter Learning Explained, Xin Rong](https://arxiv.org/pdf/1411.2738.pdf)
- [GloVe: Global Vectors for Word Representation (2014)](https://www.aclweb.org/anthology/D14-1162)
- [Enriching Word Vectors with Subword Information (2017)](http://aclweb.org/anthology/Q17-1010)
- [ELMo: Deep contextualized word representations (2018)__](https://aclweb.org/anthology/N18-1202)
- [Contextual String Embeddings for Sequence Labelling__ (2018)](https://aclweb.org/anthology/C18-1139)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/pdf/1810.04805.pdf)
- [https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
- [http://nlp.seas.harvard.edu/2018/04/03/attention.html](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
- [Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)