---
layout: post
title: Named-Entity Recognition based on Neural Networks
date: 2018-10-22 00:0:00
categories: blog
tags: conditional-random-fields NER neural-networks sequence-prediction viterbi LSTM
#comments: true
#disqus_identifier: 20181022
preview_pic: /assets/images/2018-10-21-CNN-Char-Embeddings.png
description: This blog post review some of the recent proposed methods to perform named-entity recognition using neural networks.
---

Recently (i.e., at the time of this writing since 2015~2016 onwards) new methods to perform sequence labelling tasks based on neural networks started to be proposed/published, I will try in this blog post to do a quick recap of these new methods, understanding their architectures and pointing out what each technique brought new or different to the already knew methods.


# __Introduction__

Several NLP tasks involve classifying a sequence tagging tasks has been a classic NLP task. A classical example is part-of-speech tagging, in this scenario, each $$x_{i}$$ describes a word and each $$y_{i}$$ the associated part-of-speech of the word $$x_{i}$$ (e.g.: _noun_, _verb_, _adjective_, etc.).

Another example, is named-entity recognition, in which, again, each $$x_{i}$$ describes a word and $$y_{i}$$ is a semantic label associated to that word (e.g.: _person_, _location_, _organization_, _event_, etc.).

# __Linear Sequence Models__

Classical approaches - i.e., prior to the neural networks revolution in NLP - to deal with these tasks involved methods which made
independent assumptions, that is, the tag decision for each word depends only on the surrounding words and not on previous classified words.

Then methods that take into consideration the sequence structure, i.e., the tag given to the previous classified word(s) is considered when deciding the tag to give to the following word.

You can read more about these last methods here:

* __[Hidden Markov Model and Naive Bayes relationship](../../../../../blog/2017/11/11/HHM_and_Naive_Bayes/)__

* __[Maximum Entropy Markov Models and Logistic Regression](../../../../../blog/2017/11/12/Maximum_Entropy_Markov_Model/)__

* __[Conditional Random Fields for Sequence Prediction](../../../../../blog/2017/11/13/Conditional_Random_Fields/)__

But recently, methods based on neural networks started succeed and are nowadays state-of-the-art in mostly NLP sequence prediction tasks.

Most of this methods combine not one simple neural network but several neural networks working in tandem, i.e., combining different architectures. One important architecture common to all recent methods is recurrent neural network (RNN).

A RNN introduces the connection between the previous hidden state and current hidden state, and therefore a recurrent layer weight parameters. This recurrent layer is designed to store history information. When reading through a sequence of words, the input and output layers have:

- Input layer:
  - same dimensionality as feature size

- Output layer:
  - represents a probability distribution over labels at time $$t$$
  - same dimensionality as size of labels.

__TODO__: imagem de uma RNN

However, in most proposed techniques, the RNN is replaced by a Long short-term memory (LSTM), where hidden layer updates are replaced by purpose-built memory cells. As a result, they may be better at finding and exploiting long range dependencies in the data.

__TODO__: imagem de uma LSTM, e uma curta descrição
<!-- Basically, a LSTM unit is composed of three multiplicative gates which control the proportions of information to forget and to pass on to the next time step. -->

Another architecture that is combined with LSTMs in the works described in this post is __[Convolutional Neural Networks](../../../../../blog/2018/03/31/SentenceClassificationConvNets/)__.


# __Neural Sequence Labelling Models__

The first ever work to try to use try to LSTMs for the task of Named Entity Recognition was published back in 2003:

- [Named Entity Recognition with Long Short-Term Memory (James Hammerton 2003)](http://www.aclweb.org/anthology/W03-0426)

but lack of computational power led to small and not expressive models, and therefore result behind the other methods at that time.

But I will presented four more recent papers which propose neural network architectures to perform NLP sequence labelling tasks such as NER, chunking, or POS-tagging, I will focus only on the architectures proposed and detailed them, and leave out of the datasets or scores

- [Bidirectional LSTM-CRF Models for Sequence Tagging (Huang et. al 2015)](https://arxiv.org/pdf/1508.01991v1.pdf)

- [Named Entity Recognition with Bidirectional LSTM-CNNs (Chiu and Nichols 2016)](https://www.aclweb.org/anthology/Q16-1026)

- [Neural Architectures for Named Entity Recognition (Lample et. al 2016)](http://www.aclweb.org/anthology/N16-1030)

- [End-to-end Sequence Labelling via Bi-directional LSTM-CNNs-CRF (Ma and Hovy 2016)](http://www.aclweb.org/anthology/P16-1101)

At time this writing there are already new proposed methods, published in 2017 and 2018, but I will leave these for another blog post, for now I just wanted to dissect and understand something from the ones listed above :-)

---

<br>


### [Bidirectional LSTM-CRF Models for Sequence Tagging (2015)](https://arxiv.org/pdf/1508.01991v1.pdf)


### __Architecture__

This was, to the best of my knowledge, the first work to apply a bidirectional-LSTM-CRF architecture for sequence tagging. The idea is to use two LSTMs, one reading each word in a sentence from beginning to end and another reading the same but from end to beginning, producing for each word a vector representation made from both the un-folded LSTM (i.e., forward and backward) read up to that word.

There is no explicit mention in the paper on how the vectors from each LSTM are combined to produce a single vector for each word, I will assume that they are just concatenated.

This bidirectional-LSTM architecture is then combined with a CRF layer at the top, represented by lines which connect consecutive output layers. A Conditional Random Field (CRF) layer has a state transition matrix as parameters, which can efficiently use past and future tags to predict the current tag.

<figure>
  <img style="width: 55%; height: 55%" src="/assets/images/2018-10-21_A_bi-LSTM-CRF_model.png">
  <figcaption><b>TODO: descrever</b></figcaption>
</figure>

<!--
Evaluate them in three sequence tagging task:
- Penn TreeBank (PTB) POS tagging
- CoNLL 2000 chunking
- CoNLL 2003 named entity tagging
-->

<br>

### __Features and Embeddings__

Word embeddings are combined with hand-crafted features: spelling (e.g.: capitalization, punctuation, word patters, etc.) and context (e.g: uni-, bi- and tri-gram features). The embeddings used are those produced by [Collobert et al., 2011]((http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)) which has 130K vocabulary size and each word corresponds to a 50-dimensional embedding vector.

__Features connection tricks__: inputs of networks include both word, spelling and context features, however, the authors suggest direct connections from spelling and context features to outputs accelerate training and they result in very similar tagging accuracy, when comparing without direct connections. That is, in my understanding, the vector representing the hand-crafted features are passed directly to the CRF and are not passed through the bidirectional-LSTM

<figure>
  <img style="width: 55%; height: 55%" src="/assets/images/2018-10-21_A_bi-LSTM-CRF_model_with_max_ent_features.png">
  <figcaption>A bi-LSTM-CRF model with Maximum Entropy features.</figcaption>
</figure>

<br>

## __Summary__

In essence, one can see this architecture as using the output of the bi-LSTM, vector representations for each word in a sentence, together with a vector of features derived from spelling and context hand-crafted rules, these vectors are concatenated and passed to a CRF layer.


<br>

---

#### [Named Entity Recognition with Bidirectional LSTM-CNNs (2016)](https://www.aclweb.org/anthology/Q16-1026)

### __Architecture__

The authors propose a hybrid model combining bi-directional LSTMs with CNNs which learns both character- and word-level features. The proposed system uses words-embeddings, additional hand-crafted word features, and CNN-extracted character-level features. All these extracted features, for each word, are fed into a bidirectional-LSTM.

<figure>
  <img style="width: 42.5%; height: 42.5%" src="/assets/images/2018-10-21-CNN-Char-Embeddings.png">
  <figcaption><b>TODO: descrever</b></figcaption>
</figure>

The output vector of each LSTM (i.e., forward and backward) at each time step is decoded by a linear layer and a log-softmax layer into log-probabilities for each tag category. These two vectors are then added together.

<figure>
  <img style="width: 35%; height: 45%" src="/assets/images/2018-10-21-output_layer.png">
  <figcaption><b>TODO: descrever</b></figcaption>
</figure>

<br>

Character-level features are induced by a Convolutional Neural Network (CNN) architecture, which has been successfully applied to Spanish and Portuguese NER [(Santos et al., 2015)](http://www.anthology.aclweb.org/W/W15/W15-3904.pdf) and German POS-tagging [(Labeau et al., 2015)](http://www.aclweb.org/anthology/D15-1025). For each word a convolution and a max layer are applied to extract a new feature vector from the per-character feature vectors such as character embeddings and character type.

<figure>
  <img style="width: 42.5%; height: 42.5%" src="/assets/images/2018-10-21-bi-directional-LSTM-with-CNN-chars.png">
  <figcaption><b>TODO: descrever</b></figcaption>
</figure>

<br>

### __Features and Embeddings__

__Word Embeddings__: 50-dimensional word embeddings [(Collobert et al. 2011)](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf), all words are lower-cased, embeddings are allowed to be modified during training.

__Character Embeddings__: randomly initialized a lookup table with values drawn from a uniform distribution with range [−0.5,0.5] to output a character embedding of 25 dimensions. Two special tokens are added: PADDING and UNKNOWN.

__Additional Char Features__ A lookup table was used to output a 4-dimensional vector representing the type of the character (_upper case_, _lower case_, _punctuation_, _other_).

__Additional Word Features__: each words is tagged as _allCaps_, _upperInitial_, _lowercase_, _mixedCaps_, _noinfo_.

__Lexicons__: partial lexicon matches using a list of known named-entities from DBpedia. The list is then used to perform $n$-gram matches against the words. A match is successful when the $n$-gram matches the prefix or suffix of an entry and is at least half the length of the entry.

<br>

## __Summary__

The authors explore several features: word embeddings, word shape features, character-level features and lexical features. The character-level features are extracted with a CNN. All these features are then concatenated, passed through a bi-LSTM and each time step is decoded by a linear layer and a log-softmax layer into log-probabilities for each tag category. At inference time they use the Viterbi algorithm to select the sequence that maximizes the score all possible tag-sequences.

<!--
CoNLL-2003
OntoNotes 5.0
-->


---

<br>

### [Neural Architectures for Named Entity Recognition (2016)](http://www.aclweb.org/anthology/N16-1030)

### __Architecture__

This was, to the best of my knowledge, the first work on NER to completely drop hand-crafted features, i.e., they use no language-specific resources or features beyond a small amount of supervised training data and unlabeled corpora.

Two new neural architectures are proposed:

- bidirectional LSTMs + Conditional Random Fields (CRF)
- generating labels segments using a transition-based approach inspired by shift-reduce parsers

I will just focus on the first model, which follows a similar architecture as the other models presented in this post. As in the other models, two LSTMs are used to generate a word representation by concatenating its left and right context. These are two distinct LSTMs with different parameters.

<figure>
  <img style="width: 42.5%; height: 42.5%" src="/assets/images/2018-10-21-neural-arch.png">
  <figcaption><b>TODO: descrever</b></figcaption>
</figure>

<!--
Our models are designed to capture two intuitions. First, since names often consist of multiple tokens, reasoning jointly over tagging decisions for each to- ken is important. We compare two models here, (i) a bidirectional LSTM with a sequential conditional random layer above it (LSTM-CRF; §2), and (ii) a new model that constructs and labels chunks of input sentences using an algorithm inspired by transition-based parsing with states represented by stack LSTMs (S-LSTM; §3). Second, token-level evidence for “being a name” includes both orthographic evidence (what does the word being tagged as a name look like?) and distributional evidence (where does the word being tagged tend to occur in a corpus?). To capture orthographic sensitivity, we use character-based word representation model (Ling et al., 2015b) to capture distributional sensitivity, we combine these representations with distributional representations (Mikolov et al., 2013b). Our word representations combine both of these, and dropout training is used to encourage the model to learn to trust both sources of evidence (§4).
-->

The tagging decisions are not modeled independently, and are instead modeled jointly using a CRF [(Lafferty et al., 2001)](https://repository.upenn.edu/cgi/viewcontent.cgi?article=116). The parameters of this model are thus the matrix of bigram compatibility scores $A$, and the parameters that give rise to the matrix $P$, namely the parameters of the bidirectional LSTM, the linear feature weights, and the word embeddings.


### __Embeddings__

The authors generate words embeddings from both representations of the characters of the word and from the contexts where the word occurs.

The rational behinds this idea is that many languages have orthographic or morphological evidence that something is a named-entity or not, second is that named-entities appear in somewhat regular contexts in large corpora, therefore they use embeddings learned from a large corpus that are sensitive to word order.

#### __Character Embeddings__

<figure>
  <img style="width: 42.5%; height: 42.5%" src="/assets/images/2018-10-21-nerual-arch-char-embeddings.png">
  <figcaption><b>TODO: descrever</b></figcaption>
</figure>

A character lookup table is initialized randomly containing an embedding for every character. The character embeddings corresponding to every character in a word are given in direct and reverse order to a bidirectional-LSTM. The embedding for a word derived from its characters is the concatenation of its forward and backward representations from the bidirectional-LSTM. The hidden dimension of the forward and backward character LSTMs are 25 each.

#### __Word Embeddings__

This character-level representation is then concatenated with a word-level representation from pre-trained word embeddings. Embeddings are pre-trained using skip-n-gram [(Ling et al., 2015)](http://www.aclweb.org/anthology/D15-1161), a variation of skip-gram that accounts for word order.

These embeddings are fine-tuned during training, and the authors claim that using pre-trained over randomly initialized ones results in performance improvements.

They also mention that they apply a dropout mask to the final embedding layer just before the input to the bidirectional LSTM observe a significant improvement in model’s performance after using dropout.


## __Summary__

This model is relatively simple, the authors use no hand-crafted features, just embeddings. The word embeddings are the concatenation of two vectors, a vector made of character embeddings using two LSTMs, for each character in a word, and a vector corresponding to word embeddings trained on external data.

The embeddings for word each word in a sentence are then passed through a forward and backward LSTM, and the output for each word is then fed into a CRF layer.

<!--
named entity recognition
CoNLL-2002 and CoNLL2003 datasets
named entity labels for English, Spanish, German and Dutch.

All datasets contain four different types of named entities: locations, persons, organizations, and miscellaneous entities that do not belong in any of the three previous categories.

Although POS tags were made available for all datasets, we did not include them in our models.

We did not perform any dataset preprocessing, apart from replacing every digit with a zero in the English NER dataset.
-->




## __Implementations__

- [https://github.com/glample/tagger](https://github.com/glample/tagger)
- [https://github.com/Hironsan/anago](https://github.com/Hironsan/anago)
- [https://github.com/achernodub/bilstm-cnn-crf-tagger](https://github.com/achernodub/bilstm-cnn-crf-tagger)
<!--- [https://github.com/clab/stack-lstm-ner](https://github.com/clab/stack-lstm-ner)-->

---

<br>


#### [End-to-end Sequence Labelling via Bi-directional LSTM-CNNs-CRF (2016)](http://www.aclweb.org/anthology/P16-1101)

### __Architecture__

This proposed system is very similar to the previous one. The authors use convolutional neural networks (CNNs) to encode character-level information of a word into its character-level representation. Then combine character- and word-level representations and feed them into bi-directional LSTM (BLSTM) to model context information of each word. Finally, the output vectors of BLSTM are fed to the CRF layer to jointly decode the best label sequence.
<figure>
  <img style="width: 42.5%; height: 42.5%" src="/assets/images/2018-10-21_end_to_ent2.png">
  <figcaption><b>TODO: descrever</b></figcaption>
</figure>


### __Embeddings__

#### __Character Embeddings__

The CNN is similar to the one in [Chiu and Nichols (2015)](https://www.aclweb.org/anthology/Q16-1026), the second system presented, except that they use only character embeddings as the inputs to CNN, without any character type features. A dropout layer is applied before character embeddings are input to CNN.

<figure>
  <img style="width: 42.5%; height: 42.5%" src="/assets/images/2018-10-21_end_to_ent1.png">
  <figcaption><b>TODO: descrever</b></figcaption>
</figure>

#### __Word Embeddings__

The word embeddings are the publicly available GloVe 100-dimensional embeddings trained on 6 billion words from Wikipedia and web text.


## __Summary__

This model follows basically the same architecture as the one presented before, being the only architecture change the fact that they use CNN to generate word-level char-embeddings instead of an LSTM.

<!--
POS Tagging. For English POS tagging, we use
the Wall Street Journal (WSJ) portion of Penn
Treebank (PTB) (Marcus et al., 1993), which contains
45 different POS tags. In order to compare
with previous work, we adopt the standard
splits — section 0–18 as training data, section 19–
21 as development data and section 22–24 as test
data (Manning, 2011; Søgaard, 2011).
NER. For NER, We perform experiments on
the English data from CoNLL 2003 shared
task (Tjong Kim Sang and De Meulder, 2003).
This data set contains four different types of
named entities: PERSON, LOCATION, ORGANIZATION,
and MISC. We use the BIOES tagging
scheme instead of standard BIO2, as previous
studies have reported meaningful improvement
with this scheme (Ratinov and Roth, 2009;
Dai et al., 2015; Lample et al., 2016).
The corpora statistics are shown in Table 2. We
did not perform any pre-processing for data sets,
leaving our system truly end-to-end.
-->









## __Implementations__

- [https://github.com/achernodub/bilstm-cnn-crf-tagger](https://github.com/achernodub/bilstm-cnn-crf-tagger)


---

<br>

## __Comparative Summary__

In the following table I try to summarize the main characteristics of each of the models

<table class="blueTable">
<thead>
<tr>
<th>&nbsp;</th>
<th>Features</th>
<th>Architecture Resume</th>
<th>Structured Tagging</th>
<th>Embeddings</th>
</tr>
</thead>
<tbody>
<tr>
<td>(Huang et. al 2015)</td>
<td>Yes</td>
<td>
bi-LSTM output vectors +
<br>
features vectors connected to CRF</td>
<td>CRF</td>
<td>Collobert et al. 2011
<br>
pre-trained
<br>
50-dimensions</td>
</tr>
<tr>
<td>(Chiu and Nichols 2016)</td>
<td>Yes</td>


<td>
<!--
word embeddings + features vector passed through a bi-LSTM the outupt at each time step of the bi-LSTM is added and then is decoded
-->
</td>
<td>
<!--
time step in the bi-LSTM is decoded by a linear layer and a log-softmax layer into log-probabilities for each tag category
Viterbi algorithm the tag sequence that maximizes the score all possible tag-sequences.-->
</td>


<td>
- Collobert et al. 2011
<br>
- char-level embeddings
<br>
extracted with a CNN</td>
</tr>
<tr>
<td>(Lample et. al 2016)</td>
<td>No</td>
<td>
chars and word embeddings
<br>
input for the bi-LSTM
<br>
output vectors are fed to the CRF layer to  jointly decode the best label sequence
</td>
<td>CRF</td>
<td>
- char-level embeddings
<br>
extracted with a bi-LSTM
<br>
- pre-trained word embeddings
<br>
with skip-n-gram</td>
</tr>
<tr>
<td>(Ma and Hovy 2016)</td>
<td>No</td>
<td>
chars and word embeddings
<br>
input for the bi-LSTM
<br>
output vectors are fed to the CRF layer to  jointly decode the best label sequence
</td>
<td>CRF</td>
<td>
- char embeddings extracted with a CNN
<br>
- word embeddings: GloVe 100-dimensions</td>
</tr>
</tbody>
</table>

---

## __References__

<!--
https://www.lewuathe.com/machine%20learning/crf/conditional-random-field.html
-->

- [Bidirectional LSTM-CRF Models for Sequence Tagging (Huang et. al 2015)](https://arxiv.org/pdf/1508.01991v1.pdf)

- [Named Entity Recognition with Bidirectional LSTM-CNNs (Chiu and Nichols 2016)](https://www.aclweb.org/anthology/Q16-1026)

- [Neural Architectures for Named Entity Recognition (Lample et. al 2016)](https://www.aclweb.org/anthology/N16-1030)

- [End-to-end Sequence Labelling via Bi-directional LSTM-CNNs-CRF (Ma and Hovy 2016)](http://www.aclweb.org/anthology/P16-1101)

- [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://www.robots.ox.ac.uk/~vgg/rg/papers/hmm.pdf)

- [Hugo Larochelle on-line lessons - Neural networks [4.1] : Training CRFs - loss function](https://www.youtube.com/watch?v=6dpGB60Q1Ts)

- [Blog article: CRF Layer on the Top of BiLSTM - 1 to 8](https://createmomo.github.io/)

- [Not All Contexts Are Created Equal: Better Word Representations with Variable Attention (Ling et al., 2015)](http://www.aclweb.org/anthology/D15-1161)

- [Non-lexical neural architecture for fine-grained POS Tagging (Labeau et al., 2015)](http://www.aclweb.org/anthology/D15-1025)

- [Boosting Named Entity Recognition with Neural Character Embeddings (Santos et al., 2015)](http://www.anthology.aclweb.org/W/W15/W15-3904.pdf)

- [Natural Language Processing (Almost) from Scratch (2011)](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

---

<br><br>

### __Extra: Why a Conditional Random Field (CRF) at the top?__

Having independent classification decisions is limiting when there are strong dependencies across output labels, since you decide which label for a word independently from the previous given tags.

For sequence labeling or general structured prediction tasks, it is beneficial to consider the correlations between labels in neighborhoods and jointly decode the best chain of labels for a given input sentence:

- NER is one such task, since the”"grammar” that characterizes interpretable sequences of tags imposes several hard constraints, e.g.: I-PER cannot follow B-LOC that would be impossible to model with independence assumptions.

- Another example, in POS tagging an adjective is more likely to be followed by a noun than a verb;

The idea of using a CRF at the top is to model tagging decisions jointly. This means that the CRF layer could add constrains to the final predicted labels ensuring they are valid. The constrains are learned by the CRF layer automatically based on the annotated samples during the training process.


#### __Emission score matrix__

The output of the LSTM is given as input to the CRF layer, that is, a matrix $\textrm{P}$ with the scores of the LSTM of size $n \times k$, where $n$ is the number of words in the sentence and $k$ is the possible number of labels that each word can have, $\textrm{P}_{i,j}$ is the score of the $j^{th}$ tag of the $i^{th}$ word in the sentence.

__TODO__ desenhar as matriz ligada ao CRF

#### __Transition matrix__

$$\textrm{T}$$ is a matrix of transition scores such that $$\textrm{P}_{i,j}$$ represents the score of a transition from the tag $$i$$ to tag $$j$$. Two extra tags are added, $y_{0}$ and $y_{n}$ are the _start_ and _end_ tags of a sentence, that we add to the set of possible tags, $\textrm{T}$ is therefore a square matrix of size $\textrm{k}+2$.

__TODO__ desenhar as matriz

#### __Score of a prediction__

For a given sequence of predictions for a sequence of words $$x$$:

$$\textrm{y} = (y_{1},y_{2},\dots,y_{n})$$

we can compute it's score based on the _emission_ and _transition_ matrices:

$$\textrm{score} = \sum_{i=0}^{n} \textrm{T}_{y_i,y_{i+1}} + \sum_{i=1}^{n} \textrm{P}_{i,y_i}$$

so the score of a sequence of predictions is, for each word, the sum of the transition from the current assigned tag $$y_i$$ to next assigned tag $$y_{i+1}$$ plus the probability given by the LSTM to the tag assigned for the current word $$i$$.

#### __Training: parameter estimation__

During training, assign a probability to each tag but maximizing the probability of the correct tag $$y$$ sequence among all the other possible tag sequences.

This is modeled by applying a softmax over all the possible taggings $$y$$:

$$\textrm{p(y|X)} = \frac{e^{score(X,y)}}{\sum\limits_{y' \in Y({x})} e^{score(X,y')}}$$

where $$Y(x)$$ denotes the set of all possible label sequences for $$x$$, this denominator is also known as the partition function. So, finding the best sequence is the equivalent of finding the sequence that maximizes $$\textrm{score(X,y)}$$.

The loss can be defined as the negative log likelihood of the current tagging $$y$$:

$$\textrm{-log p}(y\textrm{|X)}$$

so, in simplifying the function above, a first step is to get rid of the fraction using log equivalences, and then get rid of the $$\textrm{log}\  e$$ in the first term since they cancel each other out:

$$\textrm{-log p}(y\textrm{|X)} = -\ \textrm{score(X,y)} + \textrm{log} \sum\limits_{y' \in Y({x})} \textrm{exp}(\textrm{score(X,y')})$$

then the second term can be simplified by applying the log-space addition _logadd_, equivalence, i.e.: $$\oplus(a, b, c, d) = log(e^a+e^b+e^c+e^d)$$:

$$\textrm{-log p}(y\textrm{|X)} = -\ \textrm{score(X,y)} + \underset{y' \in Y({x})}{\text{logadd}} (\textrm{score(X,y')})$$


then, replacing the $$\textrm{score}$$ by it's definition:

$$ = - (\sum_{i=0}^{n} \textrm{T}_{y_i,y_{i+1}} + \sum_{i=1}^{n} \textrm{P}_{i,y_i}) + \underset{y' \in Y({x})}{\text{logadd}}(\sum_{i=0}^{n} \textrm{T}_{y'_i,y'_{i+1}} + \sum_{i=1}^{n} \textrm{P}_{i,y_i})$$

The first term is score for the true data. Computing the second term might be computationally expensive since it requires summing over the $$k^{n}$$ different sequences in $$Y(x)$$, i.e., the set of all possible label sequences for $$x$$. This computation can be solved using a variant of the Viterbi algorithm, the forward algorithm.

The gradients are then computed using back-propagation, since the CRF is inside the neural-network. Note that the transition scores in the matrix are randomly initialized - or can also bee initialized based on some criteria, to speed up training - and will be updated automatically during your training process.

#### __Inference: determining the most likely label sequence $$y$$ given $$X$$__

Decoding is to search for the single label sequence with the largest joint probability conditioned on the input sequence:

$$\underset{y}{\arg\max}\ \textrm{p(y|X;}\theta)$$


the parameters $$\theta$$ correspond to the _transition_ and _emission_ matrices, basically the task is finding the best $$\hat{y}$$ given the transition matrix $$\textrm{T}$$ and the matrix $$\textrm{P}$$ with scores for each tag for the individual word:

$$\textrm{score} = \sum_{i=0}^{n} \textrm{T}_{y_i,y_{i+1}} + \sum_{i=1}^{n} \textrm{P}_{i,y_i}$$

a linear-chain sequence CRF model, models only interactions between two successive labels, i.e bi-gram interactions, therefore one can find the sequence $$y$$ maximizing the __score__ function above by adopting the Viterbi algorithm (Rabiner, 1989).
