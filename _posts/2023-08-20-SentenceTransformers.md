---
layout: post
title: Sentence Transformers
date: 2023-10-22 00:00:00
tags: sentence-transformers triplet-loss embeddings fine-tuning
categories: [blog]
comments: true
disqus_identifier: 20231022
preview_pic: /assets/images/2023-10-22-sentence-transformer-fine-tunning.png
---

The `sentence-transformers` proposed in __[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410.pdf)__ is an effective and efficient way to train a neural network such that it represents good embeddings for a sentence or paragraph based on the Transformer architecture. In this post I review mechanism to train such embeddings presented in the paper.


## __Introduction__

Measuring the similarity of a pair of short-text (.e.g: sentences or paragraphs) is a common NLP task. One can achieve this with BERT, using a __cross-encoder__, concatenating both sentences with a separating token __[SEP]__, passing this input to the transformer network and the target value is predicted, in this case, similar or not similar.

But, this approach grows quadratically due to too many possible combinations. For instance, finding in a collection of $$n$$ sentences the pair with the highest similarity requires with BERT $$n \cdot(n−1)/2$$ inference computations.

Alternatively one could also rely on BERT's output layer, the embeddings, or use the output of the first token, i.e.: the __[CLS]__ token, but as the authors shows this often leads to worst results than just using static-embeddings, like GloVe embeddings __[(Pennington et al., 2014)](https://aclanthology.org/D14-1162/)__.

To overcame this issue, and still use the contextual word embeddings representations provided by BERT (or any other Transformer-based model) the authors use a pre-trained BERT/RoBERTa network and fine-tune it to yield useful sentence embeddings, meaning that semantically similar sentences are close in vector space.

## __Fine-Tuning BERT for Semantically (Dis)Similarity__

The main architectural components of this approach is a __Siamese Neural Network__, a neural network containing two or more identical sub-networks, whose weights are updated equally across both sub-networks.

<figure>
  <img style="width: 40%; height: 40%" src="/assets/images/2023-10-22-sentence-transformer-fine-tunning.png">
  <figcaption>Figure 1 - The architecture to fine-tune sentence-transformers.</figcaption>
</figure>


The training procedure of the network is the following. Each input sentence is feed into BERT, producing embeddings for each token of the sentence. To have fixed-sized output representation the authors apply a pooling layer, exploring three strategies: 

-  __[CLS]-token__
	 
- __MEAN-strategy__: computing the mean of all output vectors  
	
- __MAX-strategy__: computing a max-over-time of the output vectors

They experiment with 3 different network objective function depending on the training data:

__Classification__:

$$o = \text{softmax}(Wt(u, v, |u − v|))$$
   		
concatenating the sentence embeddings $$u$$ and $$v$$ with the element-wise difference $$ \vert u − v \vert $$ and multiply it with the trainable weight $$ W_t \in R^{3n×k} $$.


__Regression__:

$$cos(u,v)$$

simply the cosine between the two sentence embeddings.


__Triplet Objective Function__:	

 $$max(||s_{a} − s_{p}|| − || s_{a} − s_{n} || + ε, 0)$$
 
a baseline anchor $$a$$ input is compared to a positive $$p$$ input and a negative $$n$$ input. The distance from the baseline $$a$$ to the positive $$p$$ input is minimized, and the distance from the baseline $$a$$ to the negative $$n$$ is maximized. As far as I've understood this objective function is only used in one experiment with the Wikipedia Sections Distinction dataset.

### __Training Data__

The objective function (classification vs. regression) depends on the training data. For the __classification objective function__, the authors used the __[The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)__, a collection of 570,000 sentence pairs annotated with the labels: 

- __contradiction__ 
- __entailment__ 
- __neutral__ 

The __[The Multi-Genre NLI Corpus](https://cims.nyu.edu/~sbowman/multinli/)__ containing 430,000 sentence pairs and covers a range of genres of spoken and written text.

For the __regression objective function__, the authors trained on the training set of the Semantic Textual Similarity (STS) benchmark dataset from SemEval.

### __Training: fine-tuning__

In order to fine-tune __[BERT](https://aclanthology.org/N19-1423/)__ and __[RoBERTa](https://arxiv.org/abs/1907.11692)__, the authors used a __Siamese Neural Network (SNN)__ strategy to update the weights such that the produced sentence embeddings are semantically meaningful.

An SNNs can be used to find the similarity of the inputs by comparing its feature vectors, so these networks learn a similarity function that takes two inputs and outputs 1 if they belong to the same class and zero other wise, 

It learn parameters such that, if the two sentences are similar:

$$|| f(x_1) - (f(x_2)||^2 \text{ is small} $$

and if the two sentences are dissimilar:

$$|| f(x_1) - (f(x_2) ||^2 \text{ is large}$$

where $$f(x)$$ is embedding of $$x$$.

__Configuration parameters__
- batch-size: 16, 
- Adam optimizer with learning rate 2e−5,  
- linear learning rate warm-up over 10% of the training data. 
- MEAN as the default pooling strategy is .
-  3-way softmax-classifier objective function for one epoch. 

### __Evaluation__

The authors evaluated their approach on several datasets common Semantic Textual Similarity (STS) tasks, using the cosine-similarity to compare the similarity between two sentence embeddings, in opposition to learning a regression function that maps two sentence embeddings to a similarity score. They also experimented with __[Manhatten](https://en.wikipedia.org/wiki/Taxicab_geometry)__ and negative __[Euclidean distances](https://en.wikipedia.org/wiki/Euclidean_distance)__ as similarity measures, but the results remained roughly the same.

<figure>
  <img style="width: 35%; height: 40%" src="/assets/images/2023-10-22-sentence-transformer-inference.png">
  <figcaption>Figure 2 - sentence-transformers in inference mode.</figcaption>
</figure>

To recapitulate BERT and RoBERTa are fine-tuned using the training described above, and the resulting model are used to generate embeddings for sentences, whose similarity is measured by the cosine.

- SemEval 2012-2016 - Semantic Textual Similarity (STS) datasets: __[2012](https://aclanthology.org/S12-1051/), [2013](https://aclanthology.org/S13-1004/), [2014](https://aclanthology.org/S14-2010/), [2015](https://aclanthology.org/S15-2045/), [2016](https://aclanthology.org/S16-1081/)__
- __[SICK (Sentences Involving Compositional Knowledge)](http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf)__
- __SemEval-2017 Task 1: [STSimilarity Multilingual and Crosslingual Focused Evaluation](https://aclanthology.org/S17-2001/)__


<figure>
  <img style="width: 75%; height: 50%" src="/assets/images/2023-10-22-sentence-transformer-results-STS.png">
  <figcaption>Figure 3 - Results from the experimental evaluation. </figcaption>
</figure>

The authors also carry other experiments with a few other datasets, but I will refer the reader to the the original paper for further details.

### __Ablation Study__

The study explored different methods to concatenate the sentence embeddings for training the softmax classifier in the process of fine-tuning a BERT/RoBERTa transformer model. 

According to the authors the most important component is the element-wise difference $$ \vert u − v \vert $$ which measures the distance between the dimensions of the two sentence embeddings, ensuring that similar pairs are closer and dissimilar pairs are further apart.

<figure>
  <img style="width: 35%; height: 50%" src="/assets/images/2023-10-22-sentence-ablation-study.png">
  <figcaption>Figure 8 - Results from the ablations study.</figcaption>
</figure>


### __Implementation and others__

The `sentence-transformers` package gain popularity in the NLP community and can be used for multiple tasks as semantic text similarity, semantic search, retrieve and re-rank, clustering and others, see the official webpage [SBERT](https://www.sbert.net/examples/applications/clustering/README.html) for several tutorials and API documentation

One of the authors of the paper [Nils Reimers](https://www.nils-reimers.de/) has made several talks on ideas and approaches levering on sentence-transformers, here are two I've found interesting:

- __[Training State-of-the-Art Sentence Embedding Models](https://www.youtube.com/watch?v=RHXZKUr8qOY)__
- __[Introduction to Dense Text Representations](https://www.youtube.com/watch?v=qmN1fJ7Fdmo&list=PL7kaex1gKh6BDLHEwEeO45wZRDm5QlRil&index=1)__


### __References__

- __[Triplet Loss and Online Triplet Mining in TensorFlow from Olivier Moindrot blog](https://omoindrot.github.io/triplet-loss#why-not-just-use-softmax)__
