---
layout: post
title: Sentence Transformers
date: 2023-08-20 00:00:00
tags: sentence-transformers BERT triplet-loss embeddings fine-tuning
categories: [blog]
comments: true
disqus_identifier: 20230820
preview_pic: /assets/images/2023-08-20-sentence-transformer-fine-tunning.png
---

The sentence-transformers propose in __[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410.pdf)__ is an effective and efficient way to have to train a neural network such that it represents good embeddings for a sentence or paragraph based on the Transformer architecture. In this post I review mechanism to train such embeddings presented in the paper.


## __Introduction__

Measuring the similarity of a pair of short-text (.e.g: sentences or paragraphs) is common task in NLP. One can achieve this with BERT, using a cross-encoder, i.e.: concatenating both sentences with a separating token __[SEP]__, passing this input to the transformer network and the target value is predicted, in this case, similar or not similar.

Nevertheless this approach grows quadratically due to too many possible combinations, for instance, finding in a collection of $$n$$ sentences the pair with the highest similarity requires with BERT $$n \cdot(n−1)/2$$ inference computations.

Alternatively one could also rely on BERT's output layer, i.e.: BERT embeddings, or by using the output of the first token, i.e.: the __[CLS]__ token, but as the authors shows this often leads to worst results than just using static-embeddings, like GloVe embeddings (Pennington et al., 2014)

To overcame this issue, and still use the contextual word embeddings representations provided by BERT (or any other Transformer-based model) the authors use a pre-trained BERT and RoBERTa network and fine-tune it to yield useful sentence embeddings, meaning that semantically similar sentences are close in vector space.


## __Fine-Tuning BERT for Semantically (Dis)Similarity__

A main architectural components of this approach is a __Siamese Neural Network__: a neural network containing two or more identical sub-networks, meaning they have the same configuration with the same parameters and weights, and parameter updating is mirrored across both sub-networks.

<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2023-08-20-sentence-transformer-fine-tunning.png">
  <figcaption>Figure 1 - </figcaption>
</figure>


The training procedure of the network is the following:

- Each input sentence is feed into BERT, producing embeddings for each token of the sentence.

- To have fixed-sized output representation the authors apply a pooling layer, three strategies are explored: 

	- using the output of the __[CLS]-token__
	 
	- __MEAN-strategy__: computing the mean of all output vectors  
	
	- __MAX-strategy__: computing a max-over-time of the output vectors

- The network objective function depends on the training data:

	- __Classification__: 
	
		$$o = \text{softmax}(Wt(u, v, |u − v|))$$
		
		concatenate the sentence embeddings $$u$$ and $$v$$ with the element-wise difference $$ \|u − v\| $$ and multiply it with the trainable weight $$ W_t \in R3n×k $$: 
			
	- __Regression__:
	
		$$cos(u,v)$$
	
	- __Triplet Objective Function__:
	
		$$max(||s_{a} − s_{p}|| − || s_{a} − s_{n} || + ε, 0)$$
		
		a baseline (anchor) input is compared to a positive (truthy) input and a negative (falsy) input. The distance from the baseline (anchor) input to the  positive (truthy) input is minimized, and the distance from the baseline (anchor) input to the negative (falsy) input is maximized.



### __Training Data__

The objective function (classification vs. regression) depends on the training data. For the __classification objective function__, the authors used the __[The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)__, a collection of 570,000 sentence pairs annotated with the labels: 

- __contradiction__ 
- __entailment__ 
- __neutral__ 

The __[The Multi-Genre NLI Corpus](https://cims.nyu.edu/~sbowman/multinli/)__ containing 430,000 sentence pairs and covers a range of genres of spoken and written text.

For the __regression objective function__, the authors trained on the training set of the Semantic Textual Similarity (STS) benchmark dataset from SemEval.

### __Training: fine-tuning__

In order to fine-tune BERT and RoBERTa, the authors used a __Siamese Neural Network (SNN)__ strategy to update the weights such that the produced sentence embeddings are semantically meaningful.

An SNNs can be used to find the similarity of the inputs by comparing its feature vectors, so these networks learn a similarity function that takes two inputs and outputs 1 if they belong to the same class and zero other wise, 

It learn parameters such that, if the two sentences are similar:

$$|| f(x_1) - (f(x_2)||^2 \text{ is small} $$

and if the two sentences are dissimilar:

$$|| f(x_1) - (f(x_2) ||^2 \text{ is large}$$

<!--
Another way using the triplet loss function to train the network to output a good encoding.

We used a batch-size of 16, Adam optimizer with learning rate 2e−5,  linear learning rate warm-up over 10% of the training data. 

Our default pooling strategy is MEAN.

We fine-tune SBERT with a 3-way softmax-classifier objective function for one epoch. 
-->


### __Evaluation__

We evaluate the performance of SBERT for common Semantic Textual Similarity (STS) tasks. State-of-the-art methods often learn a (complex) regression function that maps sentence embed- dings to a similarity score. However, these regres- sion functions work pair-wise and due to the combinatorial explosion those are often not scalable if the collection of sentences reaches a certain size. Instead, we always use cosine-similarity to com- pare the similarity between two sentence embed- dings. We ran our experiments also with nega- tive Manhatten and negative Euclidean distances as similarity measures, but the results for all approaches remained roughly the same.


<figure>
  <img style="width: 40%; height: 40%" src="/assets/images/2023-08-20-sentence-transformer-inference.png">
  <figcaption>Figure 2 - </figcaption>
</figure>


We evaluate the performance of SBERT for com- mon Semantic Textual Similarity (STS) tasks.


STS12-STS16: SemEval 2012-2016, STSb: STSbenchmark, SICK-R: SICK relatedness dataset.


Semantic Textual Similarity (STS) from SemEval datasets of [2012](https://aclanthology.org/S12-1051/), [2013](https://aclanthology.org/S13-1004/), [2014](https://aclanthology.org/S14-2010/), [2015](https://aclanthology.org/S15-2045/), [2016](https://aclanthology.org/S16-1081/)


STS benchmark :https://aclanthology.org/S17-2001/

SICK (Sentences Involving Compositional Knowledge): http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf

SentEval https://aclanthology.org/L18-1269/


<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2023-08-20-sentence-transformer-results-STS.png">
  <figcaption>Figure 3 - </figcaption>
</figure>


#### __Supervised STS__


<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2023-08-20-sentence-transformer-supervised-results-STS.png">
  <figcaption>Figure 4 - </figcaption>
</figure>

We use the training set to fine-tune SBERT us- ing the regression objective function. At predic- tion time, we compute the cosine-similarity be- tween the sentence embeddings.



####  __Argument Facet Similarity__


<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2023-08-20-sentence-transformer-argument-facet-sim.png">
  <figcaption>Figure 5 - </figcaption>
</figure>


#### __Wikipedia Sections Distinction__

<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2023-08-20-sentence-transformer-wikipedia-triples.png">
  <figcaption>Figure 6 - </figcaption>
</figure>

Triplet Objective is used on this experiment


### __Evaluation - SentEval__


<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2023-08-20-sentence-transformer-senteval.png">
  <figcaption>Figure 7 - </figcaption>
</figure>


#### __Ablation Study__

We evaluated different pooling strategies (MEAN, MAX, and CLS). For the classification objective function, we evaluate different concatenation methods. For each possible configuration, we train SBERT with 10 different random seeds and average the performances.

At inference, when predicting similarities for the STS benchmark dataset, only the sentence embeddings u and v are used in combi- nation with cosine-similarity.

<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2023-08-20-sentence-ablation-study.png">
  <figcaption>Figure 8 - </figcaption>
</figure>


### __Conclusions__

<!--
We showed that BERT out-of-the-box maps sentences to a vector space that is rather unsuitable to be used with common similarity measures like cosine-similarity. The performance for seven STS tasks was below the performance of average GloVe embeddings.

To overcome this shortcoming, we presented Sentence-BERT (SBERT). SBERT fine-tunes BERT in a siamese / triplet network architecture. We evaluated the quality on various com- mon benchmarks, where it could achieve a significant improvement over state-of-the-art sen- tence embeddings methods. Replacing BERT with RoBERTa did not yield a significant improvement in our experiments.
SBERT is computationally efficient. 

SBERT can be used for tasks which are computationally not feasible to be modeled with BERT. For exam- ple, clustering of 10,000 sentences with hierarchical clustering requires with BERT about 65 hours, as around 50 Million sentence combinations must be computed. With SBERT, we were able to re- duce the effort to about 5 seconds.
-->


### __References__

- __[Triplet Loss and Online Triplet Mining in TensorFlow from Olivier Moindrot blog](https://omoindrot.github.io/triplet-loss#why-not-just-use-softmax)__
