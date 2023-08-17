---
layout: post
title: NLP Text Summarisation
date: 2023-08-13 00:00:00
tags: text-summarisation transformers
categories: [blog]
comments: true
disqus_identifier: 20230813
preview_pic: /assets/images/2023-08-13-NLP_Text_Summarisation.jpg
---

Text Summarisation is a technique that resumes in a shorter version the information contained in the original text, this can be achieved by selecting 
representative words or phrases from the original text and combining them to make a summary (extractive), or by writing a summary as a human would do, 
by generating a textual summary of the main content in the original text (abstractive). In this post, I will review some recent works on NLP Text summarisation.


### __Extractive summarisation__
* Select representative words or phrases from the original text and combine them to make a summary
* Scores sentences by their relevance to the whole text's meaning


#### __2004 - TextRank__
* [TextRank: Bringing Order into Text](https://aclanthology.org/W04-3252)
* inspired by PageRank was part of gensim package - removed on 4.0.0


#### __2011 - LexRank__
* [LexRank: Graph-based Lexical Centrality as Salience in Text Summarization](https://arxiv.org/abs/1109.2128)


#### __2019 - HIBERT__
* [Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://aclanthology.org/P19-1499/)
* Relies on two encoders to obtain the representation of a document
* A sentence encoder to transform each sentence to a vector
* A document encoder learns sentence representations based on their surrounding context sentences


#### __2019 - BERTSumExt__
* [Text Summarization with Pretrained Encoders](https://aclanthology.org/D19-1387.pdf)
* An encoder creates sentence representations and a classifier predicts which sentences should be selected as summaries
* Has a document-level encoder based on BERT to obtain sentence representations
* The modification mainly consists of surrounding each sentence of the text with a [CLS] (which represents the entire sentence) and [SEP] (which represents the boundary between two sentences) and assigning different segment embeddings for every pair of sentences
* Sentence-level contextual representations fed to a classifier for binary classification


#### __2020 - MatchSum__
* [Extractive Summarization as Text Matching](https://aclanthology.org/2020.acl-main.552/)
* The source document and candidate summaries will be extracted from the original text and matched in a semantic space
* Siamese-BERT architecture to compute the similarity between the source document and the candidate summary
* Leverages the pre-trained BERT in a Siamese network structure to derive semantically meaningful text embeddings that can be compared using cosine-similarity


<br>

---

<br>

### __Abstractive summarisation__
* Generate a textual summary of the main content in the original text
* Writes a summary as a human would do
* “old school”
	* template-based, rule-based
* Seq2Seq (encoder-decoder) - RNN/LSTM - vanilla + augmented with Attention Mechanism
	* Abstractive Sentence Summarization with Attentive Recurrent Neural Networks (2016)
* “Attention is All You Need” (2017) replaced the RNN/LSTM arch by using Self-Attention
	* BertSumAbs: Text Summarization with Pretrained Encoders (2019)
	* T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2019)
	* BART Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation... (2020)
	* PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization (2020)
* LLM
* ChatGPT


#### BERTSumAbs
− Adopts an encoder-decoder architecture, combining the same pre-trained BERT encoder and a randomly-initialised Transformer decoder
− Training separates the optimisers of the encoder and the decoder in order to accommodate the fact that the encoder is pre-trained while the decoder must be trained from scratch
− Propose a two-stage fine-tuning approach, where we first fine-tune the encoder on the extractive summarisation task and then fine-tune it on the abstractive summarisation task
− Combine extractive and abstractive objectives, a two-stage approach: the encoder is fine-tuned twice, first with an extractive objective and subsequently on the abstractive summarisation task.



#### T5 (Text-to-Text Transfer Transformer)
- Pre-trained Transformer encoder-decoder unifying many tasks as the same text-to-text problem
− Input of the encoder is a task description (e.g., “Summarize: ”) followed by task input (e.g., a sequence of tokens from an article), and the decoder predicts the task output (e.g., a sequence of tokens summarizing the input article)
− T5 is trained to generate some target text conditional on input text to perform as text-to-text


#### BART
- Trained on reconstructing documents to which noise has been introduced
− The noise may take multiple forms, ranging from removing tokens to permuting sentences


#### PEGASUS
− specifically designed and pre-trained neural network for the automatic text summarisation task
− pre-training self-supervised objective - gap-sentence generation - for Transformer encoder-decoder models
− fine-tuned on 12 diverse summarisation datasets
− select and mask whole sentences from documents, and concatenate the gap sentences into a pseudo-summary.


### Comparative Summsarisation


|                | Method         | Max. Input     | Code           | Pre-Trained models    | Languages      |
|:--------------:|:--------------:|:--------------:|:--------------:|:---------------------:|:--------------:|
| Row 1, Col 1   | Row 1, Col 2   | Row 1, Col 3   | Row 1, Col 4   | Row 1, Col 5          | Row 1, Col 6   |
| Row 2, Col 1   | Row 2, Col 2   | Row 2, Col 3   | Row 2, Col 4   | Row 2, Col 5          | Row 2, Col 6   |
| Row 3, Col 1   | Row 3, Col 2   | Row 3, Col 3   | Row 3, Col 4   | Row 3, Col 5          | Row 3, Col 6   |
| Row 4, Col 1   | Row 4, Col 2   | Row 4, Col 3   | Row 4, Col 4   | Row 4, Col 5          | Row 4, Col 6   |
| Row 5, Col 1   | Row 5, Col 2   | Row 5, Col 3   | Row 5, Col 4   | Row 5, Col 5          | Row 5, Col 6   |
| Row 6, Col 1   | Row 6, Col 2   | Row 6, Col 3   | Row 6, Col 4   | Row 6, Col 5          | Row 6, Col 6   |
| Row 7, Col 1   | Row 7, Col 2   | Row 7, Col 3   | Row 7, Col 4   | Row 7, Col 5          | Row 7, Col 6   |
| Row 8, Col 1   | Row 8, Col 2   | Row 8, Col 3   | Row 8, Col 4   | Row 8, Col 5          | Row 8, Col 6   |
| Row 9, Col 1   | Row 9, Col 2   | Row 9, Col 3   | Row 9, Col 4   | Row 9, Col 5          | Row 9, Col 6   |




### Advantages and Disadvantages of each approach


__Extractive summarisation__
* Advantages
  * Language and domain agnostic
  * No need to account for model training
  * Generated summaries are grammatically and factually correct
* Disadvantages
  * Reduction of semantic quality and cohesion possible wrong connections between selected sentences
  * The generated summary may not cover all important content
  * longer than necessary, may contain unnecessary parts that may not be needed in the summary


__Abstractive summarisation__
* Advantages
  * Ideally a summary will be concise and capture the essence of the original text
  * Decrease the amount of generated text
  * Removes redundancy, obtaining a concise and expressive summary
* Disadvantages
  * The algorithm must have a deep understanding of the text
  * Use natural language generation technology
  * Needs a precise syntactic and semantic representation of text data
  * May contain factual errors due to the algorithm not grasping the text’s context well enough


#### Evaluation
* ROUGE (Recall-Oriented Understudy for Gisting Evaluation) - Recall
  * compare produced summary against reference summary by overlapping n-grams
  * ROUGE-N: report with different n-grams size
  * ROUGE-L: instead of n-gram overlap measures the Longest Common Subsequence
* BLEU (Bilingual Evaluation Understudy) - Precision
  * compare the reference summary against the produced summary
  * how much the words (and/or n-grams) in the reference appeared in the machine summary
* Limitations
  * always need a reference summary
  * just measuring string overlaps
  * alternative is to have a human evaluation


---

### __References__










