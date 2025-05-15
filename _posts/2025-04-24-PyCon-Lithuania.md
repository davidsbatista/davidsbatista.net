---
layout: post
title: Retrieval methods in RAG - Haystack
date: 2025-04-24 00:00:00
tags: haystack information-retrieval RAG 
categories: [blog]
comments: true
disqus_identifier: 2025024
preview_pic: /assets/images/2025-04-24-PyCon-Lithuania.png
---

Retrieval Augmented Generation (RAG) is a model architecture for tasks requiring information retrieval from large corpora combined with generative models to fulfill a user information need. It's typically used for question-answering, fact-checking, summarization, and information discovery.

The RAG process consists of indexing, which converts textual data into searchable formats; retrieval, which selects relevant documents for a query using different methods; and augmentation, which feeds retrieved information and the user's query into a Large Language Model (LLM) via a prompt for output generation.

Typically, one has little control over the augmentation step besides what's provided to the LLM via the prompt and a few parameters, like the maximum length of the generated text or the temperature of the sampling process. On the other hand, the indexing and retrieval steps are more flexible and can be customized to the specific needs of the task or the data.


In this blog post I will different retrieval techniques, some rooted in the area of classic information retrieval others, which were proposed recently, and based on LLMs.


## __From Classic Information Systems to RAG__

<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-IR-to-RAG_1.png">
  <figcaption>Figure 1 - Classical Information System.</figcaption>
</figure>

- Classical Information Retrieval system returns a list of documents or snippets
- Requiring users to read through multiple results to find the information they need
- A complex or nuanced query requires a deeper understanding of the context and relationships between different pieces of information

<br>

<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-IR-to-RAG_2.png">
  <figcaption>Figure 2 - Retrieval Augmented Generation system.</figcaption>
</figure>

- What if, instead the user sifting through the results, we build a prompt composed by retrieved snippets together with the query and feed it to an LLM? 


<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-RAG.png">
  <figcaption>Figure 3 - Retrieval Augmented Generation system.</figcaption>
</figure>


### __Baseline Retrieval__


**User Query**

● **Indexing**: split documents into chunks and index in a vector db

● **Query**: retrieve chunks 

a. embedding similarity with query

b. using query as keyword filter

● **Ranking**: rank by similarity with the query 11



**Outline**

1. Classic Retrieval Techniques

2. LLM-based Retrieval Techniques

3. Comparative Summary

4. Experiment





**Sentence-Window Retrieval**


…

**User Query**

● Retrieve the chunks before and after the matching chunk 14

<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-Baseline-Retrieval-System.png">
  <figcaption>Figure 4 - Baseline RAG system.</figcaption>
</figure>



## __Classical Techniques__




### __Sentence-Window Retrieval__


● Retrieve the chunks before and after the matching chunk

● A simple way to gather more context

● Indexing needs to preserve the order of the chunks 15

<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-Sentence-Window-Retrieval.png">
  <figcaption>Figure 5 - Sentence Window Retrieval.</figcaption>
</figure>



### __Auto-Merging Retrieval__

● Transform documents into an Hierarchical Tree structure 16

**Index**

● Transform documents into an Hierarchical Tree structure

● Children chunks/sentences are index and used for retrieval 17





**Index**

● With a threshold of 0.5

- The paragraph\_1 is returned, instead of 4 sentences

- Plus, the one sentence from paragraph\_2

● A whole paragraph might be more informative than individual chunks 18


<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-Auto-Merging-Retrieval.png">
  <figcaption>Figure 6 - Auto-Merging Retrieval.</figcaption>
</figure>


### __Maximum Marginal Relevance \(MMR\)__

● Classical retrieval ranks the retrieved documents by relevance similarity to the user query

● What about scenarios with a high number of relevant documents, but also highly redundant or containing partially or fully duplicative information? 

● We need to consider how novel is a document compared to the already retrieved docs




**Maximum Marginal Relevance \(MMR\)**

Each retrieved document is scored




**Maximum Marginal Relevance \(MMR\)**

Each retrieved document is scored:

- Similarity between a candidate document and the query 21





**Maximum Marginal Relevance \(MMR\)**

Each retrieved document is scored:

- Find maximum similarity between a candidate document and any previously selected document. 

- Maximize the similarity to already selected documents and then subtracting it - penalize documents that are too similar to what's already been selected. 

22





**Maximum Marginal Relevance \(MMR\)**

Each retrieved document is scored:

- Similarity between the candidate document and the query

- Find maximum similarity between a candidate document and any previously selected document. 

- Maximize the similarity to already selected documents and then subtracting it - penalize documents that are too similar to what's already been selected. 

- **λ** balances between these two terms 23





**Maximum Marginal Relevance \(MMR\)**

Each retrieved document is scored:

- Similarity between the candidate document and the query

- Find maximum similarity between the candidate document and any previously selected document. By maximizing the similarity to already selected documents and then subtracting it, we penalize documents that are too similar to what's already been selected. 

- **λ** balances between these two terms 24


<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2024-11-17-ml-report-toolkit.png">
  <figcaption>Figure 7 - Precision Recall Curve and a Confusion Matrix.</figcaption>
</figure>


### __Hybrid Retrieval \+ Reranking__


**Hybrid Retrieval \+ Reranking**

**chunk\_2**

Top-3

**chunk\_9**

**chunk\_3**

**BM25**

**User Query**

**chunk\_7**

Top-3

**chunk\_8**

**chunk\_5**

**Embedding Vector DB**

● Combines multiple search techniques

● keyword-based \(BM25\) and semantic-based \(embedding vector\) 26





**Hybrid Retrieval \+ Reranking**



Top-3

**chunk\_8**

**chunk\_5**

**Embedding Vector DB**

● Combines multiple search techniques

● \(BM25\) and semantic-based \(embedding vector\) keyword-based \(BM25\) and semantic-based \(embedding vector\)

● Rank-merge results

27


<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-Hybrid-Retrieval.png">
  <figcaption>Figure 8 - Hybrid Retrieval.</figcaption>
</figure>



## __LLLM-based Techniques__


### __Multi-Query__


● Expand a user query into *n* similar queries reflecting the original intent

● ..or break-down a complex query into individual questions 30

● Expand a user query into *n* similar queries reflecting the original intent

● ..or break-down a complex query into individual questions

● Each new query is used for an individual retrieval processes 31


● Expand a user query into *n* similar queries reflecting the original intent

● ..or break-down a complex query into individual questions

● Each new query is used for an individual retrieval processes

● Re-ranking process over all retrieved chunks 32

<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-Multi-Query-Retrieval.png">
  <figcaption>Figure 9 - Precision Recall Curve and a Confusion Matrix.</figcaption>
</figure>


### __Hypothetical Document Embeddings - HyDE__

● Given a user query, use a LLM to generate *n* "hypothetical" \(short\) documents whose content would ideally answer the query 33

● Given a user query, use a LLM to generate *n* "hypothetical" \(short\) documents whose content would ideally answer the query

● Each of the *n* documents is embedded into a vector 34

● Given a user query, use a LLM to generate *n* "hypothetical" \(short\) documents whose content would ideally answer the query

● Each of the *n* documents is embedded into a vector

● You perform an average pooling generating a new query embedding used to search for similar documents instead of the original query 35

<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-HyDE.png">
  <figcaption>Figure 10 - Precision Recall Curve and a Confusion Matrix.</figcaption>
</figure>



### __Document Summary Indexing__


● **Summary Index: **generate a summary for each document with an LLM

● **Summary Index: **generate a summary for each document with an LLM

● **Chunk Index: **split each document up into chunks 37


● **Summary Index: **generate a summary for each document with an LLM

● **Chunk Index: **split each document up into chunks 

● Use the **Summary Index** to retrieve top-k relevant documents to the query 38

● **Summary Index: **generate a summary for each document with an LLM

● **Chunk Index: **split each document up into chunks 

● Use the **Summary Index** to retrieve top-k relevant documents to the query

● Using the document\(s\) reference retrieve the most relevant chunks 39

<figure>
  <img style="width: 95%; height: 50%" src="/assets/images/2025-04-24-Document-Summary-Indexing.png">
  <figcaption>Figure 11 - Precision Recall Curve and a Confusion Matrix.</figcaption>
</figure>




## __Summary__


## __Comparative Experiment__

● "ARAGOG: Advanced RAG Output Grading" M Eibich, S Nagpal, A Fred-Ojala arXiv preprint, 2024

● **Dataset:**

○ ArXiv preprints covering topics around Transformers and LLMs

○ 13 PDF papers \(https://huggingface.co/datasets/jamescalam/ai-arxiv\)

○ 107 questions and answers generated with the assistance of an LLM

○ All questions and answers were manually validated and corrected

**● Experiment:**

○ Run the questions over each retrieval technique

○ Compare ground-truth answer with generated answer

○ Semantic Answer Similarity: cos sim embeddings of both answers 41





## __Takeaways__

● Build a dataset for our use case - **50~100 annotated questions**

● Start with the simple RAG approach and set it as your baseline

● Start by exploring “cheap” and simple techniques

● **Sentence-Window Retriever** and **Hybrid Retrieval - **good results and no need for complexing indexing or an LLM

● If none of these produces satisfying results then, explore indexing/retrieval methods based on LLMs


**Haystack Implementations**

Sentence-Window Retrieval

**haystack.components.retrievers.SentenceWindowRetriever**

Auto-Merging Retrieval

**haystack.components.retrievers.AutoMergingRetriever**

**haystack.components.preprocessors.HierarchicalDocumentSplitter**

Maximum Margin Relevance

**haystack.components.rankers.SentenceTransformersDiversityRanker**

Hybrid Retrieval w/ ReRanking

**haystack.components.retrievers.InMemoryEmbeddingRetriever**

**haystack.components.retrievers.InMemoryBM25Retriever**

**haystack.components.joiners.DocumentJoiner** \(ranking techniques\) Multi-Query

**https://github.com/davidsbatista/haystack-retrieval**

**https://haystack.deepset.ai/blog/query-expansion**

**https://haystack.deepset.ai/blog/query-decomposition**

Hypothetical Document Embeddings

**https://haystack.deepset.ai/blog/optimizing-retrieval-with-hyde**

Document Summary Indexing

**https://github.com/davidsbatista/haystack-retrieval**

44



**Haystack Implementations - SuperComponents** Sentence-Window Retrieval

**haystack.components.retrievers.SentenceWindowRetriever**

Auto-Merging Retrieval

**haystack.components.retrievers.AutoMergingRetriever**

**haystack.components.preprocessors.HierarchicalDocumentSplitter**

Maximum Margin Relevance

**haystack.components.rankers.SentenceTransformersDiversityRanker**

Hybrid Retrieval w/ ReRanking

**Will be soon available as a SuperComponent\! **

Multi-Query

**Can be built as a SuperComponent\! **

Hypothetical Document Embeddings

**Can be built as a SuperComponent\! **

Document Summary Indexing

**Can be built as a SuperComponent\! **

**SuperComponent in Haystack 2.12.0\! **

- wrap complex pipelines into reusable \(super\)components

- easy to reuse them across applications

- Initialize a SuperComponent with a pipeline 45



## __References__ ##

- "The use of MMR, diversity-based reranking for reordering documents and producing summaries" J Carbonell, J Goldstein - ACM SIGIR 1998**

- "ARAGOG: Advanced RAG Output Grading" M Eibich, S Nagpal, A Fred-Ojala arXiv preprint, 2024**

- ****Advanced RAG: Query Expansion” - Haystack Blog, 2024**

- ”Advanced RAG: Query Decomposition & Reasoning” ****- Haystack Blog, 2024**

- “Precise Zero-Shot Dense Retrieval without Relevance Labels” Luyu Gao, ****Xueguang Ma, Jimmy Lin, and Jamie Callan- * ACL 2023***

- "A New Document Summary Index for LLM-powered QA Systems", Jerry Liu 2023 **

**Code and experiments**
