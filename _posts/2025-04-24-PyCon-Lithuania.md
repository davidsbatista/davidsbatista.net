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


- Return a list of documents or snippets, requiring users to read through multiple results to find the information they need
- A complex or nuanced query requires a deeper understanding of the context and relationships between different pieces of information 6
- What if, instead the user sifting through the results, we build a prompt composed by retrieved snippets together with the query and feed it to an LLM? 



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


## __Classical Techniques__


### __Sentence-Window Retrieval__


● Retrieve the chunks before and after the matching chunk

● A simple way to gather more context

● Indexing needs to preserve the order of the chunks 15


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




## __LLLM-based Techniques__


### __Multi-Query__

**User Query**

**LLM Query 1**

**LLM Query 2**

**LLM Query 3**

**LLM**

● Expand a user query into *n* similar queries reflecting the original intent

● ..or break-down a complex query into individual questions 30





**Multi-Query**


**LLM**


● Expand a user query into *n* similar queries reflecting the original intent

● ..or break-down a complex query into individual questions

● Each new query is used for an individual retrieval processes 31





**Multi-Query**


**User Query**



**LLM Query 1**

chunk\_7

chunk\_1

**LLM Query 2**

chunk\_9

**ReRanking**

chunk\_3

**LLM Query 3**

chunk\_8

chunk\_6

…

chunk\_5

**LLM**

chunk\_6

chunk\_3

chunk\_4

chunk\_5

● Expand a user query into *n* similar queries reflecting the original intent

● ..or break-down a complex query into individual questions

● Each new query is used for an individual retrieval processes

● Re-ranking process over all retrieved chunks 32


### __Hypothetical Document Embeddings - HyDE__

**User Query**

Doc

Doc

Doc

**LLM**

● Given a user query, use a LLM to generate *n* "hypothetical" \(short\) documents whose content would ideally answer the query 33



**Hypothetical Document Embeddings - HyDE**

**User Query**

Doc

vector\_1

Doc

**Embedder**

vector\_2

Doc

vector\_3

**LLM**

● Given a user query, use a LLM to generate *n* "hypothetical" \(short\) documents whose content would ideally answer the query

● Each of the *n* documents is embedded into a vector 34





**Hypothetical Document Embeddings - HyDE**

**User Query**

Doc

vector\_1

Doc

**Embedder**

vector\_2

query\_vector

**Avg. Pool**

Doc

vector\_3

**LLM**

● Given a user query, use a LLM to generate *n* "hypothetical" \(short\) documents whose content would ideally answer the query

● Each of the *n* documents is embedded into a vector

● You perform an average pooling generating a new query embedding used to search for similar documents instead of the original query 35


### __Document Summary Indexing__

Doc

**LLM**

summary

**Summary **

**Index**

● **Summary Index: **generate a summary for each document with an LLM

36





**Document Summary Indexing**

chunks

**Chunker**

…

Doc

**LLM**

summary

**Summary **

**Chunk **

**Index**

**Index**

● **Summary Index: **generate a summary for each document with an LLM

● **Chunk Index: **split each document up into chunks 37





**Document Summary Indexing**

chunks

**Chunker**

…

Doc

**LLM**

summary

Doc 

ref. 

**Summary **

**Chunk **

**Index**

**Index**

**User Query**

● **Summary Index: **generate a summary for each document with an LLM

● **Chunk Index: **split each document up into chunks 

● Use the **Summary Index** to retrieve top-k relevant documents to the query 38





**Document Summary Indexing**

chunks

**Chunker**

…

chunks

…

Doc

**LLM**

summary

Doc 

ref. 

chunks

…

**Summary **

**Chunk **

**Index**

**Index**

**User Query**

● **Summary Index: **generate a summary for each document with an LLM

● **Chunk Index: **split each document up into chunks 

● Use the **Summary Index** to retrieve top-k relevant documents to the query

● Using the document\(s\) reference retrieve the most relevant chunks 39



## __Summary__

Custom Index 

ReRanking

Query Rewriting

Combining

Relies on a LLM

Structure

multiple sources

Sentence-Window Retrieval

X

Auto-Merging Retrieval

X

Maximum Margin Relevance

X

Hybrid Retrieval

X

Multi-Query

X

X

Hypothetical Document Embeddings

X

X

Document Summary Indexing

X

X

X

40



## __Comparative Experiment__

● "ARAGOG: Advanced RAG Output Grading" M Eibich, S Nagpal, A Fred-Ojala arXiv 

preprint, 2024

● **Dataset:**

○ ArXiv preprints covering topics around Transformers and LLMs

○ 13 PDF papers \(https://huggingface.co/datasets/jamescalam/ai-arxiv\)

○ 107 questions and answers generated with the assistance of an LLM

○ All questions and answers were manually validated and corrected

**● Experiment:**

○ Run the questions over each retrieval technique

○ Compare ground-truth answer with generated answer

○ Semantic Answer Similarity: cos sim embeddings of both answers 41





**Comparative Experiment: ARAGOG**

**Semantic Answer Similarity**

**Specific Parameters**

**Sentence-Window Retrieval**

0.688

window=3

**Auto-Merging Retrieval**

0.619

threshold=0.5, block\_sizes=\{10, 5\}

**Maximum Margin Relevance**

0.607

lambda\_threshold=0.5

**Hybrid Retrieval**

0.701

join\_mode=”concatenate” 

**Multi-Query**

0.692

n\_variations=3

**Hypothetical Document Embeddings**

0.642

nr\_completions=3

**Document Summary Indexing**

0.731

-

●

sentence-transformers/all-MiniLM-L6-v2

**LLM**

●

chunk\_size = 15

●

split\_by = "sentence" 

OpenAI: gpt-4o-mini

●

top\_k = 3

42



## __Takeaways__

● Build a dataset for our use case - **50~100 annotated questions**

● Start with the simple RAG approach and set it as your baseline

● Start by exploring “cheap” and simple techniques

● **Sentence-Window Retriever** and **Hybrid Retrieval - **good results and no need for complexing indexing or an LLM

● If none of these produces satisfying results then, explore indexing/retrieval methods based on LLMs

43



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



**"The use of MMR, diversity-based reranking for reordering documents and producing summaries" J Carbonell, J Goldstein - ACM SIGIR 1998**

**●**

**"ARAGOG: Advanced RAG Output Grading" M Eibich, S Nagpal, A Fred-Ojala arXiv preprint, 2024**

**●**

**" ****Advanced RAG: Query Expansion” - Haystack Blog, 2024**

●

**”Advanced RAG: Query Decomposition & Reasoning” ****- Haystack Blog, 2024**

●

**“Precise Zero-Shot Dense Retrieval without Relevance Labels” Luyu Gao, **

**Xueguang Ma, Jimmy Lin, and Jamie Callan- * ACL 2023***

**●**

**"A New Document Summary Index for LLM-powered QA Systems", Jerry Liu 2023 **





**Code and experiments**

**Haystack**

**Code \+ Slides**

**www.davidsbatista.net**

47



48



**Comparative Experiment**

● HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering Zhilin 

Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan 

Salakhutdinov, Christopher D. Manning

● **Dataset:**

○ Question-Answering dataset over Wikipedia articles

○ Very short answers, e.g.: 

■ *'Who is older Glenn Hughes or Ross Lynch?' *

*■ 'Are Gin and tonic and Paloma both cocktails based on tequila?' *

○ Select the first 100 questions from hotpot\_train\_v1.1.json

**● Experiment:**

○ Evaluate the retrievers: Precision, Recall and Fallout

○ Evaluate generated answer: Semantic Answer Similarity 49





**Comparative Experiment: HotpotQA**

**Recall**

**Precision**

**Fall-out**

**Semantic Answer **

**Parameters**

**Similarity**

**Sentence-Window Retrieval**

0.73

0.49

0.51

0.688

window=3

**Auto-Merging Retrieval**

0.36

0.26

0.74

0.376

0

**Maximum Margin Relevance**

0.73

0.48

0.52

0.500

lambda\_threshold=0.75

**Hybrid Retrieval**

0.81

0.40

0.60

0.512

join\_mode=”concatenate” 

**Multi-Query**

0.75

0.36

0.64

0.515

n\_variations=3

**Hypothetical Document Embeddings**


**Document Summary Indexing**



sentence-transformers/all-MiniLM-L6-v2

**LLM**

●

chunk\_size = 15

●

split\_by = "sentence" 

OpenAI: gpt-4o-mini

●

top\_k = 3

50





**Maximum Marginal Relevance \(MMR\)**

51



**Number of LLM calls**

● **Multi-Query:**

○ Each query results in 2 LLM calls

**● Hypothetical Document Embeddings**

**○ **Each query results in 2 LLM calls

**● Document Summary Indexing**

**○ **Dependent on the detail parameter

○ LLM\_calls = 1 \+ detail \* \(X / minimum\_chunk\_size - 1\) 52



