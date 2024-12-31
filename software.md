---
layout: page
title: Software
---

# __Software__


#### __BREDS__ ‑ Bootstrapping of Relationship Extractors with Distributional Semantics

A Python package implementation based on results from my Ph.D. thesis. BREDS is an approach to extract named‑entity relationships without labelled data by relying instead on an initial set of seeds, i.e. pairs of named entities representing the relationship type to be extracted. The algorithm uses the seeds to learn extraction patterns and expands the initial set of seeds using distributional semantics to generalise the relationship while limiting the semantic drift.

__[GitHub](https://github.com/davidsbatista/breds){:target="_blank"}__
&nbsp;
__[PyPI Package](https://pypi.org/project/breds){:target="_blank"}__


---

#### __nervaluate__ ‑ NER Evaluation Considering Partial Matching

An open‑source software package to evaluate named‑entity recognition systems considering partial entity matching. Originally started with a blog post I wrote about the subject which attracted the interest of several people and converged into a Python package which is currently maintained by myself and other contributors.

__[GitHub](https://github.com/MantisAI/nervaluate/){:target="_blank"}__
&nbsp;
__[PyPI Package](https://pypi.org/project/nervaluate){:target="_blank"}__


---

#### __snowball-extractor__ ‑ Extracting Relations from Large Plain-Text Collections

An open‑source software package to evaluate named‑entity recognition systems considering partial entity matching. Originally started with a blog post I wrote about the subject which attracted the interest of several people and converged into a Python package which is currently maintained by myself and other contributors.

__[GitHub](https://github.com/davidsbatista/snowball){:target="_blank"}__
&nbsp;
__[PyPI Package](https://pypi.org/project/snowball-extractor/){:target="_blank"}__


---


#### __Politiquices.PT__ ‑ Support and Opposition Relationships in Portuguese Political News Headlines




I've analysed thousands of archived titles, identifying those that report supportive or opposing relationships between political actors and also associated the political personalities with their identifier on Wikidata. The result was a semantic graph,  [__politiquices.pt__](http://www.politiquices.pt) allowing answering questions involving political personalities and parties. The project was awarded [__2nd place in the "Arquivo.pt Awards 2021"__](https://sobre.arquivo.pt/en/meet-the-winners-of-the-arquivo-pt-award-2021/).


__[Web](http://www.politiquices.pt/){:target="_blank"}__
&nbsp;
__[GitHub](https://github.com/politiquices){:target="_blank"}__


<br>

---


<br>


# __Datasets__

#### __Relationship Extraction__

I've been keeping track of [__public and free datasets__](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets) for semantic relationship extraction. The datasets are organised into three different groups:

* [__Traditional Information Extraction__](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets/blob/master/README.md#tie)

* [__Open Information Extraction__](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets/blob/master/README.md#oie)

* [__Distantly Supervised__](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets/blob/master/README.md#ds)

---

#### __Named-Entity Recognition__

Named-Entity Recognition datasets are organised by different languages, also some are for different domains:

* [__Portuguese__](https://github.com/davidsbatista/NER-datasets/tree/master/Portuguese)

* [__German__](https://github.com/davidsbatista/NER-datasets/blob/master/README.md#de)

* [__Dutch__](https://github.com/davidsbatista/NER-datasets/blob/master/README.md#nl)

* [__French__](https://github.com/davidsbatista/NER-datasets/blob/master/README.md#fr)

* [__English__](https://github.com/davidsbatista/NER-datasets/blob/master/README.md#en)

---

#### __Lexicons and Dictionaries__

Several [__lexicons__](https://github.com/davidsbatista/lexicons) I gathered for different NLP tasks, including lists of names, acronyms and their extensions, stop-words, overlap of names and toponyms, etc.:

* [__NomesLex-PT__](https://github.com/davidsbatista/lexicons/blob/master/NomesLex-PT.zip) a lexicon of Portuguese person names made up of 2,027 first names and 8,019 surnames.

* [__names-surnames-NL-UK-IT-PT-ES.zip__](https://github.com/davidsbatista/lexicons/blob/master/names-surnames-NL-UK-IT-PT-ES.zip) a list of names and surnames for Dutch, English, Portuguese and Spanish.

* [__publico-cargos.txt__](https://github.com/davidsbatista/lexicons/blob/master/publico-cargos.txt) a list of Portuguese noun quantifiers, i.e., words that occur before a proper noun, gathered from the on-line newspaper publico.pt.

* [__publico-acronyms.txt__](https://github.com/davidsbatista/lexicons/blob/master/publico-acronyms.txt) a list of acronyms and its possible extensions, extracted from a collection of Portuguese news gathered from the on-line newspaper publico.pt.

* [__wikipedia-acronyms.txt__](https://github.com/davidsbatista/lexicons/blob/master/wikipedia-acronyms.txt) a list of acronyms and its possible extensions, extracted from the English Wikipedia.

* [__PT-stopwords.txt__](https://github.com/davidsbatista/lexicons/blob/master/PT-stopwords.txt) a collections of stop-words for Portuguese.
