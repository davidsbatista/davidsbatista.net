---
layout: post
title: Portuguese Word Embeddings
date: 2019-11-03 00:00:00
tags: embeddings word-embeddings gensim fasttext word2vec portuguese
categories: [blog]
comments: true 
disqus_identifier: 20191103
preview_pic: /assets/images/2019-11-03-pt-embeddings.jpg
---

While working on some projects of mine I come to a point where I needed pre-trained
word embeddings for Portuguese. I could have trained some on my own on some corpora
but I did not wanted to spent time on cleaning and running the training, so instead
I searched the web for collections of word vectors for Portuguese, here's a compiled
list of what I've found.


### __NILC-Embeddings (2017)__

A very comprehensive evaluation of different methods and parameters to generate
word embeddings for both Brazilian and European variants. In total 31 word embedding
models based on FastText, GloVe, Wang2Vec and Word2Vec, evaluated intrinsically on
syntactic and semantic analogies and extrinsically on POS tagging and sentence
semantic similarity tasks.

* __Domain__: Mixed (News, Wiki, Subtitles, literay works, etc.)
* __Methods__: FastText, GloVe, Wang2Vec and Word2Vec
- :pencil:  ["Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language \[...\]"](https://www.aclweb.org/anthology/W17-6615.pdf)
- :floppy_disk: [Download](http://nilc.icmc.usp.br/embeddings)
- :computer: [Code](https://github.com/nathanshartmann/portuguese_word_embeddings)



---

<br>


### __LX-DSemVectors (2018)__

The authors apply the Skip-Gram model to a dataset composed of mostly European
Portuguese news papers. I would say that if you want embeddings for the new domain
in European Portuguese this is probably a very good choice.

* __Domain__: News Articles
* __Method__: Skip-Gram
- :pencil: ["Finely Tuned, 2 Billion Token Based Word Embeddings for Portuguese"](http://www.di.fc.ul.pt/~ahb/pubs/2018RodriguesBranco.pdf)
- :floppy_disk: [Download](http://lxcenter.di.fc.ul.pt/datasets/models/)
- :computer: [Code](https://github.com/nlx-group/LX-DSemVectors)

---

<br>


### __Facebook fasttext (2018)__

This is the famous dataset published by Facebook research containing word embeddings
trained on the Wikipedia and Common Crawl data. It contains Portuguese among a
total of 157 languages.

* __Methods__: FastText, GloVe, Wang2Vec and Word2Vec
* __Domain__: Wikipedia + Common Crawl
- :pencil: ["Learning Word Vectors for 157 Languages"](http://www.lrec-conf.org/proceedings/lrec2018/pdf/627.pdf)
- :floppy_disk: [Download](https://fasttext.cc/docs/en/crawl-vectors.html)
- :computer: [Code](https://github.com/facebookresearch/fastText)

---

<br>


### __Wikipedia2Vec (2018)__

Unlike other word embedding tools, this software package learns embeddings of
entities as well as words, the method jointly maps words and entities into the
same continuous vector space. They provide such embeddings for 11 Languages,
including Portuguese.


* __Methods__: Word2Vec
* __Domain__: Wikiepdia
- :pencil:  ["Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation"](https://www.aclweb.org/anthology/K16-1025.pdf)
- :floppy_disk: [Download](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)
- :computer: [Code](https://github.com/wikipedia2vec/wikipedia2vec)



---

<br>

### __NLPL word embeddings repository__

From the paper "shared repository of large-text resources for creating word
vectors, including pre-processed corpora and pre-trained vectors for a range
of frameworks and configurations. This will facilitate reuse, rapid
experimentation, and replicability of results". The repository contains different
types of embedding for many languages, including embeddings based on the
Portuguese CoNLL17 corpus.

* __Methods__: Several
* __Domain__: Several
- :pencil: ["Word vectors, reuse, and replicability: Towards a community repository \[...\]"](https://www.aclweb.org/anthology/W17-0237/)
- :floppy_disk: [Download](http://vectors.nlpl.eu/repository/)

