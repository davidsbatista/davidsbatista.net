---
layout: page
title: Resources
---

This page contains link to useful NLP resources: [NLP datasets](#datasets) and [books](#books). The datasets link to my [github page](https://github.com/davidsbatista) where I share datasets for named-entity recognition and relationship extraction tasks, plus some lexicons and dictionaries which I constructed during my PhD/Research time.

I also list some [books](#books) which I think are a good starting point to learn more about Natural Language Processing and how to apply Machine learning to NLP tasks. Most of this books and tutorials are nice to have around, so that you can quickly clarify any doubts or review how a certain algorithm or technique works. I personally like to have them at hand :)


---

# __Datasets__

#### __Relationship Extraction__

I've been keeping track of public and free datasets for semantic relationship extraction, this [github repository](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets) contains annotated datasets which can be used to train supervised models to perform semantic relationship extraction.

The datasets are organized into three different groups:

* [__Traditional Information Extraction__](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets/blob/master/README.md#tie)

* [__Open Information Extraction__](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets/blob/master/README.md#oie)

* [__Distantly Supervised__](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets/blob/master/README.md#ds)

---

#### __Named-Entity Recognition__

Named-Entity Recognition datasets organised by different languages, also some are for different domains:

* [__Portuguese__](https://github.com/davidsbatista/NER-datasets/tree/master/Portuguese)

* [__German__](https://github.com/davidsbatista/NER-datasets/blob/master/README.md#de)

* [__Dutch__](https://github.com/davidsbatista/NER-datasets/blob/master/README.md#nl)

* [__French__](https://github.com/davidsbatista/NER-datasets/blob/master/README.md#fr)

* [__English__](https://github.com/davidsbatista/NER-datasets/blob/master/README.md#en)

---

#### __Lexicons and Dictionaries__

Several [lexicons](https://github.com/davidsbatista/lexicons) I gathered for different NLP tasks, including lists of names, acronyms and it's extensions, stop-words, overlap of names and toponyms, etc.:

* [NomesLex-PT](https://github.com/davidsbatista/lexicons/blob/master/NomesLex-PT.zip) a lexicon of Portuguese person names made up of 2,027 first names and 8,019 surnames.

* [names-surnames-NL-UK-IT-PT-ES.zip](https://github.com/davidsbatista/lexicons/blob/master/names-surnames-NL-UK-IT-PT-ES.zip) a list of names and surnames for Dutch, English, Portuguese and Spanish.

* [publico-cargos.txt](https://github.com/davidsbatista/lexicons/blob/master/publico-cargos.txt) a list of Portuguese noun quantifiers, i.e., words that occur before a proper noun, gathered from the on-line newspaper publico.pt.

* [publico-acronyms.txt](https://github.com/davidsbatista/lexicons/blob/master/publico-acronyms.txt) a list of acronyms and it's possible extensions, extracted from a collection of Portuguese news gathered from the on-line newspaper publico.pt.

* [wikipedia-acronyms.txt](https://github.com/davidsbatista/lexicons/blob/master/wikipedia-acronyms.txt) a list of acronyms and it's possible extensions, extracted from the English Wikipedia.

* [PT-stopwords.txt](https://github.com/davidsbatista/lexicons/blob/master/PT-stopwords.txt) a collections of stop-words for Portuguese.


Learning about Natural Language Processing and Machine Learning is a continuous task, but there is always a starting point, where one learn the basics and the most common algorithms to solve some tasks.

Here I list some resources to learn more about Natural Language Processing, Machine learning and related areas. Most of this books/tutorials are things always nice to have around, and that you can quickly consult and clarify any doubts or review how certain algorithm or technique works.I personally like to have them at hand :)

<!--
https://my.fsf.org/donate
-->

<!--
https://www.flipcause.com/secure/supporters/MjM2OA==
-->

<!--
https://www.apache.org/foundation/contributing.html
-->

---



# __Books__

Learning about Natural Language Processing is a continuous task, which appears not to have an end, but, there is always a starting point where one learns the problem definitions and the common algorithms to solve them.

#### [__Neural Network Methods for Natural Language Processing__](https://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037)

<p>
<a href="https://www.morganclaypool.com/doi/abs/10.2200/S00762ED1V01Y201703HLT037"><img style="float: left; margin: 0px 15px 15px 0px;" alt="Neural Network Methods for Natural Language Processing" src="/assets/images/resources/neural_networks_for_nlp.jpg" height="220" width="165"></a>
I would recommend this book to anyone who is not a beginner and wants to make the jump from "classic NLP", i.e.: exploring hand-made features together with SVMs, HMM, CRFs and alike and wants to have a overview of how neural networks can be applied to several NLP tasks. It takes a very good overview on how neural networks took over NLP from the 2012/2013 up until 2016. It first covers the basics on neural networks and then slowly goes explains how they can be used to solve different NLP tasks, starting with more simple ones and going into specific ones, it contains many references to papers that at the time made the first breakthroughs on using neural networks for NLP.
</p>

---

#### [__Deep Learning with Python__](https://www.manning.com/books/deep-learning-with-python)

<p>
<a href="https://www.manning.com/books/deep-learning-with-python">
<img style="float: left; margin: 0px 15px 15px 0px;" alt="Deep Learning with Python" src="/assets/images/resources/Chollet-DLP-HI.png" height="220" width="150"></a>

This is a hands-on book on Keras written by the creator of Keras himself François Chollet. It starts with a quick overview over what Machine Learning explaining the basics in such a way that the non-mathematical person can understand the concepts behind it. I then continues through with examples of simple regressions problems and also binary and multi-class classification.
It contains then full chapters dedicated to dedicated to image and another dedicated to text and sequence processing, and also more advanced topics such as neural network fine tuning and generative models. I would suggest this book to programmers who want to get started with neural networks, specially using Keras.
</p>

---

#### [__Deep Learning__](https://www.deeplearningbook.org/)

<p>
<a href="https://www.deeplearningbook.org/">
<img style="float: left; margin: 0px 15px 15px 0px;" alt="Deep learning" src="/assets/images/resources/deep_learning.jpg" height="200" width="150"></a>
I would dare to say that this is the current neural networks "bible", it goes into deep detail on every aspect of Machine Learning and more specifically on Neural Networks. It really starts from scratch, the first four chapters are entirely dedicated to topics such as Linear Algebra, Machine Learning concepts. Then it details neural networks, starting with forward neural networks, and introducing the concepts of regularization and optimization on neural networks, then it describes the different architectures, i.e: convolutional networks and sequence modeling. The final part of the book goes into advanced topics on deep learning, such like auto-encoders and deep generative models. This book goes in very detail explaining every architecture of different neural networks. If you really want to go in the details this is the one.
</p>



---

#### [__Foundations of Natural Language Processing__](https://nlp.stanford.edu/fsnlp/)


<p>
<a href="https://nlp.stanford.edu/fsnlp/">
<img style="float: left; margin: 0px 15px 15px 0px;" alt="Foundations of Natural Language Processing" src="/assets/images/resources/foundations_of_statistical_nlp.jpg" height="200" width="150"></a>
This is book is the classic :) It was published in 1999 so all the content is before the Machine Learning hype during the 2000s and later (and currently) the Deep Learning hype. All the methods are very clearly detailed and explained. It starts with simple but very important concepts related with word counts, collocations, Zipf's Law, etc. It dives into mathematical foundations around probability and information theory and shows how one can use Markov Models for sequence tagging. It also introduces the concepts of Probabilistic Context Free Grammars and Parsing.
</p>

<br>

---

#### [__Introduction to Information Retrieval__](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)

<p>
<a href="https://nlp.stanford.edu/IR-book/information-retrieval-book.html"><img style="float: left; margin: 0px 15px 15px 0px;" alt="Introduction to Information Retrieval" src="/assets/images/resources/information_retrieval.jpg" height="200" width="150"></a>
I started my journey on NLP and Machine Learning back in 2007 with Information Retrieval, more precisely Geographic Information Retrieval. This book is a great start before you  jump into more advanced topics. It covers subjects related to indexing and retrieving documents and goes into great detail on the algorithms and techniques behind the scenes on a document search engine. It also covers some machine learning algorithms like Naïve Bayes and Support Vector Machines, and how they can be used for document retrieval.
</p>

<br>

---

#### [__The Elements of Statistical Learning__](http://web.stanford.edu/~hastie/ElemStatLearn/)

<p>
<a href="http://web.stanford.edu/~hastie/ElemStatLearn/">
<img style="float: left; margin: 0px 15px 15px 0px;" alt="The Elements of Statistical Learning" src="/assets/images/resources/elements_of_statistical_learning.jpg" height="200" width="150"></a>

This is, I would say one of the main reference for machine learning from the statistical point of view. The book is also kind of a classic going in it's 2nd Edition. It's considered a very complete reference book, one of the authors is Prof. Tibshirani, who proposed the LASSO regularization technique. It uses some mathematical statistics avoiding complicated proofs to explain machine learning concepts. It describes popular machine algorithms such as Logistic Regressions, SVMs, Random Forests, etc.; but each is developed only after the appropriate statistical framework has already been introduced.
</p>

<br>

---

#### [__Speech and Language Processing__](https://web.stanford.edu/~jurafsky/slp3/)

<p>
<a href="https://web.stanford.edu/~jurafsky/slp3/">
<img style="float: left; margin: 0px 15px 15px 0px;" alt="Speech and Language Processing" src="/assets/images/resources/speech_and_lang_process.jpg" height="200" width="150"></a>

This is another very complete book which goes well beyond theory by also presenting some practical examples, one can see this book as an extension to the <i>Foundations of Natural Language Processing</i>. It contains besides the description of statistical methods applied to NLP, also some chapters on Phonology, Formal Grammars, Parsing and other Language related topics. It also contains, as the title suggests chapters dedicated only to Speech, which might be interesting for these interested in speech related applications, such as text-to-speech. I would say this is an interesting book for those also interested in the computational linguistics aspects of Natural Language Processing.

</p>