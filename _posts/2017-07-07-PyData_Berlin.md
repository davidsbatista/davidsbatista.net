---
layout: post
title: PyData Berlin 2017
date: 2017-07-07 00:00:00
tags: [pydata, berlin, python, data, conference]
categories: [blog]
comments: true
disqus_identifier: 20170707
preview_pic: /assets/images/2017-07-07-PyData_Berlin.png
---

The [PyData Berlin](https://pydata.org/berlin2017/) conference took place in the first weekend July, at the HTW. Full 3 days of many interesting subjects including Natural Language Processing, Machine Learning, Data Visualization, etc. I was happy to have my talk proposal accepted, and had the opportunity to present work done during my PhD on Semantic Relationship extraction.

<figure>
  <img style="width: 45%; height: 45%" src="/assets/images/2017-07-07-PyData_Berlin.png">
</figure>

#### Evaluating Topic Models

[Matti Lyra](https://github.com/mattilyra) gave an interesting talk on how to evaluate unsupervised models, mostly focused on how to evaluate topic models generated from LDA. He went through the proposed evaluation methodologies and described the pros and cons of each one of them. Slides are available [here](https://github.com/mattilyra/pydataberlin-2017)

[![Evaluating Topic Models](https://img.youtube.com/vi/UkmIljRIG_M/hqdefault.jpg)](https://www.youtube.com/watch?v=UkmIljRIG_M)

---

#### AI assisted creativity

[Roelof Pieters](https://twitter.com/graphific) from [creative.ai](creative.ai) presented a talk about AI assisted creativity, mainly focusing on how generative models based on neural networks can be used to generate text, audio, speech, and images. The talk was focused on how to use these models has an augmentation tool and not an automation tool. It also presented a good history and a time-line on machine learning assisted art generation.

[![AI assisted creativity](https://img.youtube.com/vi/1XmCz3HFU9I/hqdefault.jpg)](https://www.youtube.com/watch?v=1XmCz3HFU9I)

---

#### Developments in Test-Driven Data Analysis

Nick Radcliffe presented something which is new to me and grab my attention: Test-driven data analysis (TDDA), inspired in test-driven development. The TDDA library allows keep you data clean and to automatically discover constraints and validation rules. [TDDA web site](http://www.tdda.info/) and [GitHub project page](https://github.com/tdda/). Slides are available [here](https://github.com/pydataberlin/conf2017slides/blob/master/tdda-pydata-berlin-2017.pdf).

[![Developments in Test-Driven Data Analysis](https://img.youtube.com/vi/oBXO-dxF0Gg/hqdefault.jpg)](https://www.youtube.com/watch?v=oBXO-dxF0Gg)

---

#### What does it all mean? - Compositional distributional semantics for modelling natural language

Thomas Kober gave us a very good overview of techniques to compose word embeddings in a way that captures the meaning of longer units of text. In other words, how to combine single word embeddings by applying some kind of function in order to represent sentences or paragraphs. Slides are available [here](https://github.com/pydataberlin/conf2017slides/blob/master/compositional_distributional_semantics/pydata2017.pdf).

[![What does it all mean? - Compositional distributional semantics for modelling natural language](https://img.youtube.com/vi/hTmKoHJw3Mg/hqdefault.jpg)](https://www.youtube.com/watch?v=hTmKoHJw3Mg)

---

#### Machine Learning to moderate ads in real world classified's business

Vaibhav Singh and Jaroslaw Szymczak shared their experiences and learnings in building machine learning models to act as human moderators for ads in an on-line marketplace, OLX. They share some good tips on what to focus when you want to deploy machine learning models in production. Slides are available [here](https://github.com/pydataberlin/conf2017slides/blob/master/machine_learning_to_moderate_ads/MachineLearningToModerateAdsInRealWorldClassifiedsBusiness.pdf).

[![Machine Learning to moderate ads in real world classified's business](https://img.youtube.com/vi/MoOjj8eF8FM/hqdefault.jpg)](https://www.youtube.com/watch?v=MoOjj8eF8FM)

---

#### Find the text similiarity you need with the next generation of word embeddings in Gensim

Lev Konstantinovskiy from [RaRe Technologies](https://rare-technologies.com/) gave on overview of the different methods to generate embeddings: Word2Vec, FastText, WordRank, GloVe and some of the wrappings existent in [gensim](https://radimrehurek.com/gensim/).

[![Find the text similiarity you need with the next generation of word embeddings in Gensim](https://img.youtube.com/vi/tAxrlAVw-Tk/hqdefault.jpg)](https://www.youtube.com/watch?v=tAxrlAVw-Tk)

---

#### On Bandits, and, Bayes swipes: gamification of search

Stefan Otte talked about a topic which I find very interesting but somehow it does not get much attention, [Active Learning (wiki)](https://www.wikiwand.com/en/Active_learning_(machine_learning)), [Active Learning Literature Survey](http://burrsettles.com/pub/settles.activelearning.pdf). He did an introduction to the topic, briefly going through the different active learning strategies, and connected it to multi-armed bandit setting, in a product ranking scenario. Slides are available [here](https://sotte.github.io/pydata-talk-2017-on-bandits-and-swipes-gamification-of-search.html).

[![On Bandits, and, Bayes swipes: gamification of search](https://img.youtube.com/vi/SpRg8KSLZ2w/hqdefault.jpg)](https://www.youtube.com/watch?v=SpRg8KSLZ2w)


---

#### Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics

And finally, a link to my talk, where I showed the results of a bootstrapping system for relationship extraction based on word embeddings. Slides are available [here](../../../../../assets/documents/talks/PyData2017-Berlin-presentation.pdf).

[![Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics](https://img.youtube.com/vi/Ra15lX-wojg/hqdefault.jpg)](https://www.youtube.com/watch?v=Ra15lX-wojg)
