---
layout: post
title: Named Entity Evaluation Metrics
date: 2018-05-09 00:0:00
categories: [blog]
tags: NER evaluation_metrics
comments: false
# disqus_identifier: #20180509
preview_pic: /assets/images/2018-05-09-NER_metrics.jpeg
description: Named Entity Evaluation Metrics
---
When you train a NER system the most typically evaluation method is to measure __precision__, __recall__ and __f1-score__ at a tag level, for each token that was classified. This metrics are indeed useful to tune a NER system and understand which tags the system can better classify and which ones it can miss. But when using the predicted named-entities for downstream tasks, it is more useful evaluate metrics at a full named-entity level. In this post I will go through some metrics that go beyond simple token-level performance.

## __Comparing NER system output and golden standard__

Chris Manning explains in a [very eloquent way through a blog post](https://nlpers.blogspot.de/2006/08/doing-named-entity-recognition-dont.html) the problems of relying only on F1 at token/tag level as a way to evaluate (or optimise) NERC systems, which I briefly resume.

<!--

system guessed it right:
(in/O/O Palo/LOC/LOCAlto/LOC/LOC ./O/O),

system missed it:
(in/O/O Palo/LOC/O Alto/LOC/O ./O/O),

system hypothesized an entity:
(an/O/O Awful/O/ORG Headache/O/ORG ./O/O).

This all fall naturally into the false negatives (fn), true positives (tp), false negatives (fp), and false positives (fp) of a simple classification evaluation;

But other events can happen. A NER system can notice that there is an entity but give it the wrong label:

(I/O/O live/O/O in/O/O Palo/LOC/ORG Alto/LOC/ORG ./O/O).

A system can notice that there is an entity but get its boundaries wrong:
(Unless/O/PERS Karl/PERS/PERS Smith/PERS/PERS resigns/O/O).

Or it can make both mistakes at once:
(Unless/O/ORG Karl/PERS/ORG Smith/PERS/ORG resigns/O/O).
-->

<!--

<figure>
  <img style="width: 65%; height: 65%" src="/assets/images/2017-11-13-Label_Bias_Problem.png">
  <figcaption>Label Bias Problem. <br> (taken from Lafferty et al. 2001)</figcaption>
</figure>

https://www.pexels.com/photo/tapemeasure-on-20-164957/
-->


Throughout the years different NERC forums proposed different evaluation metrics that beyond simple precision/recall metrics over token tags.

* __MUC and SEmEval'13__:
* __CONLL'03__: evaluates on only on exact matches, so everything falls into simple classification evaluation;
* __ACE__:

<!--
https://github.com/comtravo/ct-backend/blob/master/python_lib/nlp_support/e2e_pipeline/score_annotations.py
https://github.com/jantrienes/nereval
-->

## __References__

* [Chris Manning blog post: "Doing Named Entity Recognition? Don't optimize for F1"](https://nlpers.blogspot.de/2006/08/doing-named-entity-recognition-dont.html)

* [MUC-5 EVALUATION METRICS](https://aclanthology.info/pdf/M/M93/M93-1007.pdf)

* ["Semi-Supervised Named Entity Recognition" David Nadeau PhD Thesis](http://cogprints.org/5859/1/Thesis-David-Nadeau.pdf)

* ["Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"](http://www.aclweb.org/anthology/W03-0419)

* ["Evaluation of the SemEval-2013 Task 9.1"](https://www.cs.york.ac.uk/semeval-2013/task9/data/uploads/semeval_2013-task-9_1-evaluation-metrics.pdf)