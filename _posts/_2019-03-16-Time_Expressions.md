---
layout: post
title: Detecting and Grounding Temporal Expressions
date: 2019-03-16 00:00:00
tags: time-expressions semantics grounding time-semantification
categories: [blog]
# comments: true
# disqus_identifier: 20181206
preview_pic: /assets/images/2019-03-16-time-expressions.jpg
---

Detecting temporal expressions is the task of identifying tokens or sequences of tokens with temporal meaning within the context of a sentence or a document. Grounding or normalising temporal expressions maps from a sequences of tokens with temporal meaning to an unambiguous time, date, or duration machine-readable format. In this blog post I will try to review datasets approach for the two tasks.


## __Introduction__

Humans naturally use expressions like __tomorrow afternoon__ or __early next Monday morning__, but these expressions are contextual and ambiguous. The exact date of __it next Monday__ and __tomorrow__ depends on a reference date, i.e., the date when these expressions are interpreted. Furthermore, {\it early morning} has different meaning to different people, although some common overlap can probably be agreed upon. Computers on the other hand can only deal with well defined and unambiguous times.

Detecting temporal expressions is the task of identifying tokens or sequences of tokens with temporal meaning within the context of a sentence or a document. Grounding or normalising temporal expressions maps from a sequences of tokens with temporal meaning to an unambiguous time, date, or duration machine-readable format. In this blog post I will try to review datasets approach for the two tasks.

<!--
Time normalization ([Verhagen et al. 2007](http://www.aclweb.org/anthology/S07-1014)) is the task of mapping time expressions in natural language into a machine-readble format. This task involves essentially two steps, first identifying the tokens or sequence of tokens that make a time expression and then translating that expression into a proper formal time representation.
-->


## __Time Expressions (TIMEX)__

- __fully qualified__: unambiguously refers to precise interval or point in the time domain, e.g.: "10th July 1984", "10/08/1995 at 13:00";
- __deictic__: they require inferring the binding with the time domain requires to take into account the time of utterance, for instance, when the document was written, e.g.: "today", "yesterday", "Monday next week";
- __anaphoric__: a particular case of deictic for which the utterance time varies according to the temporal expressions previously mentioned in the text, e.g: "that year", "the same week", "the previous month"

---
<br>


## __Tasks__

- __identification__: detect the correct boundaries for a token or sequence of tokens in text that represent a time expression;
- __normalisation__: interpret and represent the temporal meaning of each pre-identified expressions using, for instance, TimeML format.

---
<br>


## __Evaluation Challenges__

- __SemEval 2007 Task 15: TempEval-1__
   - [http://nlp.cs.swarthmore.edu/semeval/tasks/index.php](https://web.archive.org/web/20080420071035/http://nlp.cs.swarthmore.edu/semeval/tasks/index.php)
   - [paper](https://www.aclweb.org/anthology/S07-1014)

- __SemEval 2010 Task 13: TempEval-2__
    - [http://semeval2.fbk.eu/semeval2.php?location=tasks-short](http://semeval2.fbk.eu/semeval2.php?location=tasks-short)
    - [paper](https://www.aclweb.org/anthology/S10-1010)

- __SemEval 2013 Task 11: TempEval-3__
   - [https://www.cs.york.ac.uk/semeval-2013/task1/index.html](https://www.cs.york.ac.uk/semeval-2013/task1/index.html)
   - [paper](http://www.aclweb.org/anthology/S/S13/S13-2001.pdf)


---
<br>

## __Datasets__

<!-- see: https://web.archive.org/web/20130727183253/http://www.timexportal.info/corpora -->

Several datasets and approaches were proposed to tackle the task of mapping a time expression into it's formal representation. Most of this datasets however have two principal characteristics, they are target at he English language and are part of the newswire or wikipedia domain.

- ### __English__

  - __TIMEBANK Corpus (2003)__: Pustejovsky et al. proposed the [TIMEBANK Corpus](http://www.timeml.org/timebank/timebank.html) with annotations of events, time expressions, and temporal relations from transcribed English broadcast news and newswire are annotated according to the ISO-TimeML schema.

  - __WikiWars: A New Corpus for Research on Temporal Expressions (2010)__: [Mazur et al.](http://aclweb.org/anthology/D10-1089) produced a dataset with time expressions annotations from English Wikipedia articles that describe the historical course of wars.

  - __TIMEN: An Open Temporal Expression Normalisation Resource (2012)__ [paper](http://lrec.elra.info/proceedings/lrec2012/pdf/128_Paper.pdf): a mix of TempEval-02 and annotated data from TAC 2010 KBP Source Data (news wire)

- ### __German__

<!--
http://nlpprogress.com/english/temporal_processing.html
https://dbs.ifi.uni-heidelberg.de/files/Team/jannik/publications/fischer-stroetgen_temporal-expressions-in-literary-corpora_dh2015_final_2015-03-01.pdf
-->

---
<br>


## __TIMEX normalisation: systems and approaches__


- __CogCompTime: A Tool for Understanding Time in Natural Language (2018)__ [paper](http://aclweb.org/anthology/D18-2013)

- __A Baseline Temporal Tagger for all Languages (2015)__ [paper](https://aclweb.org/anthology/D/D15/D15-1063.pdf) [code](https://github.com/HeidelTime/heideltime)

- __A synchronous context free grammar for time normalisation (2013)__ [paper](http://www.aclweb.org/anthology/D13-1078)

- __SUTIME: A Library for Recognizing and Normalizing Time Expressions__ [paper](https://nlp.stanford.edu/pubs/lrec2012-sutime.pdf) [code](https://nlp.stanford.edu/software/sutime.html)

- __Parsing Time: Learning to Interpret Time Expressions (2012)__ [paper](https://aclanthology.info/pdf/N/N12/N12-1049.pdf)

- __TIMEN: An Open Temporal Expression Normalisation Resource (2012)__ [paper](http://lrec.elra.info/proceedings/lrec2012/pdf/128_Paper.pdf)

- __EVENT AND TEMPORAL EXPRESSION EXTRACTION FROM RAW TEXT: FIRST STEP TOWARDS A TEMPORALLY AWARE SYSTEM (2010)__ [paper](https://www.researchgate.net/profile/Naushad_UzZaman/publication/220233264_Event_and_Temporal_Expression_Extraction_from_Raw_Text_First_Step_towards_a_Temporally_Aware_System/links/004635384b4c1f3793000000.pdf)

- __Edinburgh-LTG: TempEval-2 System Description (2010)__ [paper](https://www.aclweb.org/anthology/S10-1074)









<br><br><br><br>

## __References__

- [Temporal expression extraction with extensive feature type selection and a posteriori label adjustment](https://www.sciencedirect.com/science/article/pii/S0169023X15000725)
- Towards Task-Based Temporal Extraction and Recognition.pdf
- [1] ([Verhagen et al. 2007](http://www.aclweb.org/anthology/S07-1014))
- [2] http://ucrel.lancs.ac.uk/publications/cl2003/papers/pustejovsky.pdf
- [3] (Time Bank Corpus)[http://www.timeml.org/timebank/timebank.html]
- [Recognizing Time Expressions (from the Stanford NLP group)](https://nlp.stanford.edu/projects/time.shtml)