---
layout: post
title: Training a NER model with Stanford NER toolkit and deploying it as a web service
date: 2018-01-23 00:00:00
tags: [StanfordNER Conditional Random Fields Named-Entity Recognition]
categories: [blog]
comments: true
disqus_identifier: 20180123
preview_pic:
preview_pic: /assets/images/2018-01-23-stanford_ner.png
description: Using StanfordNER to train a Named-Entity Recognizer and setting up a web service
---

Stanford NER is a named-entity recognizer based on linear chain Conditional Random Field (CRF) sequence models. This post details some of the experiments I've done with it, using a custom corpus to train Named-Entity Recognizer: the features I've explored (some undocumented), how to setup a web service exposing the trained model and how to call it from a python script.


<!--
image taken from: http://slideplayer.com/slide/5898548/
Named Entity Recognition and the Stanford NER Software Jenny Rose Finkel Stanford University March 9, 2007.
-->

# __Stanford NER__

Stanford NER requires Java, I've used [StanfordNER 3.8.0](https://nlp.stanford.edu/software/stanford-ner-2017-06-09.zip), which requires Java v1.8+, so the first thing is to have Java v1.8+ installed and running on your system.

<!-- set up 2. quick run: use corpus already trained -->

Once Java is setup, you can run Stanford NER using one of  the already trained models, which are distributed together with the zip file.

Create a file with a sample sentence in english.

    echo "Switzerland, Davos 2018: Soros accuses Trump of wanting a 'mafia state' and blasts social media." > test_file.txt

Then run the java command, applying the `english.all.3class.distsim.crf.ser.gz` model to the sentence above:

    java -cp stanford-ner-2017-06-09/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier classifiers/english.all.3class.distsim.crf.ser.gz -textFile test_file.txt

This should output:

    Switzerland/LOCATION ,/O Davos/PERSON 2018/O :/O Soros/PERSON accuses/O Trump/PERSON of/O wanting/O a/O `/O mafia/O state/O '/O and/O blasts/O social/O media/O ./O

The output of the model can be configured with the `-outputFormat` parameter, for instance, with `-outputFormat tsv`:

    Switzerland	LOCATION
    ,	O
    Davos	PERSON
    2018	O
    :	O
    Soros	PERSON
    accuses	O
    Trump	PERSON
    of	O
    wanting	O
    a	O
    `	O
    mafia	O
    state	O
    '	O
    and	O
    blasts	O
    social	O
    media	O
    .	O

Other possible formats are: `xml`, `inlineXML`, `tsv` or `slashTags`


# __Training your own models__


## __Corpus__

I've used a annotated corpus which unfortunately isn't public available, and this experiments were done in the context of a research project with the goal to train a named-entity recognizer for Portuguese. The [CINTIL Corpus – International Corpus of Portuguese](http://catalog.elra.info/product_info.php?products_id=1102&language=en) is available through a commercial or academic research license.

The first thing I did was to pre-process the original corpus. In it's original form, CINTIL doesn't contain any contractions of prepositions and articles, most probably the tokenizer used to pre-process the corpus, before the annotations, extended all the possible contractions, for instance:

`Na freguesia, a populaçao ...` appears as: `Em a freguesia, a população ...`

`Daí que ele julgou ...` appears as `De aí que ele julgou que ...`

You can read more about Portuguese contractions [here](https://www.wikiwand.com/en/Contraction_(grammar)#/Portuguese) and [here](https://blogs.transparent.com/portuguese/contractions-in-portuguese/).

I also transformed the original corpus format into a CoNNL style format, since the original comes XML. Then, I discard some annotations (i.e., MSC and EVT) which were not relevant for my experiments. I used BIO encoding for 4 different types of named-entities:

    B-LOC       5 831
    I-LOC       2 999

    B-ORG       6 404
    I-ORG       5 766

    B-PER      11 308
    I-PER       8 728

    B-WRK       1 532
    I-WRK       2 192

        O     594 077
              638 837


<!-- 3.1 setup -->
<!-- 3.2 extract features -->
<!-- 3.3 training -->
<!-- 3.4 testing/results -->

<!---
Train classifier
================

java -Xmx10g -cp stanford-ner-2017-06-09/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop stanford_ner.prop -trainFile corpus/train.tsv -testFile corpus/test.tsv
java -Xmx10g -cp stanford-ner-2017-06-09/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop stanford_ner.prop 1> annotation_results.txt 2> performance_results.txt


Apply trained classifier
=======================

java -mx2g -cp stanford-ner-2017-06-09/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier CINTIL-BIO-CRF-model.ser.gz -textFile text.txt -outputFormat tsv -encoding utf8

-outputFormat: xml, inlineXML, tsv or slashTags

NERServer
========
java -mx2g -cp stanford-ner-2017-06-09/stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier CINTIL-BIO-CRF-model.ser.gz -textFile -port 9191 -outputFormat inlineXML


Documentation
==============

NERFeatureFactory : https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ie/NERFeatureFactory.html
SeqClassifierFlags: https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/sequences/SeqClassifierFlags.html
FAQ:                https://nlp.stanford.edu/software/crf-faq.shtml#input




# QNMinimizer terminated due to average improvement: | newest_val - previous_val | / |newestVal| < TOL

0.15
0.13

TOL = 10
TOL = 1e-4
-->















<!-- running as a server -->

<!-- python interface -->

<!-- example of calling webservice with telnet or python -->