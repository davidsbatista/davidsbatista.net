---
layout: post
title: StanfordNER - training a new model and deploying a web service
date: 2018-01-23 00:00:00
tags: [StanfordNER Conditional Random Fields Named-Entity Recognition]
categories: [blog]
comments: true
disqus_identifier: 20180123
preview_pic: /assets/images/2018-01-23-stanford_ner.png
description: Using StanfordNER to train a Named-Entity Recognizer and setting up a web service
---

Stanford NER is a named-entity recognizer based on linear chain Conditional Random Field (CRF) sequence models. This post details some of the experiments I've done with it, using a corpus to train a Named-Entity Recognizer: the features I've explored (some undocumented), how to setup a web service exposing the trained model and how to call it from a python script.

<!--
image taken from: http://slideplayer.com/slide/5898548/
Named Entity Recognition and the Stanford NER Software Jenny Rose Finkel Stanford University March 9, 2007.
-->

# __Stanford NER__


<figure>
  <img style="width: 50%; height: 50%" src="/assets/images/2018-01-23-stanford_ner.png">
  <figcaption>Named Entity Recognition and the Stanford NER Software (Jenny Finkel)</figcaption>
</figure>


Stanford NER requires Java, I've used [StanfordNER 3.8.0](https://nlp.stanford.edu/software/stanford-ner-2017-06-09.zip), which requires Java v1.8+, so the first thing is to have Java v1.8+ installed and running on your system.

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


---


# __Training your own model__

This section describes the basic steps to train your own NER model, from pre-processing the corpus (if needed), creating _k_-folds for cross-fold validation, defining the features to use, and running Stanford NER in evaluation mode.

## __Corpus__

I've used a annotated corpus which unfortunately isn't public available, and this experiments were done in the context of a research project with the goal to train a named-entity recognizer for Portuguese. The [CINTIL Corpus – International Corpus of Portuguese](http://catalog.elra.info/product_info.php?products_id=1102&language=en) is available through a commercial or academic research license.

### __Pre-Processing__

The first thing I did was to pre-process the original corpus. In it's original form, CINTIL doesn't contain any contractions of prepositions and articles, most probably the tokenizer used to pre-process the corpus, before the annotations being added, extended all the possible contractions among prepositions and articles, for instance:

`Na freguesia, a populaçao ...` appears as: `Em a freguesia, a população ...`

`Daí que ele julgou ...` appears as `De aí que ele julgou que ...`

You can read more about Portuguese contractions [here](https://www.wikiwand.com/en/Contraction_(grammar)#/Portuguese) and [here](https://blogs.transparent.com/portuguese/contractions-in-portuguese/).

I also converted the original corpus format into a CoNNL style format, since the original comes XML. Then, I discard some annotations (i.e., MSC and EVT) which were not relevant for my experiments. I used BIO encoding for 4 different types of named-entities, resulting in a total of 30 343 sentences, and the following number of tags for each token:

{% highlight python %}
          Total Number of Tokens

B-LOC            5 831   
I-LOC            2 999   

B-ORG            6 404
I-ORG            5 766

B-PER           11 308
I-PER            8 728  

B-WRK            1 532
I-WRK            2 192  

O              594 077
----------------------
Total          638 837

{% endhighlight %}


## __K-folds__

I've created 5 folds, with an average of 6 069 sentences per fold, over the original corpus. I noticed after looking at the distribution of types of entities per fold, that the folds were unbalanced.

I ran some experiments, using this original distribution of tags per folds, and quickly noticed that the results were very low for some tags.

Therefore, I simply shuffled the order of sentences in the corpus, and then generated new folds. Notice that the order of the sentences doesn't influence the model, since the CRF will tag each sentence individually. I then inspected the distribution of tokens, which was now balanced compared to before.

You can apply more robust techniques to achieve this balance, but sometimes, like in this case,  a simply shuffle will do the trick.


## __Training / Testing__

The command to train each fold using StanfordNER is the following:

    java -Xmx10g -cp stanford-ner-2017-06-09/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop stanford_ner.prop -trainFile corpus/fold_1_2_3_4 -testFile corpus/fold_0 1>tagged_fold_0.csv 2>results_fold_0.txt -serializeTo model_0.ser.gz`

Breaking it down:

`-prop stanford_ner.prop`: the file which contains the configuration for the model be learned, such as features and the learning algorithm (see next section).

`-trainFile corpus/fold_1_2_3_4`: file containing training data

`-testFile corpus/fold_0`: file containing testing data

`1>tagged_fold_0.csv`: output of the tagged training data, it contains 3 columns: word, true_tag, predicted_tag; this can be useful to perform further evaluations

`2>results_fold_0.txt`: output of the evaluation results and also the logging produced during the learning phase, might also be useful to confirm which checks are on, the number of iterations of the learning algorithm, etc.

`-serializeTo model_fold_0.ser.gz`: file to save the learned model.


The built-in evaluation of StanfordNER shows the results per label and aggregated, that is, you see the overall results for `LOC`, `ORG`, `PER` and `WRK`, and not on each possible token label, i.e.: `B-LOC`, `I-LOC`, etc.

I wrote a simple script which read all the results for each fold, that is, the true label and the predicted label, a computes precision, recall and f1 by each possible token label.


## __Features: experiments and results__

One of the great advantages of StanfordNER is the powerful template of features, which I believe is the contribution of different persons. The downside is that the documentation is not so great and you need to go through the source code to understand exactly what each feature represents.


#### __Experiment 1__

I started with a simple set of baseline features which also include gazetteer and distributional similarity features:

The first set of features is:

```
usePrev = true
useNext = true
useTags = true
```

according to the documentation (javadoc file for NERFeatureFactory.java):

* usePrev:	Gives you feature for (pw,c), and together with other options enables other previous features, such as (pt,c) [with useTags)

* useNext:	Gives you feature for (nw,c), and together with other options enables other next features, such as (nt,c) [with useTags)

* useTags:	Gives you features for (t,c), (pt,c) [if usePrev], (nt,c) [if useNext]

* w = word, t = tag, c = class, p = position (word index in sentence)

* , = AND

this will fire features which associate the position of a word, and the position of it's previous and next words, to the _part-of-speech tag_, _distsim class_ and _entity type_.


```
useGazettes=true
gazette=resources/gazettes/DBPedia-pt-per-gazette.txt;resources/gazettes/DBPedia-pt-loc-gazette.txt;resources/gazettes/DBPedia-pt-org-gazette.txt
cleanGazette=true

checkNameList=true
lastNameList=resources/gazettes/lastNames.txt
maleNameList=resources/gazettes/all-first-names.txt
femaleNameList=resources/gazettes/all-first-names.txt

useDistSim=true
distSimLexicon=resources/word_cluster.txt
casedDistSim=true
```

##### __Gazetteers__

The `useGazettes=true` and `gazetee=files` states that the model should use gazetteers and to generate features, the gazetteers files format is:
`ent_type \t entry` for instance:

```
LOC  Berlin
LOC  Lisboa
```

`cleanGazette=true` means that a feature is generated for a sequence of tokens in text when all the tokens match a gazette entry.

##### __Name Lists__

`checkNameList=true` if set to true it will look at the files specified by `lastNameList`, `maleNameList`, `femaleNameList`, and add these as features for person names, here each name is split into an individual word, and the feature is just a single word, in the list of names.

##### __Distributional Similarity__

The flag `useDistSim=true`, forces the load of the file specified by `distSimLexicon`, this file should contain a word and a class identifier for that word, or the other way around depending on `distSimFileFormat`. The idea if to use distributional similarity classes as features. There many methods to generate, from a big corpus, classes of words. Two possible methods are:

* [Class-Based n-gram Models of Natural Language](https://github.com/percyliang/brown-cluster)

* Clustering words based on the embeddings representation, you can see __demo-classes.sh__ file, part of [word2vec](https://github.com/tmikolov/word2vec) package for an example.

`casedDistSim=true` states wether the tokens should be lowercased or not before looking in the `distSimLexicon` file. You can also set a default class for words that are not found, with the parameter `unknownWordDistSimClass`, more parameters regarding the distributional similarity can be found in the `SeqClassifierFlags.java` file.

Below you can see the performance results for this set of features:

{% highlight python %}
          Precision    Recall     F1

B-LOC	    0.713      0.719	 0.716
I-LOC	    0.675      0.419	 0.517

B-ORG	    0.810      0.675	 0.736
I-ORG	    0.746      0.518	 0.611

B-PER	    0.896      0.770	 0.828
I-PER	    0.857      0.701	 0.771

B-WRK	    0.825      0.505	 0.627
I-WRK	    0.749      0.311	 0.440

{% endhighlight %}



#### __Experiment 2__

Next, on a second experiment I switched on a few more features. The idea is then to run again the training and testing for the same folds and see how the performance varies, and doing this with different sets of switched on or off.

I added the following two features to the baseline features:

```
useWordTag=True
useWordPairs=True
```

* useWordPairs: Gives you features for (pw, w, c) and (w, nw, c)

* useWordTag: Include word and tag pair features

this will fire joint features like the position of the word and the word and the class of the word.

Running again the train/testing on the 5 folds, I got the following results:

{% highlight python %}
          Precision    Recall     F1

B-LOC	    0.735      0.750	 0.743
I-LOC	    0.684      0.462	 0.552

B-ORG	    0.828      0.718	 0.769
I-ORG	    0.753      0.554	 0.639

B-PER	    0.907      0.812	 0.857
I-PER	    0.853      0.715	 0.778

B-WRK	    0.857      0.578	 0.690
I-WRK	    0.783      0.370	 0.502
{% endhighlight %}

There is a significant boost in the both precision and recall results, comparing with the previous results, so it seems we are on the right path :) The only thing is the recall for the `I-LOC` and `I-ORG` which still a bit low comparing with the recall values for the other tags.

#### __Experiment 3__

Next I added two more features:

```
useShapeConjunctions=True
useSymTags=True
```

* useShapeConjunctions: conjoins the shape of word with it's tag and position

* useSymTags: gives the features (pt, t, nt, c), (t, nt, c), (pt, t, c)

<!--
* t = tag
* p = position (word index in sentence)
* c = class
* p = paren
* g = gazette
* a = abbrev
* s = shape
* r = regent (dependency governor)
* h = head word of phrase
* n(w) = ngrams from w
* g(w) = gazette entries containing w
* l(w) = length of w
* o(...) = occurrence patterns of words

-->

{% highlight python %}
          Precision    Recall     F1

B-LOC	    0.737      0.745	 0.737
I-LOC	    0.677      0.460	 0.548

B-ORG	    0.823      0.703	 0.758
I-ORG	    0.746      0.547	 0.631

B-PER	    0.904      0.797	 0.847
I-PER	    0.850      0.710	 0.774

B-WRK	    0.857      0.565	 0.681
I-WRK	    0.784      0.381	 0.513
{% endhighlight %}


This resulted in, overall, lower results than before, I suspected this was mainly due to the huge amount of feature generated by the `useSymTags` flag.

#### __Experiment 4__

So in the next experiment I turned off the `useSymTags` but keep `useShapeConjunctions` and added `useOccurrencePatterns`:

```
useWordTag=True
useWordPairs=True
useShapeConjunctions=True
useOccurrencePatterns=True
```

{% highlight python %}
          Precision    Recall     F1

B-LOC	    0.738      0.746	 0.742
I-LOC	    0.680      0.468	 0.555

B-ORG	    0.826      0.725	 0.772
I-ORG	    0.739      0.562	 0.639

B-PER	    0.905      0.814	 0.857
I-PER	    0.847      0.717	 0.776

B-WRK	    0.854      0.580	 0.691
I-WRK	    0.771      0.370	 0.500
{% endhighlight %}

this kept more or less the same results, with some improvements for the `I-LOC` and `I-ORG`, but a lower `I-WRK`, as you can see it's hard to keep improving every tag, and feature exploration/engineering is not always a straightforward task.


## __Best Results__

I continued played around with more features until finally obtained some satisfactory results, after running several experiments, and trying different features and parameters, this was the final list of features I got:

```
useWordTag=True
useWordPairs=True
useShapeConjunctions=True
useOccurrencePatterns=True
shapes_all=True
useDisjunctive=True
useLastRealWord=True
useNextRealWord=True
```

plus the first baseline features described in the first experiment.

{% highlight python %}
          Precision    Recall     F1

B-LOC	    0.792      0.765	 0.778
I-LOC	    0.770      0.547	 0.639

B-ORG	    0.845      0.712	 0.773
I-ORG	    0.831      0.669	 0.741

B-PER	    0.909      0.798	 0.850
I-PER	    0.875      0.768	 0.818

B-WRK	    0.858      0.589	 0.698
I-WRK	    0.811      0.491	 0.612
{% endhighlight %}


The recall for `I-LOC`, `I-ORG` and `I-WRK` could probably still be improved a bit, but stop experiment after around 10 iterations/experiments.

After this feature selection/engineering step you might want to train a model on all your data with using best set of features.

<!--
# QNMinimizer terminated due to average improvement: | newest_val - previous_val | / |newestVal| < TOL

0.15
0.13

TOL = 10
TOL = 1e-4
-->


---


# __Setting up a web service__

Once a model has been trained you can apply it to text just as shown in the beginning of this post, but a most common use case is to have a web service or a HTTP endpoint, where you submit a sentence or articles, and get back the text with the named-entities identified.

StanfordNER can work as a server, it's a built-in feature, indeed very useful. To start StanfordNER as a server:

    java -mx2g -cp stanford-ner-2017-06-09/stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier my_model.ser.gz -textFile -port 9191 -outputFormat inlineXML

You can then just simply telnet to that port, type a sentence, and get back the text tokenized and tagged, for instance:

```
dsbatista@Davids-MacBook-Pro:~$ telnet 127.0.0.1 9191
Trying 127.0.0.1...
Connected to localhost.
Escape character is '^]'.
Switzerland, Davos 2018: Soros accuses Trump of wanting a 'mafia state' and blasts social media.

<LOCATION>Switzerland</LOCATION>, <PERSON>Davos</PERSON> 2018: <PERSON>Soros</PERSON> accuses <PERSON>Trump</PERSON> of wanting a 'mafia state' and blasts social media.

Connection closed by foreign host.
dsbatista@Davids-MacBook-Pro:~$
```

or one can also wrap the `edu.stanford.nlp.ie.NERServer` inside a python script with [PyNER](https://github.com/dat/pyner):

{% highlight python %}

In [14]: tagger = ner.SocketNER(host='localhost', port=9191)

In [15]: tagger.get_entities("Switzerland, Davos 2018: Soros accuses Trump of wanting a 'mafia state' and blasts social media.")
Out[15]: {'LOCATION': ['Switzerland'], 'PERSON': ['Davos', 'Soros', 'Trump']}

In [16]: tagger.tag_text("Switzerland, Davos 2018: Soros accuses Trump of wanting a 'mafia state' and blasts social media.")
Out[16]: "<LOCATION>Switzerland</LOCATION>, <PERSON>Davos</PERSON> 2018: <PERSON>Soros</PERSON> accuses <PERSON>Trump</PERSON> of wanting a 'mafia state' and blasts social media."

{% endhighlight %}

---


## __References__

Template features file used by this experiment as well as the evaluation scripts:

* [https://github.com/davidsbatista/NER-experiments](https://github.com/davidsbatista)

The dictionaries and gazetteers used for feature generation are publicly available here:

* [https://github.com/davidsbatista/REACTION-resources](https://github.com/davidsbatista/REACTION-resources)

There are also other NER datasets which can be used for supervised learning:

* WikiNER:
  * [script to generate corpus](https://github.com/JonathanRaiman/wikipedia_ner)
  * [paper/thesis](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.178.2963&rep=rep1&type=pdf)

* [NER on noisy text](https://github.com/noisy-text/noisy-text.github.io)

Some features are documented in [Frequently Asked Questions](https://nlp.stanford.edu/software/crf-faq.shtml), but by looking at following classes one can find much more information and details, and also undocumented and newly released template features:

###### __NERFeatureFactory.java__
* [code](https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/ie/NERFeatureFactory.java)
* [javadoc](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ie/NERFeatureFactory.html)


###### __CRFClassifier.java__
* [code](https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/ie/crf/CRFClassifier.java)
* [javadoc](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ie/crf/CRFClassifier.html)


###### __SeqClassifierFlags.java__
* [code](https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/sequences/SeqClassifierFlags.java)
* [javadoc](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/sequences/SeqClassifierFlags.html)


### __Presentations__

* [Named Entity Recognition and the Stanford NER Software Jenny Rose Finkel Stanford University March 9, 2007](http://slideplayer.com/slide/5898548/)
