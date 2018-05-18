---
layout: post
title: Named-Entity evaluation metrics based on entity-level
date: 2018-05-09 00:0:00
categories: [blog]
tags: NER evaluation_metrics
comments: true
disqus_identifier: 20180509
preview_pic: /assets/images/2018-05-09-NER_metrics.jpeg
description: Named-Entity evaluation metrics based on entity-level
---
When you train a NER system the most typically evaluation method is to measure __precision__, __recall__ and __f1-score__ at a token level. These metrics are indeed useful to tune a NER system. But when using the predicted named-entities for downstream tasks, it is more useful evaluate metrics at a full named-entity level. In this post I will go through some metrics that go beyond simple token-level performance.

Note you can find the complete code for this blog post on this repository:

* [https://github.com/davidsbatista/NER-Evaluation](https://github.com/davidsbatista/NER-Evaluation)

### __Comparing NER system output and golden standard__

Comparing the golden standard annotations with the output of a NER system different scenarios might occur:

#### __I. Surface string and entity type match__

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-us36{border-color:inherit;vertical-align:top}
.tg .tg-7btt{font-weight:bold;border-color:inherit;text-align:center;vertical-align:top}
</style>

<center>
<table class="tg">
  <tr>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">Golden Standard</span></th>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">System Prediction</span></th>
  </tr>
  <tr>
    <td class="tg-7btt">Entity Type</td>
    <td class="tg-7btt">Surface String</td>
    <td class="tg-c3ow"><span style="font-weight:bold">Entity Type</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">Surface String</span></td>
  </tr>
  <tr>
    <td class="tg-us36">in</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">in</td>
    <td class="tg-us36">O</td>
  </tr>
  <tr>
    <td class="tg-us36">New</td>
    <td class="tg-us36">B-LOC</td>
    <td class="tg-us36">New</td>
    <td class="tg-us36">B-LOC</td>
  </tr>
  <tr>
    <td class="tg-us36">York</td>
    <td class="tg-us36">I-LOC</td>
    <td class="tg-us36">York</td>
    <td class="tg-us36">I-LOC</td>
  </tr>
  <tr>
    <td class="tg-us36">.</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">.</td>
    <td class="tg-us36">O</td>
  </tr>
</table>
</center>

#### __II. System hypothesized an entity__

<center>
<table class="tg">
  <tr>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">Golden Standard</span></th>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">System Prediction</span></th>
  </tr>
  <tr>
    <td class="tg-7btt">Entity Type</td>
    <td class="tg-7btt">Surface String</td>
    <td class="tg-c3ow"><span style="font-weight:bold">Entity Type</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">Surface String</span></td>
  </tr>
  <tr>
    <td class="tg-us36">an</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">an</td>
    <td class="tg-us36">O</td>
  </tr>
  <tr>
    <td class="tg-us36">Awful</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">Awful</td>
    <td class="tg-us36">B-ORG</td>
  </tr>
  <tr>
    <td class="tg-us36">Headache</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">Headache</td>
    <td class="tg-us36">I-ORG</td>
  </tr>
  <tr>
    <td class="tg-us36">in</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">in</td>
    <td class="tg-us36">O</td>
  </tr>
</table>
</center>



#### __III. System misses an entity__

<center>
<table class="tg">
  <tr>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">Golden Standard</span></th>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">System Prediction</span></th>
  </tr>
  <tr>
    <td class="tg-7btt">Entity Type</td>
    <td class="tg-7btt">Surface String</td>
    <td class="tg-c3ow"><span style="font-weight:bold">Entity Type</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">Surface String</span></td>
  </tr>
  <tr>
    <td class="tg-us36">in</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">in</td>
    <td class="tg-us36">O</td>
  </tr>
  <tr>
    <td class="tg-us36">Palo</td>
    <td class="tg-us36">B-LOC</td>
    <td class="tg-us36">Palo</td>
    <td class="tg-us36">O</td>
  </tr>
  <tr>
    <td class="tg-us36">Alto</td>
    <td class="tg-us36">I-LOC</td>
    <td class="tg-us36">Alto</td>
    <td class="tg-us36">O</td>
  </tr>
  <tr>
    <td class="tg-us36">,</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">,</td>
    <td class="tg-us36">O</td>
  </tr>
</table>
</center>

Note that considering only this 3 scenarios, and considering everything else wrong, we have a simple classification evaluation that can be measured in terms of false negatives, true positives, false negatives and false positives, and subsequently compute precision, recall and f1-score for each named-entity type.

#### __IV. System assigns the wrong entity type__

<center>
<table class="tg">
  <tr>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">Golden Standard</span></th>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">System Prediction</span></th>
  </tr>
  <tr>
    <td class="tg-7btt">Entity Type</td>
    <td class="tg-7btt">Surface String</td>
    <td class="tg-c3ow"><span style="font-weight:bold">Entity Type</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">Surface String</span></td>
  </tr>
  <tr>
    <td class="tg-us36">I</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">I</td>
    <td class="tg-us36">O</td>
  </tr>
  <tr>
    <td class="tg-us36">live</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">live</td>
    <td class="tg-us36">O</td>
  </tr>
  <tr>
    <td class="tg-us36">in</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">in</td>
    <td class="tg-us36">O</td>
  </tr>
  <tr>
    <td class="tg-us36">Palo</td>
    <td class="tg-us36">B-LOC</td>
    <td class="tg-us36">Palo</td>
    <td class="tg-us36">B-ORG</td>
  </tr>
  <tr>
    <td class="tg-us36">Alto</td>
    <td class="tg-us36">I-LOC</td>
    <td class="tg-us36">Alto</td>
    <td class="tg-us36">I-ORG</td>
  </tr>
  <tr>
    <td class="tg-us36">,</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">,</td>
    <td class="tg-us36">O</td>
  </tr>
</table>
</center>



#### __V. System gets the boundaries of the surface string wrong__

<center>
<table class="tg">
  <tr>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">Golden Standard</span></th>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">System Prediction</span></th>
  </tr>
  <tr>
    <td class="tg-7btt">Entity Type</td>
    <td class="tg-7btt">Surface String</td>
    <td class="tg-c3ow"><span style="font-weight:bold">Entity Type</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">Surface String</span></td>
  </tr>
  <tr>
    <td class="tg-us36">Unless</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">Unless</td>
    <td class="tg-us36">B-PER</td>
  </tr>
  <tr>
    <td class="tg-us36">Karl</td>
    <td class="tg-us36">B-PER</td>
    <td class="tg-us36">Karl</td>
    <td class="tg-us36">I-PER</td>
  </tr>
  <tr>
    <td class="tg-us36">Smith</td>
    <td class="tg-us36">I-PER</td>
    <td class="tg-us36">Smith</td>
    <td class="tg-us36">I-PER</td>
  </tr>
  <tr>
    <td class="tg-us36">resigns</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">resigns</td>
    <td class="tg-us36">O</td>
  </tr>
</table>
</center>



#### __VI. System gets the boundaries and entity type wrong__

<center>
<table class="tg">
  <tr>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">Golden Standard</span></th>
    <th class="tg-c3ow" colspan="2"><span style="font-weight:bold">System Prediction</span></th>
  </tr>
  <tr>
    <td class="tg-7btt">Entity Type</td>
    <td class="tg-7btt">Surface String</td>
    <td class="tg-c3ow"><span style="font-weight:bold">Entity Type</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">Surface String</span></td>
  </tr>
  <tr>
    <td class="tg-us36">Unless</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">Unless</td>
    <td class="tg-us36">B-ORG</td>
  </tr>
  <tr>
    <td class="tg-us36">Karl</td>
    <td class="tg-us36">B-PER</td>
    <td class="tg-us36">Karl</td>
    <td class="tg-us36">I-ORG</td>
  </tr>
  <tr>
    <td class="tg-us36">Smith</td>
    <td class="tg-us36">I-PER</td>
    <td class="tg-us36">Smith</td>
    <td class="tg-us36">I-ORG</td>
  </tr>
  <tr>
    <td class="tg-us36">resigns</td>
    <td class="tg-us36">O</td>
    <td class="tg-us36">resigns</td>
    <td class="tg-us36">O</td>
  </tr>
</table>
</center>


### __Different Evaluation Schemas__

Throughout the years different NERC forums proposed different evaluation metrics:

#### __CoNLL: Computational Natural Language Learning__

The [Language-Independent Named Entity Recognition task](http://www.aclweb.org/anthology/W03-0419) introduced at CoNLL-2003 measures the performance of the systems in terms of precision, recall and f1-score, where:

 _"precision is the percentage of named entities found by the learning system that are correct. Recall is the percentage of named entities present in the corpus that are found by the system. A named entity is correct only if it is an exact match of the corresponding entity in the data file."_

 so basically it only considers scenarios I, II and III, the others described scenarios are not considered for evaluation.


#### __Automatic Content Extraction (ACE)__


The ACE challenges use a more complex evaluation metric which include a weighting schema, I will not go into detail here, and just point for the papers about it:

* ["Automatic Content Extraction 2008 Evaluation Plan (ACE08)""](http://www.eng.utah.edu/~cs6961/papers/ACE-2008-description.pdf)

* ["The Automatic Content Extraction (ACE) Program Tasks, Data, and Evaluation" ](https://pdfs.semanticscholar.org/0617/dd6924df7a3491c299772b70e90507b195dc.pdf)

I kind of gave up on trying to understand results and replicating experiments and baselines from ACE since all the datasets and results are not open and free, so I guess this challenge results and experiments will fade away with time.


#### __Message Understanding Conference (MUC)__

MUC introduced detailed metrics in an [evaluation considering different categories of errors](http://www.aclweb.org/anthology/M93-1007), these metrics can be defined as in terms of comparing the response of a system against the golden annotation:

* __Correct (COR)__   : both are the same;
* __Incorrect (INC)__ : the output of a system and the golden annotation don't match;
* __Partial (PAR)__   : system and the golden annotation are somewhat "similar" but not the same;
* __Missing (MIS)__   : a golden annotation is not captured by a system;
* __Spurius (SPU)__   : system produces a response which doesn't exit in the golden annotation;

these metrics already go a beyond the simple strict classification which only considers totally correct entities, and consider partial matching for instance, and are close to cover the scenarios defined above, we just need to find a way to consider the differences - between NER output and golden annotations - based on two axes, the surface string and the entity type.

An implementation of the MUC evaluation metrics can be found here:

* [https://github.com/jantrienes/nereval](https://github.com/jantrienes/nereval)


#### __International Workshop on Semantic Evaluation (SemEval)__

The SemEval'13 introduced four different ways to measure precision/recall/f1-score results based on the metrics defined by MUC.


* __Strict__: exact boundary surface string match and entity type;

* __Exact__: exact boundary match over the surface string, regardless of the type;

* __Partial__: partial boundary match over the surface string, regardless of the type;

* __Type__: some overlap between the system tagged entity and the gold annotation is required;

each of these ways to measure the performance accounts for correct, incorrect, partial, missed and spurious in different ways. Let's look in detail and see how each of the metrics defined by MUC falls into each of the scenarios described above

<center>
<table class="tg">
  <tr>
    <th class="text-center" colspan="1"><span style="font-weight:bold">Scenario</span></th>
    <th class="text-center" colspan="2"><span style="font-weight:bold">Golden Standard</span></th>
    <th class="text-center" colspan="2"><span style="font-weight:bold">System Prediction</span></th>
    <th class="text-center" colspan="4"><span style="font-weight:bold">Evaluation Schema</span></th>
  </tr>
  <tr>
    <td></td>
    <td><span style="font-weight:bold">Entity Type</span></td>
    <td><span style="font-weight:bold">Surface String</span></td>
    <td><span style="font-weight:bold">Entity Type</span></td>
    <td><span style="font-weight:bold">Surface String</span></td>
    <td><span class="text-center" style="font-weight:bold">Type</span></td>
    <td><span class="text-center" style="font-weight:bold">Partial</span></td>
    <td><span class="text-center" style="font-weight:bold">Exact</span></td>
    <td><span class="text-center" style="font-weight:bold">Strict</span></td>
  </tr>
  <tr>
    <td>III</td>
    <td>brand</td>
    <td>TIKOSYN</td>
    <td></td>
    <td></td>
    <td>MIS</td>
    <td>MIS</td>
    <td>MIS</td>
    <td>MIS</td>
  </tr>
  <tr>
    <td>II</td>
    <td></td>
    <td></td>
    <td>brand</td>
    <td>healthy</td>
    <td>SPU</td>
    <td>SPU</td>
    <td>SPU</td>
    <td>SPU</td>
  </tr>
  <tr>
    <td>V</td>
    <td>drug</td>
    <td>warfarin</td>
    <td>drug</td>
    <td>of warfarin</td>
    <td>COR</td>
    <td>PAR</td>
    <td>INC</td>
    <td>INC</td>
  </tr>
  <tr>
    <td>IV</td>
    <td>drug</td>
    <td>propranolol</td>
    <td>brand</td>
    <td>propranolol</td>
    <td>INC</td>
    <td>COR</td>
    <td>COR</td>
    <td>INC</td>
  </tr>
  <tr>
    <td>I</td>
    <td>drug</td>
    <td>phenytoin</td>
    <td>drug</td>
    <td>phenytoin</td>
    <td>COR</td>
    <td>COR</td>
    <td>COR</td>
    <td>COR</td>
  </tr>
  <tr>
    <td>I</td>  
    <td>Drug</td>
    <td>theophylline</td>
    <td>drug</td>
    <td>theophylline</td>
    <td>COR</td>
    <td>COR</td>
    <td>COR</td>
    <td>COR</td>
  </tr>
  <tr>
    <td>VI</td>
    <td>group</td>
    <td>contraceptives</td>
    <td>drug</td>
    <td>oral contraceptives</td>
    <td>INC</td>
    <td>PAR</td>
    <td>INC</td>
    <td>INC</td>
  </tr>
</table>
</center>

Then precision/recall/f1-score are calculated for each different evaluation schema. In order to achieve data, two more quantities need to be calculated:

Number of gold-standard annotations contributing to the final score
<center>
$\text{POSSIBLE} (POS) = COR + INC + PAR + MIS = TP + FN $
</center>

Number of annotations produced by the NER system:
<center>
$\text{ACTUAL} (ACT) = COR + INC + PAR + SPU = TP + FP$
</center>

Then we can compute precision/recall/f1-score, where roughly describing __precision__ is the percentage of correct named-entities found by the NER system, and __recall__ is the percentage of the named-entities in the golden annotations that are retrieved by the NER system. This is computed in two different ways depending wether we want an __exact match__ (i.e., _strict_ and _exact_ ) or a __partial match__ (i.e., _partial_ and _type_) scenario:

#### __Exact Match__ (i.e., _strict_ and _exact_ )

<center>
$\text{Precision} = \frac{COR}{ACT} = \frac{TP}{TP+FP}$
</center>

<center>
$\text{Recall} = \frac{COR}{POS} = \frac{TP}{TP+FN}$
</center>


#### __Partial Match__ (i.e., _partial_ and _type_)

<center>
$\text{Precision} = \frac{COR\ +\  0.5\  \times\  PAR}{ACT} = \frac{TP}{TP+FP}$
</center>

<center>
$\text{Recall} = \frac{COR}{POS} = \frac{COR}{ACT} = \frac{TP}{TP+FP}$
</center>

Putting all together:

<center>
<table class="tg">
  <tr>
    <th class="text-center" colspan="1"><span style="font-weight:bold">Measure</span></th>
    <th class="text-center" colspan="1"><span style="font-weight:bold">Type</span></th>
    <th class="text-center" colspan="1"><span style="font-weight:bold">Partial</span></th>
    <th class="text-center" colspan="1"><span style="font-weight:bold">Exact</span></th>
    <th class="text-center" colspan="1"><span style="font-weight:bold">Strict</span></th>
  </tr>
  <tr>
    <td><span style="font-weight:bold">Correct</span></td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>2</td>
  </tr>
  <tr>
    <td><span style="font-weight:bold">Incorrect</span></td>
    <td>2</td>
    <td>0</td>
    <td>2</td>
    <td>3</td>
  </tr>
  <tr>
    <td><span style="font-weight:bold">Partial</span></td>
    <td>0</td>
    <td>2</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td><span style="font-weight:bold">Missed</span></td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td><span style="font-weight:bold">Spurius</span></td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td><span style="font-weight:bold">Precision</span></td>
    <td>0.5</td>
    <td>0.5</td>
    <td>0.5</td>
    <td>0.33</td>
  </tr>
  <tr>
    <td><span style="font-weight:bold">Recall</span></td>
    <td>0.5</td>
    <td>0.66</td>
    <td>0.5</td>
    <td>0.33</td>
  </tr>
  <tr>
    <td><span style="font-weight:bold">F1</span></td>
    <td>0.5</td>
    <td>0.66</td>
    <td>0.5</td>
    <td>0.33</td>
  </tr>
</table>
</center>

## __Code__

I did a small experiment using [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/) wrapper around [CRFsuite](http://www.chokkan.org/software/crfsuite/) to train a NER over the CoNLL 2002 Spanish data. Next I evaluate the trained CRF over the test data and show the performance with the different metrics:

Note you can find the complete code for this blog post on this repository:

* [https://github.com/davidsbatista/NER-Evaluation](https://github.com/davidsbatista/NER-Evaluation)

### __Example__

```python
import nltk
import sklearn_crfsuite

from copy import deepcopy
from collections import defaultdict

from sklearn_crfsuite import metrics
from ner_evaluation import collect_named_entities
from ner_evaluation import compute_metrics
from ner_evaluation import compute_metrics_by_type
```

## Train a CRF on the CoNLL 2002 NER Spanish data


```python
nltk.corpus.conll2002.fileids()
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
```


```python
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]
```

## Feature Extraction


```python
%%time
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]
```

    CPU times: user 1.09 s, sys: 88.7 ms, total: 1.18 s
    Wall time: 1.18 s


## Training


```python
%%time
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
```

    CPU times: user 34.7 s, sys: 141 ms, total: 34.8 s
    Wall time: 34.8 s


## Performance per label type per token


```python
y_pred = crf.predict(X_test)
labels = list(crf.classes_)
labels.remove('O') # remove 'O' label from evaluation
sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0])) # group B and I results
print(sklearn_crfsuite.metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
```

                 precision    recall  f1-score   support

          B-LOC      0.810     0.784     0.797      1084
          I-LOC      0.690     0.637     0.662       325
         B-MISC      0.731     0.569     0.640       339
         I-MISC      0.699     0.589     0.639       557
          B-ORG      0.807     0.832     0.820      1400
          I-ORG      0.852     0.786     0.818      1104
          B-PER      0.850     0.884     0.867       735
          I-PER      0.893     0.943     0.917       634

    avg / total      0.809     0.787     0.796      6178



## Performance over full named-entity


```python
test_sents_labels = []
for sentence in test_sents:
    sentence = [token[2] for token in sentence]
    test_sents_labels.append(sentence)
```


```python
index = 2
true = collect_named_entities(test_sents_labels[index])
pred = collect_named_entities(y_pred[index])
```


```python
true
```




    [Entity(e_type='MISC', start_offset=12, end_offset=12),
     Entity(e_type='LOC', start_offset=15, end_offset=15),
     Entity(e_type='PER', start_offset=37, end_offset=39),
     Entity(e_type='ORG', start_offset=45, end_offset=46)]




```python
pred
```




    [Entity(e_type='MISC', start_offset=12, end_offset=12),
     Entity(e_type='LOC', start_offset=15, end_offset=15),
     Entity(e_type='PER', start_offset=37, end_offset=39),
     Entity(e_type='LOC', start_offset=45, end_offset=46)]




```python
compute_metrics(true, pred)
```




    {'ent_type': {'actual': 4,
      'correct': 3,
      'incorrect': 1,
      'missed': 0,
      'partial': 0,
      'possible': 4,
      'precision': 0.75,
      'recall': 0.75,
      'spurius': 0},
     'exact_matching': {'actual': 4,
      'correct': 4,
      'incorrect': 0,
      'missed': 0,
      'partial': 0,
      'possible': 4,
      'precision': 1.0,
      'recall': 1.0,
      'spurius': 0},
     'partial_matching': {'actual': 4,
      'correct': 4,
      'incorrect': 0,
      'missed': 0,
      'partial': 0,
      'possible': 4,
      'precision': 1.0,
      'recall': 1.0,
      'spurius': 0},
     'strict': {'actual': 4,
      'correct': 3,
      'incorrect': 1,
      'missed': 0,
      'partial': 0,
      'possible': 4,
      'precision': 0.75,
      'recall': 0.75,
      'spurius': 0}}


```python
true_named_entities_type = defaultdict(list)
pred_named_entities_type = defaultdict(list)

for true in collect_named_entities(test_sents_labels[index]):
    true_named_entities_type[true.e_type].append(true)

for pred in collect_named_entities(y_pred[index]):
    pred_named_entities_type[pred.e_type].append(pred)
```


```python
true_named_entities_type
```




    defaultdict(list,
                {'LOC': [Entity(e_type='LOC', start_offset=15, end_offset=15)],
                 'MISC': [Entity(e_type='MISC', start_offset=12, end_offset=12)],
                 'ORG': [Entity(e_type='ORG', start_offset=45, end_offset=46)],
                 'PER': [Entity(e_type='PER', start_offset=37, end_offset=39)]})




```python
pred_named_entities_type
```




    defaultdict(list,
                {'LOC': [Entity(e_type='LOC', start_offset=15, end_offset=15),
                  Entity(e_type='LOC', start_offset=45, end_offset=46)],
                 'MISC': [Entity(e_type='MISC', start_offset=12, end_offset=12)],
                 'PER': [Entity(e_type='PER', start_offset=37, end_offset=39)]})




```python
true_named_entities_type['LOC']
```




    [Entity(e_type='LOC', start_offset=15, end_offset=15)]




```python
pred_named_entities_type['LOC']
```




    [Entity(e_type='LOC', start_offset=15, end_offset=15),
     Entity(e_type='LOC', start_offset=45, end_offset=46)]




```python
compute_metrics(true_named_entities_type['LOC'], pred_named_entities_type['LOC'])
```




    {'ent_type': {'actual': 2,
      'correct': 1,
      'incorrect': 0,
      'missed': 0,
      'partial': 0,
      'possible': 1,
      'precision': 0.5,
      'recall': 1.0,
      'spurius': 1},
     'exact_matching': {'actual': 2,
      'correct': 1,
      'incorrect': 0,
      'missed': 0,
      'partial': 0,
      'possible': 1,
      'precision': 0.5,
      'recall': 1.0,
      'spurius': 1},
     'partial_matching': {'actual': 2,
      'correct': 1,
      'incorrect': 0,
      'missed': 0,
      'partial': 0,
      'possible': 1,
      'precision': 0.5,
      'recall': 1.0,
      'spurius': 1},
     'strict': {'actual': 2,
      'correct': 1,
      'incorrect': 0,
      'missed': 0,
      'partial': 0,
      'possible': 1,
      'precision': 0.5,
      'recall': 1.0,
      'spurius': 1}}



## results over all messages


```python
metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurius': 0, 'possible': 0, 'actual': 0}
results = {'strict': deepcopy(metrics),
           'exact_matching': deepcopy(metrics),
           'partial_matching': deepcopy(metrics),
           'ent_type': deepcopy(metrics)
          }

for true, pred in zip(test_sents_labels, y_pred):
    tmp_results = compute_metrics(collect_named_entities(true),collect_named_entities(pred))
    for eval_schema in results.keys():
        for metric in metrics.keys():
            results[eval_schema][metric] +=  tmp_results[eval_schema][metric]
```


```python
results
```




    {'ent_type': {'actual': 3518,
      'correct': 2909,
      'incorrect': 564,
      'missed': 106,
      'partial': 0,
      'possible': 3579,
      'spurius': 45},
     'exact_matching': {'actual': 3518,
      'correct': 3274,
      'incorrect': 199,
      'missed': 106,
      'partial': 0,
      'possible': 3579,
      'spurius': 45},
     'partial_matching': {'actual': 3518,
      'correct': 3274,
      'incorrect': 0,
      'missed': 106,
      'partial': 199,
      'possible': 3579,
      'spurius': 45},
     'strict': {'actual': 3518,
      'correct': 2779,
      'incorrect': 694,
      'missed': 106,
      'partial': 0,
      'possible': 3579,
      'spurius': 45}}



## results over all messages by ent_type


```python
entity_types = ['LOC', 'PER', 'MISC', 'ORG']
all_results = compute_results(test_sents_labels, y_pred, entity_types)
```


```python
all_results['ent_type']
```




    {'LOC': {'correct': 863,
      'incorrect': 0,
      'missed': 124,
      'partial': 0,
      'spurius': 66},
     'MISC': {'correct': 212,
      'incorrect': 0,
      'missed': 43,
      'partial': 0,
      'spurius': 7},
     'ORG': {'correct': 1183,
      'incorrect': 0,
      'missed': 166,
      'partial': 0,
      'spurius': 153},
     'PER': {'correct': 657,
      'incorrect': 0,
      'missed': 46,
      'partial': 0,
      'spurius': 17}}




```python
all_results['strict']
```




    {'LOC': {'correct': 840,
      'incorrect': 23,
      'missed': 124,
      'partial': 0,
      'spurius': 66},
     'MISC': {'correct': 173,
      'incorrect': 39,
      'missed': 43,
      'partial': 0,
      'spurius': 7},
     'ORG': {'correct': 1120,
      'incorrect': 63,
      'missed': 166,
      'partial': 0,
      'spurius': 153},
     'PER': {'correct': 646,
      'incorrect': 11,
      'missed': 46,
      'partial': 0,
      'spurius': 17}}




```python
all_results['exact_matching']
```




    {'LOC': {'correct': 840,
      'incorrect': 23,
      'missed': 124,
      'partial': 0,
      'spurius': 66},
     'MISC': {'correct': 173,
      'incorrect': 39,
      'missed': 43,
      'partial': 0,
      'spurius': 7},
     'ORG': {'correct': 1120,
      'incorrect': 63,
      'missed': 166,
      'partial': 0,
      'spurius': 153},
     'PER': {'correct': 646,
      'incorrect': 11,
      'missed': 46,
      'partial': 0,
      'spurius': 17}}




```python
all_results['partial_matching']
```




    {'LOC': {'correct': 840,
      'incorrect': 0,
      'missed': 124,
      'partial': 23,
      'spurius': 66},
     'MISC': {'correct': 173,
      'incorrect': 0,
      'missed': 43,
      'partial': 39,
      'spurius': 7},
     'ORG': {'correct': 1120,
      'incorrect': 0,
      'missed': 166,
      'partial': 63,
      'spurius': 153},
     'PER': {'correct': 646,
      'incorrect': 0,
      'missed': 46,
      'partial': 11,
      'spurius': 17}}


## __References__

* [Chris Manning blog post: "Doing Named Entity Recognition? Don't optimize for F1"](https://nlpers.blogspot.de/2006/08/doing-named-entity-recognition-dont.html)

* [MUC-5 EVALUATION METRICS](https://aclanthology.info/pdf/M/M93/M93-1007.pdf)

* ["Semi-Supervised Named Entity Recognition" David Nadeau PhD Thesis](http://cogprints.org/5859/1/Thesis-David-Nadeau.pdf)

* ["Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"](http://www.aclweb.org/anthology/W03-0419)

* ["Evaluation of the SemEval-2013 Task 9.1"](https://www.cs.york.ac.uk/semeval-2013/task9/data/uploads/semeval_2013-task-9_1-evaluation-metrics.pdf)

* ["MUC-5 Evaluation Metrics"](http://www.aclweb.org/anthology/M93-1007)