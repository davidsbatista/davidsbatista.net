---
layout: post
title: Applying scikit-learn TfidfVectorizer on tokenized text
date: 2018-02-28 00:0:00
categories: blog
tags: scikit-learn tokenization
comments: true
disqus_identifier: 20180223
preview_pic: /assets/images/2018-02-28-scikit-learn.png
description: An example showing how to use scikit-learn TfidfVectorizer class on text which is already tokenized, i.e., in a list of tokens.
---

Sometimes your tokenization process is so complex that cannot be captured by a simple regular expression that you can pass to the scikit-learn `TfidfVectorizer`. Instead you just want to pass a list of tokens, resulting of a tokenization process, to initialize a `TfidfVectorizer` object.

There are manly two things that need to be done. First, in the initialization of the `TfidfVectorizer` object you need to pass a dummy `tokenizer` and `preprocessor` that simply return what they receive. Note, you can instead of a `dummy_fun` also pass a lambda function, e.g.: `lambda x: x`, but be aware that if you then want to use the cool `n_jobs=10` for training classifiers or doing parameter grid search pickle cannot handle lambda functions.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  
```

Then you can fit a collections of documents already tokenized

```python
docs = [
    ['Two', 'wrongs', 'don\'t', 'make', 'a', 'right', '.'],
    ['The', 'pen', 'is', 'mightier', 'than', 'the', 'sword'],
    ['Don\'t', 'put', 'all', 'your', 'eggs', 'in', 'one', 'basket', '.']
]
```

```python
tfidf.fit(docs)
tfidf.vocabulary_
    {'.': 0,
     'Don\'t': 1,
     'The': 2,
     'Two': 3,
     'a': 4,
     'all': 5,
     'basket': 6,
     'don\'t': 7,
     'eggs': 8,
     'in': 9,
     'is': 10,
     'make': 11,
     'mightier': 12,
     'one': 13,
     'pen': 14,
     'put': 15,
     'right': 16,
     'sword': 17,
     'than': 18,
     'the': 19,
     'wrongs': 20,
     'your': 21}
```

The next thing to keep in mind is that whenever you want to compute the tf-idf score for a document that is already tokenized you should wrap it in a list when you call the `transform()` method from `TfidfVectorizer`, so that it is handled as a single document instead of interpreting each token as a document.

```python
doc = ['Don\'t', 'count', 'your', 'chickens', 'before', 'they', 'hatch']
```

```python
vector_1 = tfidf.transform(doc)
vector_2 = tfidf.transform([doc])
vector_1.shape
(7, 22)
vector_2.shape
(1, 22)
```