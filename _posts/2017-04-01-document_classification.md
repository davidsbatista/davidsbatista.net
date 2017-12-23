---
layout: post
title: Document Classification
comments: true
disqus_identifier: 20170401
date: 2017-04-01 00:14:00
categories: [blog]
tags: [classification, multi-label, sklearn, tf-idf, word2vec, doc2vec]
comments: true
preview_pic: /assets/images/2017-04-01-IMDB-movie-genre-classification.png
description: An introduction to Document Classification
---

Classifying a document into a pre-defined category is a common problem, for instance, classifying an email as spam or not spam. In this case there is an instance to be classified into one of two possible classes, i.e. binary classification.

However, there are other scenarios, for instance, when one needs to classify a document into one of more than two classes, i.e., multi-class, and even more complex, when each document can be assigned to more than one class, i.e. multi-label or multi-output classification.

In this post I will show an approach to classify a document into a set of pre-defined categories using different supervised classifiers and text representations. I will use the [IMDB dataset of movies](http://www.imdb.com/interfaces). Although the dataset contains several informations about a movie, for the scope of this post I will only use the plot of the movie and the genre(s) on which the movie is classified.


## Dataset

In order to create the dataset for this experiment you need to download _genres.list_ and _plot.list_ files from a [mirror FTP](ftp://ftp.fu-berlin.de/pub/misc/movies/database/), and then parse files in order to associate the titles, plots, and genres.

I've already done this step, and parsed both files in order to generate a single file, available here [movies_genres.csv](https://github.com/davidsbatista/text-classification/blob/master/movies_genres.csv.bz2), containing the plot and the genres associated to each movie.


## Pre-processing and cleaning

I started by doing some exploratory analysis on the IMDB dataset


{% highlight python %}
%matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("movies_genres.csv", delimiter='\t')
df.info()
{% endhighlight %}


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 117352 entries, 0 to 117351
    Data columns (total 30 columns):
    title          117352 non-null object
    plot           117352 non-null object
    Action         117352 non-null int64
    Adult          117352 non-null int64
    Adventure      117352 non-null int64
    Animation      117352 non-null int64
    Biography      117352 non-null int64
    Comedy         117352 non-null int64
    Crime          117352 non-null int64
    Documentary    117352 non-null int64
    Drama          117352 non-null int64
    Family         117352 non-null int64
    Fantasy        117352 non-null int64
    Game-Show      117352 non-null int64
    History        117352 non-null int64
    Horror         117352 non-null int64
    Lifestyle      117352 non-null int64
    Music          117352 non-null int64
    Musical        117352 non-null int64
    Mystery        117352 non-null int64
    News           117352 non-null int64
    Reality-TV     117352 non-null int64
    Romance        117352 non-null int64
    Sci-Fi         117352 non-null int64
    Short          117352 non-null int64
    Sport          117352 non-null int64
    Talk-Show      117352 non-null int64
    Thriller       117352 non-null int64
    War            117352 non-null int64
    Western        117352 non-null int64
    dtypes: int64(28), object(2)
    memory usage: 26.9+ MB


We have a total of 117 352 movies and each of them is associated with 28 possible genres. The genres columns simply contain a 1 or 0 depending of whether the movie is classified into that particular genre or not. This means the multi-label binary mask is already provided in this file.


Next we are going to calculate the absolute number of movies per genre. Note: each movie can be associated with more than one genre, we just want to know which genres have more movies.


{% highlight python %}
df_genres = df.drop(['plot', 'title'], axis=1)
counts = []
categories = list(df_genres.columns.values)
for i in categories:
    counts.append((i, df_genres[i].sum()))
df_stats = pd.DataFrame(counts, columns=['genre', '#movies'])
df_stats
{% endhighlight python %}


<div>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>#movies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Action</td>
      <td>12381</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adult</td>
      <td>61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adventure</td>
      <td>10245</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Animation</td>
      <td>11375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Biography</td>
      <td>1385</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Comedy</td>
      <td>33875</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Crime</td>
      <td>15133</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Documentary</td>
      <td>12020</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Drama</td>
      <td>46017</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Family</td>
      <td>15442</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Fantasy</td>
      <td>7103</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Game-Show</td>
      <td>2048</td>
    </tr>
    <tr>
      <th>12</th>
      <td>History</td>
      <td>2662</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Horror</td>
      <td>2571</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Lifestyle</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Music</td>
      <td>2841</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Musical</td>
      <td>596</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Mystery</td>
      <td>12030</td>
    </tr>
    <tr>
      <th>18</th>
      <td>News</td>
      <td>3946</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Reality-TV</td>
      <td>12338</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Romance</td>
      <td>19242</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Sci-Fi</td>
      <td>8658</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Short</td>
      <td>578</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Sport</td>
      <td>1947</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Talk-Show</td>
      <td>5254</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Thriller</td>
      <td>8856</td>
    </tr>
    <tr>
      <th>26</th>
      <td>War</td>
      <td>1407</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Western</td>
      <td>2761</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}
df_stats.plot(x='genre', y='#movies', kind='bar', legend=False, grid=True, figsize=(15, 8))
{% endhighlight python %}

<figure>
  <img class="fullw" src="/assets/images/2017-04-01-IMDB-movie-genre-classification.png">
</figure>


Since the `Lifestyle` has 0 instances we can just remove it from the data set


{% highlight python %}
df.drop('Lifestyle', axis=1, inplace=True)
{% endhighlight python %}

One thing that notice when working with this dataset is that there are plots written in different languages. Let's use [langedetect](https://pypi.python.org/pypi/langdetect?) tool to identify the language in which the plots are written


{% highlight python %}
from langdetect import detect
df['plot_lang'] = df.apply(lambda row: detect(row['plot'].decode("utf8")), axis=1)
df['plot_lang'].value_counts()
{% endhighlight python %}


    en    117196
    nl       120
    de        14
    da         6
    it         6
    pt         2
    fr         2
    no         2
    hu         1
    es         1
    sl         1
    sv         1
    Name: plot_lang, dtype: int64



There other languages besides English, let's just keep English plots, and save this to a new file.


{% highlight python %}
df = df[df.plot_lang.isin(['en'])]
df.to_csv("movies_genres_en.csv", sep='\t', encoding='utf-8', index=False)
{% endhighlight python %}

----


## Vector Representation and Classification

For vector representation and I will use two Python packages:

* [sklearn](http://scikit-learn.org/)
* [gensim](https://radimrehurek.com/gensim/)

To train supervised classifiers, we first need to transform the plot into a vector of numbers. I will explore 3 different vector representations:

* TF-IDF weighted vectors
* word2vec embeddings
* doc2vec embeddings


After having this vector representations of the text we can train supervised classifiers to train unseen plots and predict the genres on which they fall.

#### TF-IDF

Based on the [bag-of-words model](https://www.wikiwand.com/en/Bag-of-words_model), i.e., no word order is kept. I considered [TF-IDF](https://www.wikiwand.com/en/Tf%E2%80%93idf) weighted vectors, composed of different [_n_-grams](https://www.wikiwand.com/en/N-gram) size, namely: uni-grams, bi-grams and tri-grams. I also experimentally eliminated words that appear in more than a given number of documents. All this features can be easily configured with [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) class.

The _max_df_ parameter is used for removing terms that appear too frequently, i.e., _max_df_ = 0.50 means “ignore terms that appear in more than 50% of the documents”. The _ngram_range_ parameter selects how large are the sequence of words to be considered.

{% highlight python %}
tfidf__max_df: (0,25 0.50, 0.75)
tfidf__ngram_range: ((1, 1), (1, 2), (1, 3))
{% endhighlight python %}



#### Word2Vec

Under this scenario, a movie plot is represented by a single real-value dense vector based on the [word embeddings](https://www.wikiwand.com/en/Word_embedding) associated with each word. This is done by selecting words from the plot, based on their [part-of-speech (PoS)-tags](http://universaldependencies.org/u/pos/), and then summing their word embeddings and averaging them into a single vector. I used the [GoogleNews-vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), which have dimension of 300 and are derived from English corpora. For this experiment I selected only adjectives and nouns;



#### Doc2Vec

Doc2Vec is an [extension made over Word2Vec](https://www.wikiwand.com/en/Word2vec#/Extensions), which tries do model a single document or paragraph as a unique a single real-value dense vector. You can read more about it in the [original paper](http://proceedings.mlr.press/v32/le14.pdf). I will use the gensim implementation to derive vectors based on a single document.


At the of this post you have a link to the complete code, showing how to generate embeddings with word2vec and doc2vec.


### Load pre-processed data:

First we are going to load the pre-processed and cleaned data into the proper data structures which serve as input for the sklearn classifiers:

{% highlight python %}
data_df = pd.read_csv("movies_genres_en.csv", delimiter='\t')

# split the data, leave 1/3 out for testing
data_x = data_df[['plot']].as_matrix()
data_y = data_df.drop(['title', 'plot', 'plot_lang'], axis=1).as_matrix()
stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33)
for train_index, test_index in stratified_split.split(data_x, data_y):
    x_train, x_test = data_x[train_index], data_x[test_index]
    y_train, y_test = data_y[train_index], data_y[test_index]

# transform matrix of plots into lists to pass to a TfidfVectorizer
train_x = [x[0].strip() for x in x_train.tolist()]
test_x = [x[0].strip() for x in x_test.tolist()]
{% endhighlight python %}


After loading the data I also split the data into two sets:

* 2/3 ~ 66.6% of the data for tuning the parameters of the classifiers
* 1/3 ~ 33.3% will be used to test the performance of the classifiers


To achieve this I used the [StratifiedShuffleSplit class](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) which return stratified randomized folds, preserving the percentage of samples for each class.


In order to experiment with different features for the text representation and tunning the different parameters of the classifiers I used sklearn [Pipeline]() and [GridSearchCV](). I also use another class, to help transform binary classifiers into multi-label/multi-output classifiers, concretely [OneVsRestClassifier](), this class wraps ups the process of training a classifier for each possible class.

I considered the following supervise algorithms

* Naive Bayes
* SVM linear
* Logistic Regression

Note that Naive Bayes and Logistic Regression inherently support multi-class, but we are in a multi-label scenario, that's the reason why even these are wrapped in the OneVsRestClassifier process.

#### Naive Bayes

{% highlight python %}
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))),
])
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__estimator__alpha': (1e-2, 1e-3)
}
{% endhighlight python %}



#### SVM linear

{% highlight python %}
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(LinearSVC()),
])
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}
{% endhighlight python %}


#### Logistic Regression

{% highlight python %}
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag')),
])
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}
{% endhighlight python %}

### Parameter tunning through GridSearchCV

We then pass the built pipeline into a GridSearchCV object, and find the best parameters for both the bag-of-words representation and the classifier.

{% highlight python %}
grid_search_tune = GridSearchCV(
    pipeline, parameters, cv=2, n_jobs=2, verbose=3)
grid_search_tune.fit(train_x, train_y)

print
print("Best parameters set:")
print grid_search_tune.best_estimator_.steps
print

# measuring performance on test set
print "Applying best classifier on test data:"
best_clf = grid_search_tune.best_estimator_
predictions = best_clf.predict(test_x)

print classification_report(test_y, predictions, target_names=genres)
{% endhighlight python %}



## Results

### TF-IDF

    Naive Bayes best parameters set:
        TfidfVectorizer: max_df=0.25, ngram_range=(1, 3)
        MultinomialNB: alpha=0.001

    Linear SVM best parameters set:
        TfidfVectorizer: max_df=0.25 ngram_range=(1, 2)
        LinearSVC:  C=1, class_weight='balanced'

    LogisticRegression best parameters set:
        TfidfVectorizer:  max_df=0.75, ngram_range=(1, 2)
        LogisticRegression: C=1, class_weight='balanced'


                         precision    recall   f1-score   

    Naive Bayes            0.95        0.76      0.84     
    Linear SVM             0.89        0.86      0.87     
    LogisticRegression     0.70        0.89      0.78     




### Word2Vec

    Linear SVM best parameters set:
        LinearSVC:  C=1, class_weight=None

    LogisticRegression best parameters set:
        LogisticRegression: C=1, class_weight=None

                         precision    recall   f1-score   

    Linear SVM             0.68        0.37      0.45
    LogisticRegression     0.67        0.40      0.48




### Doc2Vec

    Linear SVM best parameters set:
        LinearSVC:  C=0.1, class_weight=None

    LogisticRegression best parameters set:
        LogisticRegression: C=1, class_weight=None

                         precision    recall   f1-score   

    Linear SVM             0.69        0.31      0.40
    LogisticRegression     0.65        0.36      0.45




### Conclusion

The best results are achieved with a Linear SVM and TF-IDF representation of the text, below you can see the results by genre.


    Best parameters set:
        TfidfVectorizer(max_df=0.25, ngram_range=(1, 2))
            LinearSVC(C=1, class_weight='balanced')

                  precision    recall  f1-score   support

         Action       0.89      0.84      0.86      4046
          Adult       1.00      0.67      0.80        21
      Adventure       0.89      0.81      0.85      3415
      Animation       0.92      0.86      0.89      3780
      Biography       0.95      0.58      0.72       491
         Comedy       0.89      0.87      0.88     11236
          Crime       0.86      0.90      0.88      4974
    Documentary       0.84      0.83      0.84      3986
          Drama       0.89      0.94      0.91     15110
         Family       0.89      0.84      0.86      5160
        Fantasy       0.90      0.79      0.84      2381
      Game-Show       0.95      0.87      0.91       730
        History       0.86      0.70      0.77       853
         Horror       0.93      0.66      0.77       826
          Music       0.92      0.82      0.87       951
        Musical       0.96      0.58      0.73       190
        Mystery       0.82      0.85      0.84      3918
           News       0.91      0.83      0.87      1337
     Reality-TV       0.89      0.85      0.87      4057
        Romance       0.90      0.90      0.90      6472
         Sci-Fi       0.90      0.83      0.86      2853
          Short       1.00      0.48      0.65       183
          Sport       0.91      0.73      0.81       616
      Talk-Show       0.89      0.87      0.88      1775
       Thriller       0.86      0.78      0.82      2914
            War       0.91      0.79      0.84       447
        Western       0.96      0.86      0.91       874

    avg / total       0.89      0.86      0.87     83596


The embeddings methods shows very low results, the representation based on the word2vec was just a naive way to get sentence embeddings, more robust methods could be explored like concatenating each words vector into a single vector, and give it as input to a neural network.

The doc2vec vectors were generated with gensim out-of-the-box, some parameter tunning on vectors generation process might give better results.

Also, word2vec and doc2vec, since they have a much lower dimension, i.e. 300 compared to 50 000 up to 100 000 of the TF-IDF weighted vectors, could probably be achieved with a non-linear kernel.


The full code for this post is available on my github:

[https://github.com/davidsbatista/text-classification](https://github.com/davidsbatista/text-classification)

-----


