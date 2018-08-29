---
layout: post
title: Hyperparameter optimization across multiple models in scikit-learn
comments: true
disqus_identifier: 20180223
date: 2018-02-23 00:0:00
categories: blog
tags: scikit-learn grid-search hyperparameter-optimization
comments: true
preview_pic: /assets/images/2018-02-23-model_optimization.png
description: This blog post shows how to perform hyperparameter optimization across multiple models in scikit-learn, using a helper class one can tune several models at once and print a report with the results and parameters settings.
---

I found myself, from time to time, always bumping into a piece of code (written by someone else) to perform grid search across different models in scikit-learn and always adapting it to suit my needs, and fixing it, since it contained some already deprecated calls. I finally decided to post it here in my blog, so I can quickly find it and also to share it with whoever needs it.

The idea is pretty simple, you pass two dictionaries to a helper class: the models and the the parameters; then you call the fit method, wait until everything runs, and after you call the summary() method to have a nice DataFrame with the report for each model instance, according to the parameters.

The credit for the code below goes to [Panagiotis Katsaroumpas](http://www.codiply.com/) who initially wrote it, I just fix it, since it was breaking with newer versions of scikit-learn, and also failed in Python 3. The original version is on this [blog post](http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/).

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
```

The code above defines the helper class, now you need to pass it a dictionary of models and a dictionary of parameters for each of the models.


```python
from sklearn import datasets

breast_cancer = datasets.load_breast_cancer()
X_cancer = breast_cancer.data
y_cancer = breast_cancer.target

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

models1 = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params1 = {
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ]
}
```

You create a `EstimatorSelectionHelper` by passing the models and the parameters, and then call the `fit()` function, which as signature similar to the original `GridSearchCV` object.

```python
helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_cancer, y_cancer, scoring='f1', n_jobs=2)
```

    Running GridSearchCV for ExtraTreesClassifier.
    Fitting 3 folds for each of 2 candidates, totalling 6 fits

    Running GridSearchCV for RandomForestClassifier.
    Fitting 3 folds for each of 2 candidates, totalling 6 fits

    Running GridSearchCV for GradientBoostingClassifier.
    Fitting 3 folds for each of 4 candidates, totalling 12 fits

    Running GridSearchCV for AdaBoostClassifier.
    Fitting 3 folds for each of 2 candidates, totalling 6 fits

    Running GridSearchCV for SVC.
    Fitting 3 folds for each of 6 candidates, totalling 18 fits

After the experiments has ran, you can inspect the results of each model and each parameters by calling the `score_summary` method.

```python
helper1.score_summary(sort_by='max_score')
```


<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estimator</th>
      <th>min_score</th>
      <th>mean_score</th>
      <th>max_score</th>
      <th>std_score</th>
      <th>C</th>
      <th>gamma</th>
      <th>kernel</th>
      <th>learning_rate</th>
      <th>n_estimators</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>AdaBoostClassifier</td>
      <td>0.962343</td>
      <td>0.974907</td>
      <td>0.991667</td>
      <td>0.0123335</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ExtraTreesClassifier</td>
      <td>0.966387</td>
      <td>0.973627</td>
      <td>0.987552</td>
      <td>0.00984908</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AdaBoostClassifier</td>
      <td>0.95279</td>
      <td>0.966463</td>
      <td>0.983333</td>
      <td>0.0126727</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RandomForestClassifier</td>
      <td>0.958678</td>
      <td>0.966758</td>
      <td>0.979253</td>
      <td>0.00896123</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GradientBoostingClassifier</td>
      <td>0.917031</td>
      <td>0.947595</td>
      <td>0.979253</td>
      <td>0.025414</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.8</td>
      <td>16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GradientBoostingClassifier</td>
      <td>0.950413</td>
      <td>0.962373</td>
      <td>0.979079</td>
      <td>0.0121747</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>32</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GradientBoostingClassifier</td>
      <td>0.95279</td>
      <td>0.966317</td>
      <td>0.975207</td>
      <td>0.00972142</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.8</td>
      <td>32</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GradientBoostingClassifier</td>
      <td>0.950413</td>
      <td>0.962548</td>
      <td>0.975207</td>
      <td>0.0101286</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>16</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SVC</td>
      <td>0.95122</td>
      <td>0.961108</td>
      <td>0.975207</td>
      <td>0.0102354</td>
      <td>1</td>
      <td>NaN</td>
      <td>linear</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestClassifier</td>
      <td>0.953191</td>
      <td>0.960593</td>
      <td>0.975</td>
      <td>0.0101888</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ExtraTreesClassifier</td>
      <td>0.958678</td>
      <td>0.96666</td>
      <td>0.974359</td>
      <td>0.00640498</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SVC</td>
      <td>0.961373</td>
      <td>0.963747</td>
      <td>0.967213</td>
      <td>0.00250593</td>
      <td>10</td>
      <td>NaN</td>
      <td>linear</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SVC</td>
      <td>0.935484</td>
      <td>0.945366</td>
      <td>0.955466</td>
      <td>0.00815896</td>
      <td>10</td>
      <td>0.0001</td>
      <td>rbf</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SVC</td>
      <td>0.934959</td>
      <td>0.946564</td>
      <td>0.954733</td>
      <td>0.00843008</td>
      <td>1</td>
      <td>0.0001</td>
      <td>rbf</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SVC</td>
      <td>0.926407</td>
      <td>0.936624</td>
      <td>0.94958</td>
      <td>0.00965657</td>
      <td>1</td>
      <td>0.001</td>
      <td>rbf</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SVC</td>
      <td>0.918455</td>
      <td>0.929334</td>
      <td>0.940678</td>
      <td>0.00907845</td>
      <td>10</td>
      <td>0.001</td>
      <td>rbf</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

The full code for this blog post is available in this [notebook](https://github.com/davidsbatista/machine-learning-notebooks/blob/master/hyperparameter-across-models.ipynb).

