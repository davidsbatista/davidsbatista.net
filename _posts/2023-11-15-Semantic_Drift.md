---
layout: post
title: Semantic Drift in Machine Learning
date: 2023-11-15 00:00:00
tags: semantic-drift classification
categories: [blog]
comments: true
disqus_identifier: 20231115
preview_pic: /assets/images/2023-11-15-Types_of_Drift.png
---

Machine Learning models are static artefacts based on historical data, which start consuming consuming “real-world” data when deployed into production. Real-world data might not reflected the historical training data and test data, the progressive changes between training data and “real-world” are called the drift and it can be one of the reasons model accuracy decreases over time.


### __How can drift occur?__

<br>


#### Sudden: A new concept occurs within a short time.
![](https://web-api.textin.com/ocr_image/external/0a4238fef9f5c97d.jpg)
<br>


#### Gradual: A new concept gradually replaces an old one over a period of time.
![](https://web-api.textin.com/ocr_image/external/7d86fe47764e01c7.jpg)
<br>


### Incremental: An old concept incrementally changes to a new concept.
![](https://web-api.textin.com/ocr_image/external/433a00b759c6acf7.jpg)
<br>


#### Reoccurring: An old concept may reoccur after some time.
![](https://web-api.textin.com/ocr_image/external/4cf0ee50ce8c7b78.jpg)
<br>




### __Drift detection in a nutshell__

1) Create two datasets: reference and current/serving

2) Calculate statistical measures for features values in the reference (e.g.: distribution of values over features)

3) The reference dataset act as the benchmark/baseline

4) Select “real-world” data - current/serving

5) Calculate the same statistical measures as for the reference

6) Compare them both using statistical tools (e.g.: distance metrics, test hypothesis)

7) Depending on a threshold assume or not drift occurred

NOTE: repeat steps 4 to 6 periodically





### __Drift, Skew, Shift__

• __Data Drift__: changes in the statistical properties of features over time, due seasonality, trends or unexpected events.

• __Concept Drift__: changes in the statistical properties of the labels over time, e.g.: mapping to labels in training remains static while changes in real-world.

• __Schema Skew__: training and service data do not conform to the same schema, e.g.: getting an integer when expecting a float, empty string vs None.

• __Distribution Skew__: divergence between training and serving datasets, e.g.: dataset shift caused by __covariate/concept shift__.


#### __Covariate Shift__

Marginal distribution of features $$x$$ is not the same during training and serving, but the conditional distribution remains unchanged. 

_Example_: number of predictions of relevant/non-relevant text samples is in line with development test set, but the distribution of features is different between training.


#### __Concept Shift__

Conditional distribution of labels and features are not the same during training and serving, but the marginal distribution features remain unchanged. 

_Example_: although the text samples being crawled did not change and the distribution of features values is still the same, what determines if a text sample is relevant or non-relevant changed.


<img src="/assets/images/2023-11-15-Covariate_Concept_Shift.png" width="85%" height="auto" alt="Covariate and concept shift">


#### __How to detect them?__


<img src="/assets/images/2023-11-15-How_to_Detect.png" width="85%" height="auto" alt="How to detect covariate and concept shift">



### __Measuring Embeddings Drift__


- Average the embeddings in the current and reference dataset, compare with some similarity/distance metric: Euclidean distance, Cosine similarity;

- Train a binary classification to discriminate between reference and current distributions. If the model can confidently identify which embeddings belong to which you can consider the two datasets differ significantly.


<img src="/assets/images/2023-11-15-embeddings_classifer.png" width="65%" height="auto" alt="Train classifier to discriminate embeddings">


### __Share of drifted components__

- Embeddings, can be seen as a structured tabular dataset. Rows are individual texts and columns are components of each embedding

- Treat each component as a numerical "feature" and check for the drift in its distribution between reference and current datasets. 

- If many embedding components drift, you can consider that there is a meaningful change in the data. 


<img src="/assets/images/2023-11-15-embeddings_components_drift.png" width="65%" height="auto" alt="Embeddings Components Drifting">


### __References__
- __[“Machine Learning Engineering for Production” - Specialisation from Coursera](https://www.davidsbatista.net/blog/2023/06/27/Machine_Learning_in_Production/)__
- __[“Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift“](https://arxiv.org/abs/1810.11953)__
- __[Towards Data Science - Measuring Embedding Drift](https://medium.com/towards-data-science/measuring-embedding-drift-aa9b7ddb84ae)__
- __[Evidently AI blog Embedding Drift Detection](https://www.evidentlyai.com/blog/embedding-drift-detection)__