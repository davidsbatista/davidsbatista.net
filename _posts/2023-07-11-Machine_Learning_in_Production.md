---
layout: post
title: Machine Learning in Production
date: 2023-06-27 00:00:00
tags: coursera mlops production deployment monitoring
categories: [blog]
comments: true
disqus_identifier: 20230627
preview_pic: /assets/images/2023-07-11-Machine_Learning_in_Production.jpg
---

I enrolled and successfully completed the [Machine Learning Engineering for Production (MLOps) Specialisation from Coursera](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops). This blog post aims to give a quick review of the course and detail the topics discussed, mostly based on the notes I took during the video lectures. The course covers a wide range of topics and sometimes it can feel overwhelming, which is another reason for me to write these notes

The specialisation follows the "Steps of a Machine Learning project" which is introduced in the first course and will follow throughout the 4 courses. Some of the concepts are brought up several times during the specialisation and in different courses, they are first briefly introduced and then shown in different contexts and with different levels of detail. 

One thing that stands out is that the specialisations tend to focus too much on TensorFlow-related solutions for the concepts and challenges presented throughout the course, but there are also many references to other software projects and solutions for the different challenges presented.

The specialisation is organised into 4 courses, and I will describe each one separately. I personally enjoyed most the 3rd and 4th courses, since it's where the instructors into more deep practical details. Some topics have more notes than others due to my personal interest or the novelty of the topic for me.


- [__1 -  Introduction to Machine Learning in Production__](#2---introduction-to-machine-learning-in-production)

- [__2 - Machine Learning Data Lifecycle in Production__](#2---machine-learning-data-lifecycle-in-production)

- [__3 - Machine Learning Modelling Pipelines in Production__](#3---machine-learning-modelling-pipelines-in-production)

- [__4 - Deploying Machine Learning Models in Production__](#4---deploying-machine-learning-models-in-production)


<br>

## __1 -  Introduction to Machine Learning in Production__

The first course is a high-level introduction to the topics covered in the specialisation, it briefly goes through the different steps of a Machine Learning Project, which are then detailed in the next courses.


<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_steps.png">
  <figcaption>Figure 1: Steps of a Machine Learning project.</figcaption>
</figure>


__Scoping__: the definition of an ML project. Identifying the problem, doing due diligence on the feasibility and value, and considering possible ethical concerns, milestones, and metrics.

__Data__: introduction to the definition of the data used in the project and an expected baseline, how to label and organize the data: meta-data, data provenance and lineage, balanced train/dev/test splits.

__Modelling__: how to approach the modelling of the data to solve the problem assuring the algorithm does well on the training data and test data but also positively impacts any business metrics. Auditing the framework by brainstorming ways the system might go wrong, e.g.: performance on subsets of data (e.g., ethnicity, gender), the prevalence of specific errors/outputs, and performance on rare classes.
	
__Deployment__: deployment vs. maintenance and deployment patterns. Shadow mode; Canary deployment: roll out to a small fraction (say 5%) of traffic initially monitor the system and ramp up traffic gradually;
blue-green deployment: router sends a request to old/blue or to new/green, it enables easy rollback. Monitoring: software/hardware metrics, input metrics, output metrics, thresholds for alarms, adapt metrics and thresholds over time

<br>

---

<br>

## __2 - Machine Learning Data Lifecycle in Production__

This course focuses mostly on the data aspect of a Machine Learning project, and it's organised into 4 main topics

### __Collecting, Labelling, Validating__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C2_-_Machine_Learning_Data_Lifecycle_in_Production/C2_W1.pdf))

The main focus of this topic is on the importance of data: the data pipeline, and data monitoring. It starts by explaining the data collection and labelling process, focusing on understanding the data source, the consistency of values, units and data types, detecting outliers, errors and inconsistent formatting. Mentions privacy and fairness aspects in the data collection, and the use of process feedback using logging tools, such as [logstash](https://github.com/elastic/logstash) and [fluentd](https://github.com/fluent/fluentd).

Mentions the problems with data, particularly the drift in data, which can be a consequence of trends and seasonality. The impact on the distribution of features and the relative importance of features. Here the instructors explain in great detail the concepts:

__Data Drift__: changes in data over time, i.e.: changes in the statistical properties of the features over time due to seasonality, trends or unexpected events.

__Concept Drift__: changes in the statistical properties of the labels over time, i.e.: mapping to labels in training remains static while the real-world changes.
	
__Schema Skew__: training and service data do not conform to the same schema, e.g.: getting an integer when expecting a float, empty string vs None

__Distribution Skew__: divergence between training and serving datasets, dataset shift caused by covariate/concept shift


<figure>
<img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_skew.png">
<figcaption>Figure 2: Detection distribution skew.</figcaption>
</figure>

__Dataset shift__: the joint probability of features $$x$$ and labels $$y$$ is not the same during training and serving

__Covariate shift__: a change in the distribution of the input variables present in training and serving data. In other words, it's where the marginal distribution of features $$x$$ is not the same during training and serving, but the conditional distribution remains unchanged.

__Concept shift__: refers to a change in the relationship between the input and output variables as opposed to the differences in the data distribution or input. It's when the conditional distribution of labels $$y$$ and n features $$x$$ are not the same during training and serving, but the marginal distribution features $$x$$ remain unchanged.

<br>

<figure>
<img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_skew_detection.png">
<figcaption>Figure 3: Skew detection workflow.</figcaption>
</figure>

__Software Tools__
- [Tensorflow data-validation](https://github.com/tensorflow/data-validation)
- [Great Expectations](https://github.com/great-expectations/great_expectations)
- [Deequ - Unit Tests for Data](https://github.com/awslabs/deequ)

### __Feature Engineering__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C2_-_Machine_Learning_Data_Lifecycle_in_Production/C2_W2.pdf))

An overview of the pre-processing operations and feature engineering techniques, e.g.: feature scaling, normalisation and standardisation, bucketing/binning. Also, a good summarisation of the techniques to reduce the dimensionality of features: PCA, t-SNE and UMAP. Lastly, how to combine multiple features into a new feature and the feature selection process.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_feature_selection.png">
  <figcaption>Figure 4: Supervised Feature Selection.</figcaption>
</figure>

### __Data Storage__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C2_-_Machine_Learning_Data_Lifecycle_in_Production/C2_W3.pdf))

This chapter deals with the data journey, accounting for data and model evolution and using metadata to track changes in data in the ML pipeline. How a schema to hold data can evolve and how to keep track of those changes, it also introduces the concept of feature stores, as well as Datawarehouse (OLAP) vs. Databases (OLTP) and data lakes.

### __Advanced Labelling, Augmentation and Data Preprocessing__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C2_-_Machine_Learning_Data_Lifecycle_in_Production/C2_W4.pdf))

The last topic covers data labelling and data augmentation techniques. __Semi-Supervised Labelling__, and briefly introduce graph-based label propagation, combines supervised and unsupervised data.

For __Active Learning__, a few strategies to select the best samples to be annotated by a human are introduced and explained:

- Margin sampling: label points the current model is least confident in.
- Cluster-based sampling: sample from well-formed clusters to "cover" the entire space.
- Query-by-committee: train an ensemble of models and sample points that generate disagreement.
- Region-based sampling: Runs several active learning algorithms in different partitions of the space.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_snorkel.png">
  <figcaption>Figure 5: Snorkel workflow.</figcaption>
</figure>


Lastly, for __Weak Supervision__, the instructors give the example of Snorkel:

- Start with unlabelled data, without ground-truth labels
- One or more weak supervision sources
- A list of heuristics that can automate labelling, typically provided by subject matter experts
- Noisy labels have a certain probability of being correct, but not 100%
- Objective: learn a generative model to determine weights for weak supervision sources
- Learn a supervised model

This topic also briefly explains how to do data augmentation techniques, mostly for images, and about windowing strategies for time series.


<br>

---

<br>


## __3 - Machine Learning Modelling Pipelines in Production__

The 3rd course on this specialisation is the longest one covering 5 topics and was also the one that brought, for me, the most interesting topics from all the 4 courses.

### __Neural Architectural Search and Auto ML__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C3_-_Machine_Learning_Modeling_Pipelines_in_Production/C3_W1.pdf))

The first chapter covers automatic parameter tuning. It describes briefly different strategies to find the best hyperparameters (i.e.: set before launching the learning process and not updated in each training step):

- Grid Search
- Random Search
- Bayesian Optimisation
- Evolutionary Algorithms
- Reinforcement Learning

Shows how the hyperparameters can be tuned with Keras Tuner, and it finishes by talking more broadly about the ML topic and the services provided by cloud providers to perform AutoML.

### __Model Resource Management Techniques__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C3_-_Machine_Learning_Modeling_Pipelines_in_Production/C3_W2.pdf))

This one was of particular interest to me, mainly because all of the subjects covered deal with how to make a model more efficient in terms of CPU/GPU needs. Essentially through methods of dimensionality reduction, quantisation and pruning.


__Dimensionality Reduction__: there's a brief explanation about the curse of dimensionality and the Hughes effect as motivation for dimensionality reduction, which is can be tackled with manual feature reduction techniques, and the instructors give a few examples and also explain and introduce a few algorithms:

- Unsupervised:
	- Principal Components Analysis (PCA)
	- Latent Semantic Indexing/Analysis (LSI and LSA) / Singular-Value Decomposition (SVD)
	- Independent Component Analysis (ICA)
- Non-Negative Matrix Factorisation (NMF)
- Latent Dirichlet Allocation (LDA)

__Quantisation and Pruning__

As the instructors explain, one of the motivation reasons for reducing model sizer is the deployment of ML models in IoT and mobile devices.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_quantisation.png">
  <figcaption>ToDo</figcaption>
</figure>

In a nutshell what post-training quantisation does is to efficiently convert or quantise the weights from floating point numbers to integers. This might reduce the precision representation and incur a small loss in model accuracy but significantly reduces the model size making it more feasible to run on a memory-constrained device.

Pruning aims to reduce the number of parameters and operations involved in generating a prediction by removing network connections, this reduces the model capacity, but also its size and complexity. 

The instructors also make a mention [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) with the hypothesis that "a randomly-initialized, dense neural network contains a subnetwork that is initialised such that — when trained in isolation — it can match the test accuracy of the original network after training for at most the same number of iterations"


Reading:

- [Quantisation and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) 

- [The Lottery Ticket Hypothesis - Patrick Liu Notes](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/lottery_ticket_hypothesis.html)

### __High-Performance Modelling and Distillation Techniques__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C3_-_Machine_Learning_Modeling_Pipelines_in_Production/C3_W3.pdf))

- High-Performance Modelling: 
	- distributed training, including a couple of different kinds of parallelism. 
	- Then we'll turn to high-performance modelling, including high-performance ingestion.

- Distillation Techniques
	- Knowledge Distillation
	- Teacher and student model

### __Model Analysis__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C3_-_Machine_Learning_Modeling_Pipelines_in_Production/C3_W4.pdf))

This topic covers the question of what's next after the model is trained and deployed, and its processing data. Is it performing well? Can we improve it? Has the data changed from the training dataset? 

One first aspect is how to compute metrics, most of the time, metrics are calculated on the entire dataset, and slicing deals with understanding how the model is performing on each subset of data. This is demonstrated using [Tensorflow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma) to perform top-level aggregate metrics versus slicing.

The chapter then covers different model debugging techniques:

#### __Adversarial Attacks__

- Random Attacks: expose models to high volumes of random input data
- Partial dependence plots
	
	-- [Python Partial Dependence Plot toolbox](https://github.com/SauceCat/PDPbox)
	
	-- [PyCEBbox](https://github.com/AustinRochford/PyCEbox)

- Measuring your vulnerability to attack

	-- [Cleverhans](https://github.com/cleverhans-lab/cleverhans): benchmark machine learning systems' vulnerability to adversarial examples

	-- [Foolbox](https://github.com/bethgelab/foolbox): lets you easily run adversarial attacks against machine-learning models

#### __Residual Analysis__
- Measures the difference between the model’s predictions and ground truth
- Randomly distributed errors are good
- Correlated or systematic errors show that a model can be improved	 

#### __Model Remediation Techniques__
- Model editing: manual tweaks to adapt your use case
- Model assertions: implement business rules that override model predictions
 
#### __Fairness__
- Compute performance metrics at all slices of data
- Evaluate your metrics across multiple thresholds
- If the decision margin is small, report in more detail

#### __Continuous Evaluation and Monitoring__
- Training data is a snapshot of the world at a point in time and many types of data change over time.
- Concept drift: loss of prediction quality
	- Concept Emergence: a new type of data distribution
	- Types of dataset shift: covariate shift and prior probability shift
- Statistical process control 
- Sequential analysis
- Error distribution monitoring
- Feature distribution monitoring

### __Model Interpretability__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C3_-_Machine_Learning_Modeling_Pipelines_in_Production/C3_W5.pdf))

This chapter introduces the concept and the importance of __explainability in AI__ mentioning other related subjects such as fairness, privacy and security. But ultimately this chapter presents several methods to understand how and why ML models make certain predictions. 

It introduces different categories and properties for model interpretation methods, such as post-hoc: models as black boxes, extract relationships between features and model predictions; or model-specific methods (e.g.: interpretation of regression weights in linear models; it finishes with a detailed description of many agnostic models:

- Partial Dependence Plots 
- Permutation Feature Importance 
- Shapley Values
- SHAP
- Testing Concept Activation Vectors
- Local Interpretable Model-agnostic Explanations (LIME)

#### __Software__

- [Tensorflow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma)
- [Python Partial Dependence Plot toolbox](PDPbox)	
- [PyCEBbox](https://github.com/AustinRochford/PyCEbox)
- [Cleverhans](https://github.com/cleverhans-lab/cleverhans)
- [Foolbox](https://github.com/bethgelab/foolbox)


<br>

---

<br>


## __4 - Deploying Machine Learning Models in Production__

This chapter deals with deploying a trained model, exposing your model to the world outside, and dealing with incoming data requests. It focuses on metrics to optimize such as Latency, Cost, Throughput

### __Resources and Requirements for Serving Models__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C4_-_Deploying_Machine_Learning_Models_in_Production/C4_W1.pdf))

- ### __TODO__

### __Model Serving Architecture__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C4_-_Deploying_Machine_Learning_Models_in_Production/C4_W2.pdf))

Compares different patterns and infrastructure choices to deploy a model. It starts with the different aspects of deploying a model on-premises or on the cloud and describes different pre-built servers
- TensorFlow Serving
- NVIDIA Triton Inference Server
- Torch Serve
- Kubeflow KFServing

The topic then describes how horizontal scaling can be achieved with containers and orchestration tools, and helps scale the process of serving a model, they give examples with Kubernetes.

It then describes the paradigm of online inference, i.e.: generating machine learning predictions in real-time upon request and which optimisations can be done to decrease latency and increase throughput, giving special focus to data preprocessing.

The chapter ends with the batch inference paradigm within the context of ETLs and distributed processing.

### __Model Management and Delivery__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C4_-_Deploying_Machine_Learning_Models_in_Production/C4_W3.pdf))

This chapter deals with all the model management activities such as tracking model experiments and model versioning, after this, the chapter transitions into the MLops topic. It starts by giving an ML Solution Lifecycle, bridging ML and IT with MLops:

- Continuous Integration (CI): Testing and validating code, components, data, data schemas, and models
- Continuous Delivery (CD): Deploying model prediction service
- Continuous Training (CT): A process that automatically retrains candidate models for testing and serving
- Continuous Monitoring (CM): Catching errors in production systems, and monitoring production inference data and model performance metrics

A proposal on how to manage model versions, __MAJOR.MINOR.PIPELINE__
- MAJOR: Incompatibility in data or target variable
- MINOR: Model performance is improved
- PIPELINE: The pipeline of model training is changed

and it describes what a model registry can do.

The chapter ends by going into a very detailed and practical description of __Continuous Delivery__ and __Progressive Delivery__, an improvement over the former.

### __Model Monitoring__ ([slides](/assets/documents/Coursera-MLOps_Specialization/C4_-_Deploying_Machine_Learning_Models_in_Production/C4_W4.pdf))

The last chapter focuses on the last step of a Machine Learning project. It essentially reviews and consolidates concepts already mentioned in the course, but goes into a bit more detail. 

#### __Observability and Logging__ ####

The chapter introduces the concept of logging as a way of providing __observability__ of the model. __Logging__ should be used to keep track of the model inputs and predictions and detect potential red flags, e.g.: a feature becoming unavailable, or notable shifts in the distributions. 

Next, introduces the concept of __tracing for ML Systems__, mentioning some tools:

- Dapper
- Zipkin
- Jaeger

#### __Model Decay__ ####

It follows with a review of the causes of model decay, __data drift__: statistical properties of input changes; and __concept drift__: the relationship between features and label changes, the very meaning of what you are trying to predict changes.
Model Decay can be mitigated by first detecting drift through logging of request predictions and responses. By observing the statistical properties of logged data and comparing it with the training data one can detect drift.

Libraries for detecting drift:
- [TensorFlow Data Validation (TFDV)](https://github.com/tensorflow/data-validation)
- [Scikit-multiflow library](https://github.com/scikit-multiflow/scikit-multiflow)

Lastly, to mitigate the model drift, we need to deal with data:

- Determine the portion of your training set that is still correct
- Keep the good data, discard the bad, and add new data - OR -
- Discard data collected before a certain date and add new data - OR -
- Create an entirely new training dataset from new data

Model:

- Continue training your model, fine-tuning from the last checkpoint using new data - OR -
- Start over, reinitialise your model, and completely retrain it

#### __Responsible AI__ ####

This chapter ends by talking about Responsible AI, mainly:

Best practices:

-  Human-Centered Design: 
	- Model potential adverse feedback early in the design process
	- Engage with a diverse set of users and use-case scenarios
- Identify Multiple Metrics
- Analyse your raw data carefully: does your data reflect all your users?


Legal Requirements for Secure & Private AI:

- General Data Protection Regulation (GDPR)
- Informational Harms: unintended or unanticipated leakage of information
- Behavioural Harms: manipulating the behaviour of the model itself, impacting the predictions or outcomes of the model


The chapter ends on the __Anonymisation & Pseudonymisation__ topic, showing examples of how to anonymise data under GPDR, and 
talking about the __Right to Be Forgotten__, __Right to Rectification__ and other __Rights of the Data Subject__, all part of the GPDR.

---

### __References__

- __[1 - Introduction to Machine Learning in Production](https://www.coursera.org/learn/introduction-to-machine-learning-in-production) - [Lesson Slides](/assets/documents/Coursera-MLOps_Specialization/C1_-_Introduction_to_Machine_Learning_in_Production/)__

- __[2 - Machine Learning Modeling Pipelines in Production](https://www.coursera.org/learn/machine-learning-modeling-pipelines-in-production)  - [Lesson Slides](/assets/documents/Coursera-MLOps_Specialization/C2_-_Machine_Learning_Data_Lifecycle_in_Production/)__

- __[3 - Deploying Machine Learning Models in Production](https://www.coursera.org/learn/deploying-machine-learning-models-in-production) - [Lesson Slides](/assets/documents/Coursera-MLOps_Specialization/C3_-_Machine_Learning_Modeling_Pipelines_in_Production/)__

- __[4 - Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production) - [Lesson Slides](/assets/documents/Coursera-MLOps_Specialization/C4_-_Deploying_Machine_Learning_Models_in_Production/)__

- Figures 1, 2, 3 and 4 taken from












