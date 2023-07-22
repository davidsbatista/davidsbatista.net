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

I enrolled and successfully did the [Machine Learning Engineering for Production (MLOps) Specialisation from Coursera](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops). This blog post aims to give a quick review of the course and detail the topics discussed in the course. The course covers a range number of topics and sometimes it can feel overwhelming, which is one more reason for writing these notes, it's a way to review and consolidate what I've learned. 

The specialisation is organised into 4 courses. I will describe each one separately. I personally enjoyed a lot the 3 and 4 courses of the specialisations, it's where they go into details and give some practical tips.

<br>

<!--

- ML Pipelines
	- Pipeline orchestration frameworks
	- TensorFlow Extended (TFX)
-->


### __1 - Introduction to Machine Learning in Production__

The first course is a high-level introduction to the topics covered in the specialisation, it briefly goes through the different steps of a Machine Learning Project, which are then detailed in the next courses.


<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_steps.png">
  <figcaption>Figure 1: Steps of a Machine Learning project.</figcaption>
</figure>


__Scoping__: the definition of an ML project. Identifying the problem, doing due diligence on the feasibility and value, considering possible ethical concerns, and milestones and metrics.

__Data__: introduction to the definition of the data used in the project and an expected baseline, how to label and organize the data: meta-data, data provenance and lineage, balanced train/dev/test splits.

__Modelling__: how to approach the modelling of the data to solve the problem assuring the algorithm does well on the training data, test data but also has a positive impact on any business metrics. Auditing the framework by brainstorming ways the system might go wrong, e.g.: performance on subsets of data (e.g., ethnicity, gender), the prevalence of specific errors/outputs, and performance on rare classes.
	
__Deployment__: deployment vs. maintenance and deployment patterns. Shadow mode; Canary deployment: roll out to a small fraction (say 5%) of traffic initially monitor the system and ramp up traffic gradually;
blue-green deployment: router sends a request to old/blue or to new/green, it enables easy rollback. Monitoring: software/hardware metrics, input metrics, output metrics, thresholds for alarms, adapt metrics and thresholds over time


---


### __2 - Machine Learning Data Lifecycle in Production__

This course focuses mostly on the data aspect of a Machine Learning project, and it's organised into 4 main topics

#### __Collecting, Labelling, Validating__

The main focus of this topic is on the importance of data: the data pipeline, and data monitoring. It starts by explaining the data collection and labelling process, focusing on understanding the data source, the consistency of values, units and data types, detecting outliers, errors and inconsistent formatting. Mentions privacy and fairness aspects in the data collection, and the use of process feedback using logging tools, such as [logstash](https://github.com/elastic/logstash) and [fluentd](https://github.com/fluent/fluentd).

Mentions the problems with data, particularly the drift in data. Changes occur due to trends and seasonality, which have an impact on the distribution of features and the relative importance of features. Here the instructors introduce the concept of 


- Validating
	- Data and Concept change in Production ML
		- Model performance decays over time
			- Data and Concept drift

- Data Issues
	- Data Drift: 
		 - changes in data over time, such as data collected once a day
		 - changes in the statistical properties of the features over time
		 	- seasonality or trend, unexpected events
	
	- Concept Drift:
		- changes in the statistical properties of the labels over time
		- mapping to labels in training remains static while the real-world changes
	

	- Detecting Data Issues
		- detecting schema skew: Training and serving data do not conform to the same schema
		- detecting distribution skew: Dataset shift → covariate or concept shift


	- Schema Skew: 
		- training and service data do not conform to the same schema
		- examples:
			- getting an integer when expecting a float
			- empty string vs None

	- Distribution Skew
		- divergence between training and serving datasets (i.e., data being received at the 
		inference stage of an ML model)
		- dataset shift caused by covariant and concept shift
		
		
		<figure>
		  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_skew.png">
		  <figcaption>Figure 2: Detection distribution skew.</figcaption>
		</figure>


		<figure>
		  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_skew_detection.png">
		  <figcaption>Figure 3: Skew detection workflow.</figcaption>
		</figure>
		
		
		- Dataset shift
			the joint probability of x are features and y are labels is not the same during training and serving
		
		- Covariate shift
			Covariate shift refers to the change in the distribution of the input variables present in training and serving data. In other words, it's where the marginal distribution of x are features is not the same during training and serving, but the conditional distribution remains unchanged.
			
		- Concept
			Concept shift refers to a change in the relationship between the input and output variables as opposed to the differences in the Data Distribution or input itself.
			In other words, it's when the conditional distribution of y are labels given x are features is not the same during training and serving, but the marginal distribution of x are features remains unchanged.
			
		
		- Skew Detection
			- The first stage is looking at training data and computing baseline statistics and a reference schema.
			- Then you do basically the same with your serving data, you're going to generate the descriptive statistics.
			- Then you compare the two.
			You compare your serving baseline statistics and instances.
			You check for differences between that and your training data.
			You look for skew and drift.
			Significant changes become anomalies and they'll trigger an alert.
			That alert goes to whoever's monitoring system, that can either be a human or another system to analyze the change and decide on the proper course of action. That's got to be the remediation of the way that you're going to fix and react to that problem.

Software
- https://medium.com/datamindedbe/data-quality-libraries-the-right-fit-a6564641dfad
- https://github.com/tensorflow/data-validation
- https://github.com/great-expectations/great_expectations
- https://github.com/awslabs/deequ


#### __Feature Engineering__

An overview of the pre-processing operations and feature engineering techniques, e.g.: feature scaling, normalisation and standardisation, bucketing/binning. Also, a good summarisation of the techniques to reduce the dimensionality of features: PCA, t-SNE and UMAP. Lastly, how to combine multiple features into a new feature and the feature selection process.

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_feature_selection.png">
  <figcaption>Figure 2: Supervised Feature Selection.</figcaption>
</figure>


#### __Data Storage__

- Data Journey and Data Storage
- Accounting for data and model evolution
- Using ML metadata to track changes
- Schema Development
- Features Stores
- Datawarehouse (OLAP) vs Databases (OLTP)
- Data lakes


#### __Advanced Labelling, Augmentation and Data Preprocessing__

- Semi-supervised labelling: 
	- label propagation graph based
- Active Learning
	- Margin sampling: Label points the current model is least confident in.
	- Cluster-based sampling: sample from well-formed clusters to "cover" the entire space.
	- Query-by-committee: train an ensemble of models and sample points that generate disagreement.
	- Region-based sampling: Runs several active learning algorithms in different partitions of the space.
- Weak supervision with Snorkel
	-Unlabelled data, without ground-truth labels
		● One or more weak supervision sources
		○ A list of heuristics that can automate labelling
		○ Typically provided by subject matter experts
		● Noisy labels have a certain probability of being correct, but not 100%
		● Objective: learn a generative model to determine weights for weak supervision sources


---


### __3 - Machine Learning Modeling Pipelines in Production__

#### __Week 1__

Neural Architecture Search
Neural architecture search (NAS) is is a technique for automating the design of artificial neural networks
● It helps finding the optimal architecture
● This is a search over a huge space
● AutoML is an algorithm to automate this search  

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_hyperparameter_search_strategies.png">
  <figcaption>ToDo</figcaption>
</figure>

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_hyperparameter_grid_search_random_search.png">
  <figcaption>ToDo</figcaption>
</figure>

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_hyperparameter_bayesian_optimisation.png">
  <figcaption>ToDo</figcaption>
</figure>

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_hyperparameter_evolutionary_methods.png">
  <figcaption>ToDo</figcaption>
</figure>

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_hyperparameter_reinforcement_learning_1.png">
  <figcaption>ToDo</figcaption>
</figure>

<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_hyperparameter_reinforcement_learning_2.png">
  <figcaption>ToDo</figcaption>
</figure>

- Automated Machine Learning (AutoML)
- Automating hyperparameter tuning with Keras Tuner
- Microsoft Azure Automated Machine Learning
- Google Cloud AutoML

#### __Week 2__

this one was of particular interest to me, mainly because of these topics:

- Model Resource Management Techniques
	- C3_W2_Lab_2_Algorithmic_Dimensionality.ipynb
	- Dimensionality Reduction
		- PCA
		- Unsupervised:
			- Latent Semantic Indexing/Analysis (LSI and LSA) (SVD)
			- Independent Component Analysis (ICA)
		- Matrix Factorisation
			- Non-Negative Matrix Factorisation (NMF)
		- Latent Methods
			-  Latent Dirichlet Allocation (LDA)

	- Quantisation
		- Post-Training
			- What post-training quantisation basically does is efficiently convert, or more precisely, quantize the weights from floating point numbers to integers. 
			- reduced precision representation
			- incur a small loss in model accuracy
		- Quantisation-aware training (QAT)
			- The core idea is that quantization aware training simulates low precision inference time computation in the forward pass of the training process. By inserting fake quantization nodes, the rounding effects of quantization are assimilated in the forward pass, as it would normally occur in actual inference. The goal is to fine-tune the weights to adjust for the precision loss. If fake quantization nodes are included in the model graph at the points where quantization is expected to occur, for example, convolutions. Then in the forward pass, the flood values will be rounded to the specified number of levels to simulate the effects of quantization. This introduces the quantization error as noise during training and is part of the overall loss which the optimization algorithm tries to minimize. Here, the model learns parameters that are more robust to quantization. 

		- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
		
	- Pruning
		- Pruning aims to reduce the number of parameters and operations involved in generating a prediction by removing network connections. 
		- Reduce model search space/capacity
	    Finding Sparse Neural Networks
	    “A randomly-initialized, dense neural network contains a subnetwork that is initialised such that — when trained in isolation — it can match the test accuracy of the original network after training for at most the same number of iterations”
		- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
		- https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/lottery_ticket_hypothesis.html

#### __Week 3__

- High-Performance Modeling: 
	- distributed training, including a couple of different kinds of parallelism. 
	- Then we'll turn to high-performance modelling, including high-performance ingestion.

- Distillation Techniques
	- Knowledge Distillation: 
	- Teacher and student model

#### __Week 4__

- Model Analysis
- TensorFlow Model Analysis

#### __Week 5__

- Interpretability
- ....
- Shapley Value
- LIME
- .....


---


### __4 - Deploying Machine Learning Models in Production__

#### __Week 1__

- Introduction to Model Serving
- Resources and Requirements for Serving Models
- Tensorflow serving

#### __Week 2__

- Model Serving: Patterns and Infrastructure

- NVIDIA Triton Inference Server
- TensorFlow Serving Architecture
- Torch Serve
- KFServing
- Scalling
- Containers and Orchestration Tools
-  Batch Inference

#### __Week 3__

- Model Management and Delivery
- ML Experiments Management and Workflow Automation
- Experiment Tracking
- MLOps
- ML Solution Lifecycle
- Tensorflow Extended
- Managing Model Versions
- Continuous Delivery
- Progressive Delivery

#### __Week 4__

- Model Monitoring
- Logging for ML Monitoring
- Tracing for ML Systems
-  What is Model Decay?
	- Data Drift
	- Concept Drift
- Ways to Mitigate Model Decay
- Responsible AI
-  Legal Requirements for Secure & Private AI
-  Anonymization & Pseudonymisation


<!--
- feature engineering, data transformation, data lineage and provenance, and how to rely on schemas to follow data evolution.
- Authors tend to oversell TensorFlow Extended, anyway, the main thing is to learn the concepts.
-->


---


### __References__


- __[1 - Introduction to Machine Learning in Production](https://www.coursera.org/learn/introduction-to-machine-learning-in-production) - [Lesson Slides](/assets/documents/Coursera-MLOps_Specialization/C1_-_Introduction_to_Machine_Learning_in_Production/)__

- __[2 - Machine Learning Modeling Pipelines in Production](https://www.coursera.org/learn/machine-learning-modeling-pipelines-in-production)  - [Lesson Slides](/assets/documents/Coursera-MLOps_Specialization/C2_-_Machine_Learning_Data_Lifecycle_in_Production/)__

- __[3 - Deploying Machine Learning Models in Production](https://www.coursera.org/learn/deploying-machine-learning-models-in-production) - [Lesson Slides](/assets/documents/Coursera-MLOps_Specialization/C3_-_Machine_Learning_Modeling_Pipelines_in_Production/)__

- __[4 - Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production) - [Lesson Slides](/assets/documents/Coursera-MLOps_Specialization/C4_-_Deploying_Machine_Learning_Models_in_Production/)__













