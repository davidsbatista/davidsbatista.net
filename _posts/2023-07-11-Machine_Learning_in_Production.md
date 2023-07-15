---
layout: post
title: Machine Learning in Production
date: 2023-06-27 00:00:00
tags: Transformers NLP
categories: [blog]
comments: true
disqus_identifier: 20230627
preview_pic: /assets/images/2023-07-11-Machine_Learning_in_Production.jpg
---

I recently did the Machine Learning Engineering for Production (MLOps) Specialization from Coursera. This blog post aims to give a quick review of the course and detail the topics discussed in the course. Overall I can say I did enjoy the course and learn something.




### __Introduction to Machine Learning in Production__


<figure>
  <img style="width: 75%; height: 75%" src="/assets/images/2023-07-11-Machine_Learning_in_Production_steps.png">
  <figcaption>Figure 1: Steps of a Machine Learning project.</figcaption>
</figure>

- Very high-level just to peek into what's coming 
- Goes through briefly through every step which is then detailed in the next courses, part of this specialisations



- Steps of an ML project
	- Scoping
		- define a project, identify a problem
		- diligence on feasibility and value
		- ethical considerations
			- value for society?
			- fair and free from bias?
			- any ethical concerns?
		- milestones and resources
			- ml metrics
			- software metrics
			- business metrics
			- resources
	
	- Data
		- Define data and establish a baseline
		-  Label and organize data
			- meta-data, data provenance and lineage
			- balanced train/dev/test splits
	
	- Modeling
		- model + hyperparameters + data
		- doing well on train data, test data and also on business metrics
		- literature search + open source
		- reasonable algorithm with good data will often perform a great algorithm with not-so-good data
		- Auditing framework
			- Brainstorm the ways the system might go wrong.
			" Performance on subsets of data (e.g., ethnicity, gender).
			" Prevalence of specific errors/outputs (e.g., FP, FN).
			" Performance on rare classes.
		- Experiment tracking
	
	- Deployment
		- Concept drift and Data drift
		- First deployment vs. maintenance
		- Deployment patterns
			- Shadow mode
			- Canary deployment
				- Roll out to small fraction (say 5%) of traffic initially.
				- Monitor system and ramp up traffic gradually.
			- Blue green deployment
				- router sends request to old/blue or to new/green
				- enable easy rollback
			- Monitoring
				- software/hardware metrics
				- input metrics
				- output metrics
				- Set thresholds for alarms, Adapt metrics and thresholds over time



- Deployment patterns
- Monitoring





### __Machine Learning Modeling Pipelines in Production__




### __Deploying Machine Learning Models in Production__

this one was of particular interest to me, mainly because of these topics:

- dimensionality reduction
- quantization and pruning
- distillation techniques
- interpretability


### __Machine Learning Data Lifecycle in Production__


- feature engineering, data transformation, data lineage and provenance, and how to rely on schemas to follow data evolution.
- Authors tend to oversell TensorFlow Extended, anyway, the main thing is to learn the concepts.





### __References__


- __[Introduction to Machine Learning in Production](https://www.coursera.org/learn/introduction-to-machine-learning-in-production)__

- __[Machine Learning Modeling Pipelines in Production](https://www.coursera.org/learn/machine-learning-modeling-pipelines-in-production)__

- __[Deploying Machine Learning Models in Production](https://www.coursera.org/learn/deploying-machine-learning-models-in-production)__

- __[Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production)__