---
layout: post
title: nervaluate - evaluating NER with partial entity matching
date: 2025-05-18 00:00:00
tags: ner evaluation
categories: [blog]
comments: true
disqus_identifier: 20250518
preview_pic: /assets/images/2025-05-18-NERvaluate.png
---

One important step in the development of a NER system is the evaluation, where the output of the system, for a given input, is compared against a manual annotation for the same input. From this comparison, several evaluation metrics can be derived, which together with an evaluation schema indicate the performance of the system, and different insights can be gathered, for instance, how well is the system doing for a specific entity type. This evaluation process is typically used to guide the development process of the NER system.

## nervaluate

Back in 2018 I __[wrote a blog post](/blog/2018/05/09/Named_Entity_Evaluation/)__ about named-entity evaluation metrics based on entity-level. It gain some visibility and Matthew Upson from Mantis NLP suggest me to make a pip package out of the code from the blog post, and so we started working together on it.

The evaluation metrics output by __[nervaluate](https://pypi.org/project/nervaluate)__ go beyond a simple token/tag based schema, and consider different scenarios  based on whether all the tokens that belong to a named entity were classified or not, and also whether the correct  entity type was assigned.






