---
layout: post
title: nervaluate: Evaluating Named-Entity Recognition systems considering partial entity matching
date: 2025-05-18 00:00:00
tags: ner evaluation
categories: [blog]
comments: true
disqus_identifier: 20250518
preview_pic: 
---

Named-Entity Recognition (NER) is a now well established Natural Language Processing (NLP) task, where a system identifies sequences of tokens and classifies them into a specific category. The sequence of tokens identified by a NER system form named entities which can be further used for other tasks such as question answering, relationship extraction, information extraction or knowledge base population.

One important step in the development of a NER system is the evaluation, where the output of the system, for a given input, is compared against a manual annotation for the same input. From this comparison, several evaluation metrics can be derived, which together with an evaluation schema indicate the performance of the system, and different insights can be gathered, for instance, how well is the system doing for a specific entity type. This evaluation process is typically used to guide the development process of the NER system.

This paper is structured as follows: in Section 2 we reference and briefly describe proposed schemas for the evaluation of NER systems, in Section 3 we detail all the possible considered scenarios when evaluating the NER output against a human annotation. In Section 4 we introduce `nerevaluate`, a Python package to perform NER evaluation considering different metrics and scenarios, in Section 5 we showcase the use of the tool to evaluate a NER system on datasets. Finally in we finish this paper in Section 6 with conclusions and some considerations for future work.

## Related Work

Throughout the years, different NER evaluation challenges proposed several evaluation metrics and schemas.

### The Message Understanding Conference

The Message Understanding Conference (MUC) introduced detailed metrics in an evaluation considering different scoring categories measured by comparing the response of a system against the golden annotation. Not all scoring metrics introduced by MUC can be computed automatically, some metrics need interactive human judgment actions. Since the scope of this article is on automatic methods of evaluating a NER system the following scoring categories from MUC are relevant:

- Correct (COR): both are the same;
- Partial (PAR): system and the golden annotation are somewhat “similar” but not the same;
- Incorrect (INC): the output of a system and the golden annotation don’t match;
- Spurious (SPU): system produces a response which doesn’t exist in the golden annotation;
- Missing (MIS): a golden annotation is not captured by a system;
- Possible (POS): Possible is the sum of the correct, partial, incorrect, and missing.
- Actual (ACT): Actual is the sum of the correct, partial, incorrect, and spurious.

Based on the scoring categories defined above the authors define then following evaluation metrics, for each entity:

- Precision: (COR + PART × 0.5) / POS
- Recall: (COR + PART × 0.5) / ACT
- Undergeneration: MIS / POS
- Overgeneration: SPU / POS

| Scenario | Golden Standard Type | Golden Standard String | System Prediction Type | System Prediction String | Evaluation Schema Type | Evaluation Schema Partial | Evaluation Schema Exact | Evaluation Schema Strict |
|----------|----------------------|------------------------|------------------------|--------------------------|------------------------|---------------------------|-------------------------|--------------------------|
| I        | drug                 | phenytoin              | drug                   | phenytoin                | COR                    | COR                       | COR                     | COR                      |
| II       |                      |                        | brand                  | healthy                  | SPU                    | SPU                       | SPU                     | SPU                      |
| III      | brand                | TIKOSYN                |                        |                          | MIS                    | MIS                       | MIS                     | MIS                      |
| IV       | drug                 | propranolol            | brand                  | propranolol              | INC                    | COR                       | COR                     | INC                      |
| V        | drug                 | warfarin               | drug                   | of warfarin              | COR                    | PAR                       | INC                     | INC                      |
| VI       | group                | contraceptives         | drug                   | oral contraceptives      | INC                    | PAR                       | INC                     | INC                      |

### International Workshop on Semantic Evaluation (SemEval 2013)

The SemEval’13 introduced four different ways to measure Precision, Recall and F1-score based on the metrics defined by MUC.

- Strict: exact boundary surface string match and entity type
- Exact: exact boundary match over the surface string, regardless of the type;
- Partial: partial boundary match over the surface string, regardless of the type;
- Type: some overlap between the system-tagged entity and the gold annotation is required;

## Evaluation package: `nerevaluate`

Features:
- partial match scoring, which is implemented in nervaluate
- option to include/exclude the 'O' from the global evaluation
- option to group B and I results in the same entity
- report invalid transitions

## Demo and Experiments

## Conclusions and Future Work
