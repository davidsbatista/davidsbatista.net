---
layout: post
title: Conditional Random Fields
date: 2017-08-11 00:00:00
tags: [conditional random fields, sequence modelling, tutorial]
categories: [blog]
comments: true
disqus_identifier: 20170811
preview_pic:
---

This is the third and the last part of a series of posts about sequential supervised learning applied to NLP.

<!--

\newcommand{\argmax}[1]{\underset{#1}{\operatorname{arg}\,\operatorname{max}}\;}

https://liqiangguo.wordpress.com/page/2/
http://www.cs.columbia.edu/~smaskey/CS6998/slides/statnlp_week10.pdf
http://www.cs.columbia.edu/~smaskey/CS6998-0412/slides/week13_statnlp_web.pdf

http://videolectures.net/cikm08_elkan_llmacrf/

http://www.stokastik.in/understanding-conditional-random-fields/



A first key idea in CRFs will be to define a feature vector that maps an entire
input sequence x paired with an entire state sequence s to some d-dimensional feature vector.

IDEA: maps an entire input sequence x paired with an entire state sequence s to
some d-dimensional feature vector.

-->

# __Label Bias Problem__



### __CRF Important Observations__

* The big difference between MEMM and CRF is that MEMM is locally renormalized and suffers from the label bias problem, while CRFs are globally re-normalized.



## __Software Packages__

*




## __References__

* [Conditional Random Fields: An Introduction. Hanna M. Wallach, February 24, 2004. University of Pennsylvania CIS Technical Report MS-CIS-04-21](http://dirichlet.net/pdf/wallach04conditional.pdf)
