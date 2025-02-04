---
layout: post
title: Daily Github Tips 
date: 2025-02-04 00:00:00
tags: github reference-post
categories: [blog]
comments: true
disqus_identifier: 20250204
preview_pic: 
---


### __Create a local branch of a remote branch forked from a repository__


#### __Add a remote__

```
git remote add github git://github.com/jdoe/coolapp.git
git fetch github
```

#### __List all remote branches__

```
git branch -r
```

output
todo

#### __Create a new local branch (test) from a github's remote branch (pu):__

```
git branch base-url-validation raspawar/raspawar/base-url-validation
git checkout base-url-validation
```