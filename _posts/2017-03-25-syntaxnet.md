---
layout: post
title: Google's SyntaxNet in Python NLTK 
date: 2017-03-25 00:00:00
tags: [parsing, SyntaxNet, Python, NLTK, Dependency Graph]
categories: [blog]
comments: true
preview_pic: /assets/images/2017-03-25-syntaxnet.png
---

In May 2016 Google released [SyntaxNet](https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html), a syntactic parser whose performance beat previous proposed approaches. 

In this post I will show you how to have SyntaxNet's syntactic dependencies and other morphological information in Python, precisely how to load [NLTK](http://www.nltk.org/) structures such as [DependencyGraph](http://www.nltk.org/_modules/nltk/parse/dependencygraph.html) and [Tree](http://www.nltk.org/_modules/nltk/tree.html) with SyntaxNet's output.

 In this example will use the Portuguese model, but as you will see this can be easily adapted to any language, provided you have already a pretrained model.


## Setup

First you need to install SyntaxNet:

    https://github.com/tensorflow/models/tree/master/syntaxnet


Then, you need to download a pretrained model, from the [list of all the available models](https://github.com/tensorflow/models/blob/master/syntaxnet/g3doc/universal.md)

    http://download.tensorflow.org/models/parsey_universal/<language>.zip


As the authors show in the tutorial after installing SyntaxNet and downloading a pretrained model, one can parse a sentence with the following command:

    MODEL_DIRECTORY=/where/you/unzipped/the/model/files
    cat sentences.txt | syntaxnet/models/parsey_universal/parse.sh \
    $MODEL_DIRECTORY > output.conll


Now I will show you how to parse a file with a sentence per line and use it within Python NLTK.

    cat sentences.txt

	Quase 900 funcionários do Departamento de Estado assinaram memorando \
	que critica Trump.
	Meo, Nos e Vodafone arriscam-se a ter de baixar preços a milhões \
	de clientes.


First we load all the sentences into a list, and joined them into a single string separated by the newline '\n' character.


{% highlight python %}

import subprocess
import os
import sys

from nltk import DependencyGraph

with open(sys.argv[1], 'r') as f:
    data = f.readlines()
    sentences = [x.strip() for x in data]

{% endhighlight %}



Then we will use python [subprocess](https://docs.python.org/2/library/subprocess.html) to call SyntaxNet, process the loaded sentences, and fetch the parsed sentences from stdout.

{% highlight python %}

all_sentences = "\n".join(sentences)

# redirect std_error to /dev/null
FNULL = open(os.devnull, 'w')

process = subprocess.Popen(
    'MODEL_DIRECTORY=/Users/dbatista/Downloads/Portuguese; '
    'cd /Users/dbatista/models/syntaxnet; '
    'echo \'%s\' | syntaxnet/models/parsey_universal/parse.sh '
    '$MODEL_DIRECTORY 2' % all_sentences,
    shell=True,
    universal_newlines=False,
    stdout=subprocess.PIPE,
    stderr=FNULL)

output = process.communicate()

{% endhighlight %}

We process the captured stdout, for each token, the dependencies and other morphological information. Each token is represented by list with all it's syntactic and morphologic information. A list of lists makes the sentence.

{% highlight python %}

processed_sentences = []
sentence = []
for line in output[0].split("\n"):
    if len(line) == 0:
        processed_sentences.append(sentence)
        sentence = []
    else:
        word = line.split("\t")
        sentence.append(word)

{% endhighlight %}

We then join each word/token information in a string separated by '\tab' character, each word/token in a different line. 

{% highlight python %}

deps = []
for sentence in processed_sentences:
    s = ''
    for line in sentence:
        s += "\t".join(line) + '\n'
    deps.append(s)

{% endhighlight %}

We then pass this string into the NLTK's DependenccyGraph and can then see all the dependency triples or an ASCII print of the tree.

{% highlight python %}

for sent_dep in deps:
    graph = DependencyGraph(tree_str=sent_dep.decode("utf8"))
    print "triples"
    for triple in graph.triples():
        print triple
    print
    tree = graph.tree()
    print tree.pretty_print()

{% endhighlight %}


For the first sentence we have the following triples and tree:

    ((u'assinaram', u'VERB'), u'nsubj', (u'funcion\xe1rios', u'NOUN'))
    ((u'funcion\xe1rios', u'NOUN'), u'nummod', (u'900', u'NUM'))
    ((u'900', u'NUM'), u'advmod', (u'Quase', u'ADV'))
    ((u'funcion\xe1rios', u'NOUN'), u'name', (u'Departamento', u'PROPN'))
    ((u'Departamento', u'PROPN'), u'case', (u'do', u'ADP'))
    ((u'funcion\xe1rios', u'NOUN'), u'name', (u'Estado', u'PROPN'))
    ((u'Estado', u'PROPN'), u'case', (u'de', u'ADP'))
    ((u'assinaram', u'VERB'), u'ccomp', (u'memorando', u'VERB'))
    ((u'memorando', u'VERB'), u'ccomp', (u'critica', u'VERB'))
    ((u'critica', u'VERB'), u'mark', (u'que', u'SCONJ'))
    ((u'critica', u'VERB'), u'dobj', (u'Trump.', u'PROPN'))
    
                       assinaram
                ___________|_____________
          funcionários               memorando
       ________|___________              |
     900  Departamento   Estado       critica
      |        |           |       ______|_______
    Quase      do          de    que           Trump.


And for the second sentence:


    ((u'pode', u'VERB'), u'nsubj', (u'galinha', u'NOUN'))
    ((u'galinha', u'NOUN'), u'det', (u'Uma', u'DET'))
    ((u'pode', u'VERB'), u'dobj', (u'ovos', u'NOUN'))
    ((u'ovos', u'NOUN'), u'case', (u'por', u'ADP'))
    ((u'ovos', u'NOUN'), u'nummod', (u'250', u'NUM'))
    ((u'ovos', u'NOUN'), u'nmod', (u'ano.', u'NOUN'))
    ((u'ano.', u'NOUN'), u'case', (u'por', u'ADP'))
    
                pode
        _________|____
       |             ovos
       |      ________|____
    galinha  |        |   ano.
       |     |        |    |
      Uma   por      250  por


I'm still trying to figure it out how to have SyntaxNet running as a daemon or service, where we can give a sentence and have as a result, for instance, a JSON object with the syntactic and morphologic information.

-----