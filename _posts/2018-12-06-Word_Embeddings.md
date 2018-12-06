---
layout: post
title: Word Embeddings - word2vec, fasttext, GloVe
date: 2018-12-06 00:00:00
tags: word-embeddings word2vec fasttext glove
categories: [blog]
#comments: true
# disqus_identifier: 20181206
preview_pic: /assets/images/2018-12-06-word-embeddings.jpg
---

Word embeddings are the technique that made possible all the latest achievements in NLP using neural networks. In this blog post I will review three popular techniques to generate word embeddings: skip-gram, glove and fasttext.

## Embeddings

"Deep contextualized word representations" (2018)
paper: http://aclweb.org/anthology/N18-1202
code:  https://github.com/Hironsan/anago


"Enriching Word Vectors with Subword Information"
paper: http://aclweb.org/anthology/Q17-1010
code:  https://github.com/facebookresearch/fastText


<!--

https://www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/

http://ruder.io/word-embeddings-2017/

https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html

https://jalammar.github.io/illustrated-bert/

-->

# __Word2Vec: Skip-Gram__

* A neural network with a single hidden layer which won't be used for the for the task we trained it on;

* Goal is to find good embeddings, i.e., learn the weights of the hidden layer;

* The weights of the hidden layer are the word vector, aka embeddings;

* The neural network is trained to given a specific word in the middle of a sentence (i.e., the input word) pick one random word out of the words in a defined context window of the input word, and compute the probability of every word in the vocabulary being the random picked word.

* The output probabilities are going to relate to how likely it is find each vocabulary word nearby our input word. For example, if you gave the trained network the input word “Soviet”, the output probabilities are going to be much higher for words like “Union” and “Russia” than for unrelated words like “watermelon” and “kangaroo”.

* We’ll train the neural network to do this by feeding it word pairs found in our training documents. The below example shows some of the training samples (word pairs) we would take from the sentence “The quick brown fox jumps over the lazy dog.” I’ve used a small window size of 2 just for the example. The word highlighted in blue is the input word.

* The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“Soviet”, “Union”) than it is of (“Soviet”, “Sasquatch”). When the training is finished, if you give it the word “Soviet” as input, then it will output a much higher probability for “Union” or “Russia” than it will for “Sasquatch”.


<!--
stochastic gradient descent
backward propagation

picture by:

https://pixabay.com/en/users/terimakasih0-624267/

-->

# __GloVe__


# __fasttext__


# __Experiments__

<!--

I wanted to explore embeddings for Portuguese with news articles that I've been crawling since the days I started my PhD, by luck the [small script](https://github.com/davidsbatista/publico.pt-news-scrapper) I wrote a few years ago, still works and it's running, triggered by a crontab, on some remote server fetching daily portuguese news articles  :)

Crawling text from the web is always tricky and it involves lots of cleaning, and plus I wanted a clean dataset to learn embeddings, I explicitly removed punctuation and normalized all words to lowercase. In order to do this I used a mix of sed and python, as shown below:

Using python replace all HTML entities by it's corresponding mappings into plain text:
{% highlight bash %}
python3 -c 'import html, sys; [print(html.unescape(l), end="") for l in sys.stdin]'
{% endhighlight bash %}

Next remove all HTML tags
{% highlight bash %}
sed s/"<[^>]*>"/""/g
{% endhighlight bash %}

I also used python to convert everything to lowercase, since `tr` command could not properly hand some characters
{% highlight bash %}
python3 -c 'import sys; [print(l.lower(), end="") for l in sys.stdin]' \
{% endhighlight bash %}

Remove ticks, parenthesis, quotation marks, parenthesis, etc.
{% highlight bash %}
sed s/"['\"\(\)\`”′″‴«»„”“‘’]"/""/g
{% endhighlight bash %}

Remove punctuation "glued" to last character of a word/token
{% highlight bash %}
sed s/"\(\w\)[\.,:;\!?\"\/+]\s"/"\1 "/g
{% endhighlight bash %}

Replace two or more consecutive spaces by just one
{% highlight bash %}
tr -s " "
{% endhighlight bash %}


Putting it all together in a single script:

{% highlight bash %}
cat news_aricles | cut --complement -f1,2,3 $1 \
| tr '\t' '\n' \
| python3 -c 'import html, sys; [print(html.unescape(l), end="") for l in sys.stdin]' \
| sed s/"<[^>]*>"/""/g \
| python3 -c 'import sys; [print(l.lower(), end="") for l in sys.stdin]' \
| tr -s " " \
| sed s/"['\"\(\)\`”′″‴«»„”“‘’º]"/""/g \
| sed s/"\(\w\)[\.,:;\!?\"\/+]\s"/"\1 "/g \
| sed s/"\["/""/g \
| sed s/"\]"/""/g \
| sed -e 's/\.//g' \
| sed s/" - "/" "/g \
| sed s/" — "/" "/g \
| sed s/" — "/" "/g \
| sed s/" — "/" "/g \
| sed s/" – "/" "/g \
| sed s/" – "/" "/g \
| sed s/" , "/" "/g \
| sed s/" \/ "/" "/g \
| sed s/"-,"/""/g \
| sed s/"—,"/""/g \
| sed s/"–,"/""/g \
| sed s/"--"/""/g \
| sed s/"\(\w\)[\.,:;\!?\"\/+]\s"/"\1 "/g \
| tr -s " " > news_articles_clean.txt;
{% endhighlight bash %}

The script takes as input a text file with a news article per line, including title, date, category, etc.; it takes only the text fields (i.e., title, lead, news text) and generates a file where each line consists of a title, a lead of a news text. All the tokens are in lower case and there is no punctuation.

A quick way to inspect the generated tokens is to run the following line, which it will output all the tokens in the file ordered by frequency of occurrence:

{% highlight bash %}
cat news_articles_clean.txt | tr ' ' '\n' | sort | uniq -c | sort -gr > tokens_counts.txt
{% endhighlight bash %}

-->