---
layout: post
title: Word Embeddings - a review
date: 2018-12-06 00:00:00
tags: word-embeddings word2vec fasttext glove ELMo BERT language-models character-embeddings character-language_models
categories: [blog]
#comments: true
# disqus_identifier: 20181206
preview_pic: /assets/images/2018-12-06-word-embeddings.jpg
---

Since the work of [Mikolov et al., 2013](https://arxiv.org/pdf/1301.3781.pdf) was published and the software package _word2vec_ was made public available a new era in NLP started on which word embeddings, also referred to as word vectors, play a crucial role. Word embeddings can capture many different properties of a word and become the de-facto standard to replace feature engineering in NLP tasks.

Since that milestone many new embeddings methods were proposed some which go down to the character level, and others that take into consideration even language models. I will try in this blog post to review some of these methods, and point out their main characteristics.

## __Introduction__

<!--
https://www.aaai.org/Papers/JAIR/Vol37/JAIR-3705.pdf
-->


## __Word level__

### __Skip-Gram__

Introduced by (Mikolov et al., 2013) was the first popular embeddings method for NLP tasks. The paper itself is hard to understand, and many details are left over but essential the model is a neural network with a single hidden layer.

The embeddings are actually the weights of the hidden layer in the neural network.

<!--
* A neural network with a single hidden layer which won't be used for the for the task we trained it on;

* Goal is to find good embeddings, i.e., learn the weights of the hidden layer;

* The weights of the hidden layer are the word vector, aka embeddings;

* The neural network is trained to given a specific word in the middle of a sentence (i.e., the input word) pick one random word out of the words in a defined context window of the input word, and compute the probability of every word in the vocabulary being the random picked word.

* The output probabilities are going to relate to how likely it is find each vocabulary word nearby our input word. For example, if you gave the trained network the input word “Soviet”, the output probabilities are going to be much higher for words like “Union” and “Russia” than for unrelated words like “watermelon” and “kangaroo”.

* We’ll train the neural network to do this by feeding it word pairs found in our training documents. The below example shows some of the training samples (word pairs) we would take from the sentence “The quick brown fox jumps over the lazy dog.” I’ve used a small window size of 2 just for the example. The word highlighted in blue is the input word.

* The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“Soviet”, “Union”) than it is of (“Soviet”, “Sasquatch”). When the training is finished, if you give it the word “Soviet” as input, then it will output a much higher probability for “Union” or “Russia” than it will for “Sasquatch”.
-->

## __GloVe__

<!--
http://text2vec.org/glove.html#linguistic_regularities
http://mlexplained.com/2018/04/29/paper-dissected-glove-global-vectors-for-word-representation-explained/
-->

---

<br>

# __Subword-level embeddings__

<!--
Can handle OOV handling
-->

## __fasttext__

<!--

"Enriching Word Vectors with Subword Information"
paper: http://aclweb.org/anthology/Q17-1010
code:  https://github.com/facebookresearch/fastText

stochastic gradient descent
backward propagation
picture by:
https://pixabay.com/en/users/terimakasih0-624267/
-->

---

<br>

# __Language Model based embeddings__

### __Contextual String Embeddings for Sequence Labeling__

<!--
http://alanakbik.github.io/talks/ML_Meetup_2018.pdf
-->

### __ELMo__

<!--
"Deep contextualized word representations" (2018)
paper: http://aclweb.org/anthology/N18-1202
code:  https://github.com/Hironsan/anago
-->

### __BERT__

<!--
https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html
https://jalammar.github.io/illustrated-bert/
https://www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/
-->

<!--
Another choice for using pre-trained embeddings that integrate character information is to leverage a state-of-the-art language model (Jozefowicz et al., 2016) [14] trained on a large in-domain corpus, e.g. the 1 Billion Word Benchmark (a pre-trained Tensorflow model can be found here). While language modelling has been found to be useful for different tasks as auxiliary objective (Rei, 2017) [15], pre-trained language model embeddings have also been used to augment word embeddings (Peters et al., 2017) [16]. As we start to better understand how to pre-train and initialize our models, pre-trained language model embeddings are poised to become more effective. They might even supersede word2vec as the go-to choice for initializing word embeddings by virtue of having become more expressive and easier to train due to better frameworks and more computational resources over the last years.
-->

---

<br>



## __Summary__

<!--
- Multi-sense embeddings: embeddings fail to capture polisemy
not being able to capture multiple senses of words, word embeddings also fail to capture the meanings of phrases and multi-word expressions, which can be a function of the meaning of their constituent words, or have an entirely new meaning.

refer evaluation







<!--

## __Experiments__

Probably better to be another post?

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

## __References__

- ["Efficient Estimation of Word Representations in Vector Space" Mikolov et al. (2013)](https://arxiv.org/pdf/1301.3781.pdf)
- ["GloVe: Global Vectors for Word Representation" Pennington et al. (2014) ](https://www.aclweb.org/anthology/D14-1162)
- [Word embeddings in 2017: Trends and future directions](http://ruder.io/word-embeddings-2017/)
- [McCormick, C. (2016, April 19). Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)