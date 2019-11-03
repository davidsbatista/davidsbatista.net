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