---
layout: post
title: Google's SyntaxNet - HTTP API for Portuguese
date: 2017-07-22 00:00:00
tags:  SyntaxNet pos-tags syntactic-dependencies
categories: [blog]
comments: true
disqus_identifier: 20170722
preview_pic: /assets/images/2017-07-22-SyntaxNetHTTP.png
description: How to set up a SyntaxNet HTTP endpoint for any language, and how to submit text to be tagged through Python, this post shows an example for Portuguese, but can easily be adapted to any other supported language.
---

In a [previous post](../../../../../blog/2017/03/25/syntaxnet/) I explained how load the syntactic and morphological information given by SyntaxNet into NLTK structures by parsing the standard output. Although useful this is does not scale when one wants to process thousands of sentences, but finally I've found a Docker image to setup SyntaxNet as a web-service.

It turns out this is simple and straightforward using a Docker image. Here are the steps on how to do it, and setting up for Portuguese:

#### 1. __Install Docker__

The first thing is to install Docker, the easiest way is to follow a tutorial. I've found a simple and straight forward [tutorial for Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04).


#### 2. __Install SyntaxNet-API Docker Image__

The next step is to install a [Docker image](https://github.com/askplatypus/syntaxnet-api) which already contains Tensorflow together with SyntaxNet and providing an API interface exposing the model via a web interface.

~~~~
git clone https://github.com/askplatypus/syntaxnet-api
cd syntaxnet-api
docker build . -t syntaxnet-api
~~~~

Before building the docker image (i.e., running `docker build . -t syntaxnet-api`) you can specify which language you want SyntaxNet to parse. This is done by updating the following line in `flask_server.py`, in this case for Portuguese:

~~~~
# Overrides available languages map
-language_code_to_model_name['en'] = 'English'
+language_code_to_model_name['pt'] = 'Portuguese'
~~~~

#### 3. __Run the Docker Image__

Next, after building the docker image, you just need to run it. Type

	docker images

to see a list of current images, you want to see the `IMAGE ID` for the syntaxnet-api, passing it to the following command:

	docker run -i -p 7000:7000 -t IMAGE_ID_value

This should expose a web-service on your localhost on port 7000 similar to [http://syntaxnet.askplatyp.us/v1#/default](http://syntaxnet.askplatyp.us/v1#/default)

The `-p 7000:7000` forwards the port 7000 on your host to the same port on the running image, the parameter `-i` forces the output to be shown on stdout, you may replace this by `-d` to make the image run in the background and detach from the shell.

You can also run in from the command line, with `curl`:

~~~
curl -X POST --header 'Content-Type: text/plain; charset=utf-8' --header 'Accept: text/plain' --header 'Content-Language: pt' -d 'Olá mundo, teste!' 'http://0.0.0.0:7000/v1/parsey-universal-full'
~~~

This should output something like this:

    1	Olá	_	INTJ	in	Gender=Masc|Number=Sing	0	ROOT
    2	mundo,	_	NOUN	n|M|S	Gender=Masc|Number=Sing	1	nsubj
    3	teste!	_	ADJ	adj|M|S	Gender=Masc|Number=Sing	2	amod

NOTE: I omitted some of the ouputted info for each word to make everything fit in one line :)


## Alternatives (bit faster)

After running a few experiments in batch I notice this was still a bit slow, probably because I was also running it on a machine without any GPUs.

Looking through more SyntaxNet Docker images I've [found another](https://github.com/danielperezr88/syntaxnet-api), a fork of the one described above, which pre-loads the models, and makes the batch processing a bit faster.

The only problem I've found was that it was loading the models for all the languages, and this would take around 10GB of RAM! So I created a new image by removing all the other models except the one for Portuguese, and build it using the commands described above.

I did an experiment, comparing both images, by measuring the time taken to process 500 sentences in Portuguese.

<center>
<table>
  <thead>
    <tr>
      <th style="text-align: left">Docker Image</th>
      <th style="text-align: center">Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left"><a href="https://github.com/askplatypus/syntaxnet-api">Original</a></td>
      <td style="text-align: center">35m12.131s</td>
    </tr>
    <tr>
      <td style="text-align: left"><a href="https://github.com/davidsbatista/syntaxnet-api">Updated</a></td>
      <td style="text-align: center">25m56.689s</td>
    </tr>
  </tbody>
</table>
</center>


Using the image that first pre-loads the Portuguese model is about 10min faster. Note, this was done in a machine without any GPUS, I believe using GPUs this would be much faster.

The image, loading models only for Portuguese, it's here:

[https://github.com/davidsbatista/syntaxnet-api](https://github.com/davidsbatista/syntaxnet-api)


__NOTE__: After experimenting, building and running Docker images you might want to clean up, and free up some space on your machine, these links might help:

* [https://lebkowski.name/docker-volumes/](https://lebkowski.name/docker-volumes/)
* [https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes](https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes)

## __Related posts__

 * __[Loading SyntaxNet's syntactic dependencies into NLTK's DependencyGraph](../../../../../blog/2017/03/25/syntaxnet/)__
