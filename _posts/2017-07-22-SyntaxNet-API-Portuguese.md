---
layout: post
title: Google's SyntaxNet Web API for Portuguese 
date: 2017-07-22 00:00:00
tags: [python, NLTK, SyntaxNet, API, Portuguese, part-of-speech, syntactic dependencies]
categories: [blog]
comments: true
disqus_identifier: 20170722
preview_pic:
---

In a [previous post]() I explained how load the syntactic and morphological information given by SyntaxNet into NLTK structures, such as Dependency Graph, by parsing the std output. Although usefull this is does not scale when one wants to process thousands of sentences, but finally I've found a Docker image to setup SyntaxNet as a webservice.

It turns out this is simple and straightforward using a Docker image. Here are the steps on how to do it, and setting up for Portuguese:

#### 1. __Install Docker__

The first thing is to install Docker, the easiest way is to follow a tutorial. I've found a simple and straigthforward [tutorial for Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04).


#### 2. __Install SyntaxNet-API Docker Image__

The next step is to install a [Docker image](https://github.com/askplatypus/syntaxnet-api) which already contains Tensorflow together with SyntaxNet and providing an API interface exposing the model via a web interface.

~~~~
git clone https://github.com/askplatypus/syntaxnet-api
cd syntaxnet-api
docker build . -t syntaxnet-api
~~~~

Before building the docker image (i.e., running `docker build . -t syntaxnet-api`) you can specify which language you want SyntaxNet to parse. This is done by updateding the following line in `flask_server.py`, in this case for Portuguese:

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

The `-p 7000:7000` forwards the port 7000 on your host to the same port on the running image, the parameter `-i` forces the ouput to be shown on stdout, you may replace this by `-d` to make the image run in the background and detach from the shell.

This should expose a webservice similar to this one: [http://syntaxnet.askplatyp.us/v1#/default](http://syntaxnet.askplatyp.us/v1#/default)

You can also run in from the command line, with `curl`:

~~~
curl -X POST --header 'Content-Type: text/plain; charset=utf-8' --header 'Accept: text/plain' --header 'Content-Language: pt' -d 'Olá mundo, teste!' 'http://0.0.0.0:7000/v1/parsey-universal-full'
~~~

This should output something like this:

    1	Olá	_	INTJ	in	Gender=Masc|Number=Sing	0	ROOT
    2	mundo,	_	NOUN	n|M|S	Gender=Masc|Number=Sing	1	nsubj
    3	teste!	_	ADJ	adj|M|S	Gender=Masc|Number=Sing	2	amod


<!--
with all languages
https://hub.docker.com/r/danielperezr88/syntaxnet-api/
-->
