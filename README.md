# Jekyll::blog

My personal homepage and blog at:

http://www.davidsbatista.net/

to run with jekyll own webserver
  1. `rvm use ruby-2.4.4`
  2. `jekyll serve`

using apache:
  1. `python run tag_generator.py`
  2. `export GEM_HOME=/home/dsbatista/gems; jekyll build -d /var/www/html/blog/`


Plugins and add-ons:

- paginate: https://github.com/sverrirs/jekyll-paginate-v2
- tags: http://longqian.me/2017/02/09/github-jekyll-tag/
