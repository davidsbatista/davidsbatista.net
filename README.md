### Jekyll::blog

My personal homepage and blog at:

http://www.davidsbatista.net/

to run with jekyll own webserver
  1. `source /Users/dsbatista/.rvm/scripts/rvm`
  2. `rvm list`
  2. `rvm use ruby-x.x.x`
  3. `jekyll serve`

using apache:
  1. `python tag_generator.py` (create tag/ if doesn't exist)
  2. `jekyll build -d jekyll build -d /var/www/davidsbatista.net`


### Tips on how to set up Jekyll on macOS (it's a pain!)

- OPENSSL issues: https://stackoverflow.com/questions/74196882/cannot-install-jekyll-eventmachine-on-m1-mac
- Setup new Jekyll project: https://jekyllrb.com/docs/step-by-step/01-setup/


### Plugins and add-ons:

- Jemoji: https://davemateer.com/2019/05/27/Jemoji
- sitemap: https://github.com/jekyll/jekyll-sitemap
- paginate: https://github.com/sverrirs/jekyll-paginate-v2
- tags: http://longqian.me/2017/02/09/github-jekyll-tag/
