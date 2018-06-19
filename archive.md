---
layout: page
title: Blog Archive
permalink: /archive/
---

<!--
{% assign years = "2018,2017" | split: "," %}
{% for year in years %}
  <h3 style="border-bottom: 1px solid #e0e0e0">{{ year }}</h3>
  <ul style="margin-left: 0px; padding-left: 0px; list-style: none">
    {% for post in site.posts %}
      {% assign post_year = post.date | date: "%Y" %}
      {% if post_year == year %}
        <li style="padding-bottom:10px">
          <span class="post-meta">{{ post.date | date: "%d %b" }}</span>
          <a class="post-link" href="{{ post.url }}">{{ post.title }}</a>
        </li>
      {% endif %}
    {% endfor %}
  </ul>
{% endfor %}
-->

{% assign years = "2018,2017" | split: "," %}
{% for year in years %}
  <h3 style="border-bottom: 1px solid #e0e0e0">{{ year }}</h3>

  <ul style="margin-left: 0px; padding-left: 0px; list-style: none">

    {% for post in site.posts %}

      {% assign post_year = post.date | date: "%Y" %}
      {% if post_year == year %}
        <li style="padding-bottom:30px">
          <span class="post-meta">{{ post.date | date: "%d %b" }}</span>
          <br>
          <a class="post-link" href="{{ post.url }}">{{ post.title }}</a>
          <div class="preview-content">
            <div class="preview-excerpt">
              <div class="post-meta">
                <!--
                <div class="post-date">
                  <i class="fa fa-calendar-o"></i>   {{ post.date | date: '%Y-%m-%d' }}
                </div>
                <div class="post-tags">
                  <i class="fa fa-tags"></i>
                  {% for tag in post.tags %}
                    <span>{{tag}}</span>
                  {% endfor %}
                </div>
                -->
              </div>
              {{ post.excerpt }}
              <!--
              <a class="post-link" href="{{ post.url }}">&raquo;</a>
              -->
              <h3 style="border-bottom: 1px solid #e0e0e0"></h3>
            </div>
            <!--
            {% if post.preview_pic %}
              <div class="preview-pic">
                <a href="{{ post.url }}">
                  <img style="margin:auto" src="{{ post.preview_pic }}">
                </a>
              </div>
            {% endif %}
            -->
          </div>
        </li>
      {% endif %}
    {% endfor %}
  </ul>
{% endfor %}