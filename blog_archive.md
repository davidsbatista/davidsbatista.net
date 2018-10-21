---
layout: default
title: Blog
permalink: /archive/
---

<div class="previews">  
  {% for post in site.posts %}
  <div class="preview">
    <h1 class="post-title">
      <a class="post-link" href="{{ post.url }}">{{ post.title }}</a>
    </h1>
    <div class="preview-content">
      <div class="preview-excerpt">
        <div class="post-meta">
          <div class="post-date">
            <i class="fa fa-calendar-o"></i>   {{ post.date | date: '%Y-%m-%d' }}
          </div>
          <div class="post-tags">
            <i class="fa fa-tags"></i>
            {% for tag in post.tags %}
              <!--<span>{{tag}}</span>-->
              <span><a href="/tag/{{ tag }}"><code class="highligher"><nobr>{{ tag }}</nobr></code></a></span>
            {% endfor %}
          </div>
        </div>
        {{ post.excerpt }}        
      </div>
      {% if post.preview_pic %}
        <div class="preview-pic">
          <a href="{{ post.url }}">
            <img style="margin:auto" src="{{ post.preview_pic }}">
          </a>
        </div>
      {% endif %}
    </div>
  </div>
  {% endfor %}
</div>

<!--
<div class="pagination">
  {% if paginator.next_page %}
    <a class="nav-link" href="/blog/p{{ paginator.next_page }}/">&#8592; previous</a>
  {% endif %}
  {% if paginator.previous_page %}
    {% if paginator.page == 2 %}
      <a class="nav-link" href="/">next &#8594;</a>
    {% else %}
      <a class="nav-link" href="/blog/p{{paginator.previous_page}}/">next &#8594;</a>
    {% endif %}
  {% endif %}
</div>
-->