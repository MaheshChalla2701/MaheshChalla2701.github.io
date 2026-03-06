---
layout: default
title: Home
---

# Welcome to my blog!

Check out my latest posts:

<ul>
  {% for post in site.posts %}
    <li>
      <span>{{ post.date | date: "%B %e, %Y" }}</span> &raquo; 
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
