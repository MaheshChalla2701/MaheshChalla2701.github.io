---
layout: default
title: Mahesh Challa's Blog
---

# Latest Insights

Check out my latest posts:

<div class="posts-list">
  {% for post in site.posts %}
    <article class="post-preview">
    <br>
      <h2>
        <a href="{{ post.url }}">{{ post.title }}</a>
      </h2>
      <div class="post-excerpt">
        {{ post.description }}
      </div>
    </article>
  {% endfor %}
</div>
