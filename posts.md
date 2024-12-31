---
layout: archive
title: "All Posts"
permalink: /posts/

---

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) ({{ post.date | date: "%Y-%m-%d" }})
{% endfor %}

{% for page in site.pages %}
{% if page.title and page.layout != 'home' and page.layout != 'page' %}
- [{{ page.title }}]({{ page.url }})
{% endif %}
{% endfor %}
