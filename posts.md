---
layout: single
title: "All Posts"
permalink: /posts/
---

# Author Information

- **Name:** {{ site.author.name }}
- **Bio:** {{ site.author.bio }}
- **Location:** {{ site.author.location }}
- **Email:** [{{ site.author.email }}](mailto:{{ site.author.email }})

---

## Blog Posts

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) ({{ post.date | date: "%Y-%m-%d" }})
{% endfor %}

<!-- {% for page in site.pages %} -->
<!-- {% if page.title and page.layout != 'home' and page.layout != 'page' %} -->
<!-- - [{{ page.title }}]({{ page.url }}) -->
<!-- {% endif %} -->
<!-- {% endfor %} -->
 
