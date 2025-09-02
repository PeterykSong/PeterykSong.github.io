---
layout: archive
title: "ICP"
permalink: /robotics/icp/
---

{% assign items = site.robotics | where_exp: "p", "p.tags contains 'ICP'" %}
<ul>
  {% for post in items %}
    <li><a href="{{ post.url }}">{{ post.title }}</a> <span>{{ post.date | date: site.date_format }}</span></li>
  {% endfor %}
</ul>