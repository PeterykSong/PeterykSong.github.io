---
layout: archive
title: "SLAM"
permalink: /robotics/slam/
---

{% assign items = site.robotics | where_exp: "p", "p.tags contains 'SLAM'" %}
<ul>
  {% for post in items %}
    <li><a href="{{ post.url }}">{{ post.title }}</a> <span>{{ post.date | date: site.date_format }}</span></li>
  {% endfor %}
</ul>