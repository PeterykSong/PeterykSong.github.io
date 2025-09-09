---
layout: archive
title: "SLAM"
permalink: /robotics/slam/
---

{% assign robotics_items = site.robotics | where_exp: "p", "p.tags contains 'SLAM'" %}
{% assign subfolder_items = site.pages | where_exp: "p", "p.path contains '_robotics/slam/' and p.tags contains 'SLAM'" %}
{% assign all_items = robotics_items | concat: subfolder_items | sort: 'date' | reverse %}

<ul>
  {% for post in all_items %}
    <li><a href="{{ post.url }}">{{ post.title }}</a>{% if post.date %} <span>{{ post.date | date: site.date_format }}</span>{% endif %}</li>
  {% endfor %}
</ul>