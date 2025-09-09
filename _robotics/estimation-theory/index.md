---
layout: archive
title: "Estimation Theory"
permalink: /robotics/estimation-theory/
---

{% assign robotics_items = site.robotics | where_exp: "p", "p.tags contains 'EstimationTheory'" %}
{% assign subfolder_items = site.pages | where_exp: "p", "p.path contains '_robotics/estimation-theory/' and p.tags contains 'EstimationTheory'" %}
{% assign all_items = robotics_items | concat: subfolder_items | sort: 'date' | reverse %}

<ul>
  {% for post in all_items %}
    <li><a href="{{ post.url }}">{{ post.title }}</a> <span>{{ post.date | date: site.date_format }}</span></li>
  {% endfor %}
</ul>