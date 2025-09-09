---
layout: archive
title: "Estimation Theory"
permalink: /robotics/estimation_theory/
---

{% assign all_items = site.robotics | where_exp: "item", "item.path contains 'estimation_theory/'" | where: "tags", "EstimationTheory" | sort: 'date' | reverse %}

<ul>
  {% for post in all_items %}
    <li><a href="{{ post.url }}">{{ post.title }}</a>{% if post.date %} <span>{{ post.date | date: site.date_format }}</span>{% endif %}</li>
  {% endfor %}
</ul>
