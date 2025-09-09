---
layout: archive
title: "ICP"
permalink: /robotics/icp/
---

{% assign robotics_items = site.robotics | where: "tags", "ICP" | sort: 'date' | reverse %}
{% assign subfolder_items = site.pages | where: "path", "_robotics/icp/" | where: "tags", "ICP" | sort: 'date' | reverse %}
{% assign all_items = robotics_items | concat: subfolder_items %}

<ul>
  {% for post in all_items %}
    <li><a href="{{ post.url }}">{{ post.title }}</a>{% if post.date %} <span>{{ post.date | date: site.date_format }}</span>{% endif %}</li>
  {% endfor %}
</ul>