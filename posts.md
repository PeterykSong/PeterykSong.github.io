---
layout: archive
title: "All Posts"
permalink: /posts/

---
{% if page.author_profile or layout.author_profile or page.sidebar %}
  <div class="sidebar sticky">
  {% if page.author_profile or layout.author_profile %}{% include author-profile.html %}{% endif %}
  {% if page.sidebar %}
    {% for s in page.sidebar %}
      {% if s.image %}
        <img src="{{ s.image | relative_url }}"
             alt="{% if s.image_alt %}{{ s.image_alt }}{% endif %}">
      {% endif %}
      {% if s.title %}<h3>{{ s.title }}</h3>{% endif %}
      {% if s.text %}{{ s.text | markdownify }}{% endif %}
      {% if s.nav %}{% include nav_list nav=s.nav %}{% endif %}
    {% endfor %}
    {% if page.sidebar.nav %}
      {% include nav_list nav=page.sidebar.nav %}
    {% endif %}
  {% endif %}
  </div>
{% endif %}


{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) ({{ post.date | date: "%Y-%m-%d" }})
{% endfor %}

{% for page in site.pages %}
{% if page.title and page.layout != 'home' and page.layout != 'page' %}
- [{{ page.title }}]({{ page.url }})
{% endif %}
{% endfor %}
