---
layout: page
title: QuantLab
permalink: /
---

<p>Quantitative strategies and pricing models implemented in Python, C++ and C#.</p>
<ul class="ql-project-list">
  {% assign projects = site.quant_lab | sort: 'date' | reverse %}
  {% for project in projects %}
    <li>
      <a href="{{ project.url | relative_url }}">{{ project.title }}</a>
      {% if project.description %}<span class="ql-desc"> — {{ project.description }}</span>{% endif %}
      {% if project.tags.size > 0 %}<span class="ql-tags">{{ project.tags | join: " · " }}</span>{% endif %}
    </li>
  {% endfor %}
</ul>

<style>
.ql-project-list {
  list-style: none;
  padding: 0;
  margin-top: 24px;
}
.ql-project-list li {
  padding: 12px 0;
  border-bottom: 1px solid #e8e8e8;
}
.ql-project-list a {
  font-weight: 600;
  font-size: 1.05rem;
}
.ql-desc {
  color: #555;
  font-size: 0.95rem;
}
.ql-tags {
  display: block;
  font-size: 0.8rem;
  color: #888;
  margin-top: 4px;
}
</style>
