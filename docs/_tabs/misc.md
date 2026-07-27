---
layout: page
title: Tech Stack
icon: fas fa-toolbox
order: 6
permalink: /tech-stack/
---

<style>
.misc-lede { color:var(--text-muted-color, #666); font-size:.95rem; margin-bottom:2.5rem; max-width:580px; line-height:1.7; }
.misc-section { margin-bottom:2.5rem; }
.misc-hdr { font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--text-muted-color, #999); margin-bottom:.6rem; padding-bottom:.5rem; border-bottom:1px solid var(--border-color, #ebebeb); display:flex; align-items:baseline; gap:.75rem; }
.misc-hdr .sub { font-weight:400; font-size:.7rem; color:var(--text-muted-color, #bbb); letter-spacing:0; text-transform:none; }
.misc-list { list-style:none; margin:0; padding:0; }
.misc-list li { padding:.5rem 0; border-bottom:1px solid var(--border-color, #f4f4f4); }
.misc-list li:last-child { border-bottom:none; }
.misc-list a { font-size:.93rem; color:var(--text-color, #333); text-decoration:none; line-height:1.4; }
.misc-list a:hover { color:var(--link-color, #2a7ae2); text-decoration:underline; }
.misc-desc { display:block; font-size:.82rem; color:var(--text-muted-color, #999); margin-top:.2rem; }
</style>

<p class="misc-lede">Technology stack references &mdash; libraries, infrastructure, and tooling. Practical guides with code examples you can copy-paste.</p>

<div class="misc-section" id="projects">
  <div class="misc-hdr">Projects <span class="sub">architecture deep-dives</span></div>
  <ul class="misc-list">
    <li>
      <a href="{{ "/tech-stack/inqlab-architecture/" | relative_url }}">INQLabs &mdash; Dependency-Free Generative-Art Studio</a>
      <span class="misc-desc">Self-contained HTML canvas/WebGL generators, PNG <code>tEXt</code> provenance, Netlify Identity auth (nf_jwt bridge), and a serverless git-as-CMS backend</span>
    </li>
  </ul>
</div>

<div class="misc-section" id="tech-references">
  <div class="misc-hdr">Technology references <span class="sub">from the QuantLab project</span></div>
  <ul class="misc-list">
    <li>
      <a href="{{ "/tech-stack/apis-data-sources/" | relative_url }}">APIs &amp; Data Sources</a>
      <span class="misc-desc">yfinance, CoinGecko, Land Registry, Overpass, central bank APIs &mdash; everything FinBytes connects to</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/parquet-files/" | relative_url }}">Parquet Files in Python</a>
      <span class="misc-desc">Why CSV is not enough &mdash; columnar storage, compression, type preservation, real benchmarks</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/aws-services/" | relative_url }}">AWS Services</a>
      <span class="misc-desc">S3, Lambda, API Gateway, RDS, DynamoDB, SQS, SNS, ElastiCache, CloudWatch</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/terraform/" | relative_url }}">Terraform</a>
      <span class="misc-desc">Infrastructure as Code &mdash; init, plan, apply, state, modules</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/fastapi/" | relative_url }}">FastAPI</a>
      <span class="misc-desc">Async APIs, Pydantic models, uvicorn/gunicorn stack</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/postgresql-sqlalchemy/" | relative_url }}">PostgreSQL &amp; SQLAlchemy</a>
      <span class="misc-desc">Async ORM, Alembic migrations, Docker Compose, RDS</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/streamlit/" | relative_url }}">Streamlit</a>
      <span class="misc-desc">Python-only dashboards, secrets, multi-page apps</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/github-actions/" | relative_url }}">GitHub Actions &amp; CI/CD</a>
      <span class="misc-desc">Workflows, OIDC, monorepo patterns</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/graph-navigation/" | relative_url }}">Graph Navigation for Jekyll</a>
      <span class="misc-desc">Add an Obsidian-style D3.js expanding graph to any Jekyll blog &mdash; tutorial with gotchas</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/plotting-libraries/" | relative_url }}">Plotting Libraries Compared</a>
      <span class="misc-desc">Plotly, Matplotlib, Altair, Bokeh &mdash; same data, four views, outlier handling</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/london-house-prices/" | relative_url }}">London House Prices</a>
      <span class="misc-desc">Land Registry data, choropleth maps, postcode comparison, brand effect analysis</span>
    </li>
  </ul>
</div>

<div class="misc-section" id="redis">
  <div class="misc-hdr">Infrastructure <span class="sub">caching, queues, containers</span></div>
  <ul class="misc-list">
    <li>
      <a href="{{ "/tech-stack/redis-caching/" | relative_url }}">Redis &amp; Caching</a>
      <span class="misc-desc">Data types, cache-aside pattern, TTL, LRU, pub/sub, connection pooling</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/message-queues/" | relative_url }}">Message Queues &amp; RabbitMQ</a>
      <span class="misc-desc">Exchanges, routing, pika, DLQ, RabbitMQ vs Redis vs SQS</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/docker-cicd/" | relative_url }}">Docker &amp; CI/CD</a>
      <span class="misc-desc">Multi-stage builds, docker-compose, GitHub Actions, Kubernetes basics</span>
    </li>
  </ul>
</div>

<div class="misc-section" id="tools">
  <div class="misc-hdr">Tools <span class="sub">CLI, version control</span></div>
  <ul class="misc-list">
    <li>
      <a href="{{ "/tech-stack/argparse-cli/" | relative_url }}">argparse &amp; CLI Tools</a>
      <span class="misc-desc">Arguments, flags, subcommands, the type=bool trap, click/typer</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/git-workflow/" | relative_url }}">Git Workflow</a>
      <span class="misc-desc">Branching, PRs, protected master, merge conflicts, commit conventions</span>
    </li>
  </ul>
</div>

<div class="misc-section" id="quantlab-aws">
  <div class="misc-hdr">QuantLab &mdash; AWS &amp; Credit Risk <span class="sub">exercises 07, 09&ndash;20</span></div>
  <ul class="misc-list">
    <li>
      <a href="{{ "/tech-stack/docker-for-python-services/" | relative_url }}">Docker for Python Services</a>
      <span class="misc-desc">Exercise 07 &mdash; multi-stage builds, compose, .dockerignore</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/aws-fundamentals-treasury-data/" | relative_url }}">AWS Fundamentals &amp; Treasury Data</a>
      <span class="misc-desc">Exercise 09 &mdash; FRED API, IAM, S3 bucket, yield curve classification</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/s3-par-curve-ingestion/" | relative_url }}">S3 Par Curve Ingestion</a>
      <span class="misc-desc">Exercise 10 &mdash; boto3, moto mocks, versioned S3 storage</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/rds-postgresql-yield-schema/" | relative_url }}">RDS PostgreSQL &amp; Yield Schema</a>
      <span class="misc-desc">Exercise 11 &mdash; Alembic migrations, seed from S3, 6-table schema</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/lambda-spot-curve-bootstrapping/" | relative_url }}">Lambda &amp; Spot Curve Bootstrapping</a>
      <span class="misc-desc">Exercise 12 &mdash; Lambda handler, API Gateway REST, 14 tests</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/cicd-forward-rate-curve/" | relative_url }}">CI/CD &amp; Forward Rate Curve</a>
      <span class="misc-desc">Exercise 13 &mdash; GitHub Actions, OIDC, forward rates</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/terraform-nelson-siegel/" | relative_url }}">Terraform &amp; Nelson-Siegel</a>
      <span class="misc-desc">Exercise 14 &mdash; IaC, scipy on Lambda, deployment friction log</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/sqs-sns-credit-spreads/" | relative_url }}">SQS/SNS &amp; Credit Spreads</a>
      <span class="misc-desc">Exercise 15 &mdash; credit spreads, CDS, hazard rates, async messaging</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/websockets-realtime-spreads/" | relative_url }}">WebSockets &amp; Real-Time Spreads</a>
      <span class="misc-desc">Exercise 16 &mdash; API Gateway WS, DynamoDB connections, fan-out</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/elasticache-bond-caching/" | relative_url }}">ElastiCache &amp; Bond Caching</a>
      <span class="misc-desc">Exercise 17 &mdash; Redis cache-aside, TTL strategy, graceful degradation</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/terraform-advanced-default-probs/" | relative_url }}">Terraform Advanced &amp; Default Probabilities</a>
      <span class="misc-desc">Exercise 18 &mdash; piecewise hazard rates, survival curves, modules</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/cloudwatch-oas-zspreads/" | relative_url }}">CloudWatch &amp; OAS/Z-Spread</a>
      <span class="misc-desc">Exercise 19 &mdash; binomial trees, backward induction, callable bonds</span>
    </li>
    <li>
      <a href="{{ "/tech-stack/integration-testing-credit-var/" | relative_url }}">Integration Testing &amp; Credit VaR</a>
      <span class="misc-desc">Exercise 20 &mdash; Monte Carlo, Cholesky, spread duration, portfolio risk</span>
    </li>
  </ul>
</div>
