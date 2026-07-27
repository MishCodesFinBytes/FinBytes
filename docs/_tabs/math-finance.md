---
layout: page
title: Math / Finance
icon: fas fa-chart-line
order: 5
permalink: /math-finance/
---

<style>
.mf-lede { color:var(--text-muted-color, #666); font-size:.95rem; margin-bottom:2.5rem; max-width:620px; line-height:1.7; }
.mf-section { margin-bottom:2.5rem; }
.mf-hdr { font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--text-muted-color, #999); margin-bottom:.6rem; padding-bottom:.5rem; border-bottom:1px solid var(--border-color, #ebebeb); }
.mf-list { list-style:none; margin:0; padding:0; }
.mf-list li { padding:.5rem 0; border-bottom:1px solid var(--border-color, #f4f4f4); }
.mf-list li:last-child { border-bottom:none; }
.mf-list a { font-size:.93rem; color:var(--text-color, #333); text-decoration:none; line-height:1.4; }
.mf-list a:hover { color:var(--link-color, #2a7ae2); text-decoration:underline; }
.mf-desc { display:block; font-size:.82rem; color:var(--text-muted-color, #999); margin-top:.2rem; }
</style>

<p class="mf-lede">The mathematical and financial concepts behind the code. Each page walks through the theory with worked numerical examples you can follow with a calculator, then shows the Python implementation.</p>

<div class="mf-section">
  <div class="mf-hdr">Foundations</div>
  <ul class="mf-list">
    <li>
      <a href="{{ "/math-finance/dcf-time-value/" | relative_url }}">Discounted Cash Flow &amp; Time Value of Money</a>
      <span class="mf-desc">Present value, future value, discount factors, NPV, IRR, annuities, perpetuities &mdash; the foundation everything else builds on</span>
    </li>
  </ul>
</div>

<div class="mf-section">
  <div class="mf-hdr">Yield Curves &amp; Rates</div>
  <ul class="mf-list">
    <li>
      <a href="{{ "/math-finance/yield-curves-and-rates/" | relative_url }}">Par Yields, Spot Rates, Forward Rates, and Fitted Curves</a>
      <span class="mf-desc">Bootstrapping, no-arbitrage forwards, Nelson-Siegel/Svensson parametric fitting, cubic spline interpolation</span>
    </li>
    <li>
      <a href="{{ "/math-finance/yield-curve-inversions/" | relative_url }}">Yield Curve Inversions: What They Signal and Why They Matter</a>
      <span class="mf-desc">2s10s spread, historical recession track record, the un-inversion signal, term premium</span>
    </li>
    <li>
      <a href="{{ "/math-finance/benchmark-rates-libor-transition/" | relative_url }}">Interest Rate Benchmarks: From LIBOR to Risk-Free Rates</a>
      <span class="mf-desc">SONIA, EONIA / &euro;STR, SOFR &mdash; what they are, the LIBOR transition, fallback spreads, worked PnL example &middot; <a href="https://quantlabs.streamlit.app/Benchmark_Rates" style="color:var(--link-color,#2a7ae2)">live demo</a></span>
    </li>
  </ul>
</div>

<div class="mf-section">
  <div class="mf-hdr">Bond Pricing &amp; Risk</div>
  <ul class="mf-list">
    <li>
      <a href="{{ "/math-finance/bond-pricing/" | relative_url }}">Z-Spread, OAS, Binomial Trees, and Spread Duration</a>
      <span class="mf-desc">Discounted cash flows, Z-spread bisection, binomial rate trees, backward induction, callable bonds, OAS, CS01</span>
    </li>
    <li>
      <a href="{{ "/math-finance/duration-convexity/" | relative_url }}">Duration &amp; Convexity: How Bond Prices Move When Rates Change</a>
      <span class="mf-desc">Macaulay duration, modified duration, DV01, convexity correction, duration vs spread duration (CS01)</span>
    </li>
  </ul>
</div>

<div class="mf-section">
  <div class="mf-hdr">Credit Risk</div>
  <ul class="mf-list">
    <li>
      <a href="{{ "/math-finance/credit-risk/" | relative_url }}">Spreads, CDS, Default Probabilities, and Expected Loss</a>
      <span class="mf-desc">Credit spreads, CDS mechanics, hazard rates, survival curves, piecewise bootstrapping, PD &times; LGD &times; EAD</span>
    </li>
  </ul>
</div>

<div class="mf-section">
  <div class="mf-hdr">Portfolio Risk</div>
  <ul class="mf-list">
    <li>
      <a href="{{ "/math-finance/portfolio-risk/" | relative_url }}">VaR, CVaR, Sharpe, and Monte Carlo Simulation</a>
      <span class="mf-desc">Returns, volatility, historical and parametric VaR, expected shortfall, Sharpe ratio, Cholesky decomposition, correlated Monte Carlo</span>
    </li>
  </ul>
</div>

<div class="mf-section">
  <div class="mf-hdr">Options &amp; Derivatives</div>
  <ul class="mf-list">
    <li>
      <a href="{{ "/math-finance/greeks/" | relative_url }}">The Greeks: Option Price Sensitivities</a>
      <span class="mf-desc">Delta, Gamma, Theta, Vega, Rho &mdash; what each measures, worked Black-Scholes examples, Greek interactions</span>
    </li>
  </ul>
</div>

<div class="mf-section">
  <div class="mf-hdr">P&amp;L &amp; Attribution</div>
  <ul class="mf-list">
    <li>
      <a href="{{ "/math-finance/pnl-explained/" | relative_url }}">P&amp;L Explained: Clean, Dirty, Attributed, and Unexplained</a>
      <span class="mf-desc">Realized vs unrealized, clean vs dirty, new-trade P&amp;L, Greeks-based attribution, predicted vs actual &amp; unexplained P&amp;L, FRTB PLA (HPL/RTPL/APL, Spearman &amp; KS), carry, economic vs accounting</span>
    </li>
  </ul>
</div>
