# Systemic Risk Stress-Test Case Study

Synthetic dataset: 10000 two-guardrail demands with exact empirical marginals `P(F_1)=P(F_2)=0.20` and co-failure `P(F_1 and F_2)=0.08`.

Finite stress rows solve the fixed-marginal Wasserstein stress problem. The FH row is the budget-to-infinity limit and is included only as a comparison endpoint.

| Method | Budget | Risk | Protection | Increase | Gap to FH |
| --- | --- | --- | --- | --- | --- |
| Empirical copula baseline | 0.00 | 0.0800 | 0.9200 | 0.0000 | 0.1200 |
| CCF beta-factor point estimate | - | 0.0800 | 0.9200 | 0.0000 | 0.1200 |
| Budget-constrained stress eps=0.02 | 0.02 | 0.1000 | 0.9000 | 0.0200 | 0.1000 |
| Budget-constrained stress eps=0.06 | 0.06 | 0.1400 | 0.8600 | 0.0600 | 0.0600 |
| Budget-constrained stress eps=0.10 | 0.10 | 0.1800 | 0.8200 | 0.1000 | 0.0200 |
| Unconstrained FH upper limit | infinity | 0.2000 | 0.8000 | 0.1200 | 0.0000 |
