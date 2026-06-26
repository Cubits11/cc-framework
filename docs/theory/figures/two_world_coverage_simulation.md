# Two-World Coverage Simulation Results

Seeded validation grid generated with
`cc.kernel.causal.run_coverage_simulation(icc_values=[0.0, 0.25], cluster_sizes=[4, 10], n_clusters=36, true_effect=0.2, monte_carlo_reps=40, bootstrap_reps=120, seed=11)`.

| n_clusters | cluster_size | icc | true_effect | nominal_coverage | monte_carlo_reps | bootstrap_reps | cluster_bootstrap_coverage | naive_coverage | coverage_tolerance | within_tolerance | mean_se_understatement_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 36 | 4 | 0.0 | 0.2 | 0.95 | 40 | 120 | 0.925 | 0.925 | 0.10338036564067671 | True | 0.966755867806911 |
| 36 | 4 | 0.25 | 0.2 | 0.95 | 40 | 120 | 0.925 | 0.875 | 0.10338036564067671 | True | 1.2590149078942303 |
| 36 | 10 | 0.0 | 0.2 | 0.95 | 40 | 120 | 0.95 | 0.95 | 0.10338036564067671 | True | 0.9784433960978868 |
| 36 | 10 | 0.25 | 0.2 | 0.95 | 40 | 120 | 0.925 | 0.8 | 0.10338036564067671 | True | 1.7244729592459485 |

At positive ICC, the naive independent-observation interval under-covers and
its standard error is smaller than the cluster-bootstrap standard error. The
cluster-bootstrap intervals remain within the pre-specified Monte Carlo
tolerance for this validation grid.
