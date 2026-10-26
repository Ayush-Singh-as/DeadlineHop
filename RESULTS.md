# Measured results

All numbers below come from the frozen 400-episode test suite. Six learned checkpoints and six baselines produce 4,800 evaluated episodes. Each learned variant uses three training seeds and 250,000 steps per seed. Model and baseline settings were selected using validation only.

## Primary structured scenarios

| Policy | Mean on-time delivery | Training-seed range |
|---|---:|---:|
| Fixed channel | 65.17% | One evaluation stream |
| History predictor | 94.56% | One evaluation stream |
| Random hopping | 78.71% | One evaluation stream |
| Cyclic hopping | 76.76% | One evaluation stream |
| Stay/switch | 87.72% | One evaluation stream |
| Sliding-window UCB | 77.55% | One evaluation stream |
| Double DQN H=1 | 94.06% | 93.63–94.50% |
| Double DQN H=16 | 90.80% | 89.61–92.23% |

Episode fractions are averaged within families and then equally across the four structured families, within each training seed. The range is not a confidence interval. Baselines do not have training-seed repeats.

## Breakdown and controls

| Scenario | H=16 | H=1 | History predictor |
|---|---:|---:|---:|
| sweep | 87.73% | 95.18% | 96.70% |
| burst | 97.11% | 98.29% | 98.51% |
| follower | 87.21% | 88.15% | 88.23% |
| switching | 91.16% | 94.61% | 94.78% |
| none | 99.94% | 99.98% | 100.00% |
| iid | 69.43% | 72.99% | 89.96% |
| stress_load | 77.26% | 84.58% | 84.72% |
| stress_deadline | 83.83% | 87.49% | 85.75% |
| stress_delay | 95.32% | 94.90% | 97.57% |
| stress_speed | 74.55% | 85.60% | 86.37% |

## Interpretation

The single-observation agent delivers 94.06% overall, 0.50 percentage points below the history predictor. H=16 delivers 90.80%, 3.75 points below the predictor and 3.25 points below H=1. Both agents improve substantially over their initial validation policies, but the proposed long-history advantage is not supported under this training budget.

H=1 retains the previous action/outcome, so this is not evidence that all memory is unnecessary. The H=16 input has more parameters, and convergence can differ under a matched step budget. No parameter-matched or longer-budget experiment was run, so attributing the difference to one cause would be speculative.

The predictor wins overall, while H=1 has a 1.74-point advantage under shorter deadlines. Neither this point difference nor the aggregate ranking is a formal significance claim. The modeled reactive adversary is scripted; no real-radio, adversarial co-training, or hardware-transfer claim is justified.

Under IID interference, history cannot predict the current mask. Policies can nevertheless differ in delivery because retuning wastes service opportunities. The analytic test concerns success per actual attempt, not packet delivery. The relatively poor IID delivery of the learned policies is retained as a failure mode, consistent with their structured-only training distribution.

## Reproducibility evidence

- 34 automated tests cover simulator contracts, interference causality, information boundaries, replay overwrite/episode handling, Double DQN targets, exact checkpoint continuation, and analytical controls.
- A second environment installed the built wheel and tested the package; five released-model episodes matched every field of the recorded evaluation.
- Every final episode is audited for packet conservation and reward-component reconciliation. Checkpoint hashes and validation-only selection are verified.
- The report is two columns with two main-content pages and one references-only page. All figures are generated from saved results.
- Training used CPU PyTorch 2.14.1 on an exposed Intel Core i7-13650HX; each run took 162–309 seconds including validation. CUDA execution remains available in the CLI but was not tested here.

## Files

[Per-episode data](artifacts/results), [seed summaries](artifacts/figures/seed_summary.json), [learning curves](artifacts/figures/learning.png), [family comparisons](artifacts/figures/delivery.png), [control results](artifacts/figures/controls.png), [stress results](artifacts/figures/stress.png), [demo trace](artifacts/demo/trace.json), [verification record](artifacts/verification.json).
