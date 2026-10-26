# Frozen experiment protocol

The final test manifest is generated before training and checked by SHA-256 on load. Validation IDs begin at 100000; test IDs at 200000. Training episodes use independent seed derivation and continuous parameter sampling. The baseline fitting namespace uses 80001/800000. These namespaces prevent accidental reuse, not guarantee statistical independence by themselves.

## Core experiment

- H=16 and H=1 Double DQN, each with seeds 11, 22, 33 and 250,000 environment steps.
- MLP: two 128-unit ReLU layers, nine outputs. Adam 0.0003, gamma 0.99, batch 64, replay 50,000, warm-up 5,000, update every four steps, target copy every 1,000 updates, Huber loss, gradient norm cap 10.
- Epsilon decreases from 1 to 0.05 across the first 100,000 steps (initial 5,000 are fully random). Greedy evaluation with first-index tie breaking.
- Validation at initialization, every 25,000 steps, and final step. Select each seed's checkpoint by family-macro-average on-time delivery; energy breaks exact ties, then earlier checkpoint wins.
- Validation: 20 episodes × four structured families = 80. Final suite: 50 × four structured families, 50 × two controls, 25 × four stress conditions = 400 episodes per policy/checkpoint.
- Stress: load 0.80 and deadlines {2,4,6} in nominal switching; follower delay 4; sweep movement probability 0.9. Keep eight channels, deadline normalization 12, horizon 512.

Final weights are frozen. Histories adapt from feedback; this is not online gradient learning. Baseline statistics update from their own test observations and reset to their fixed prior at every episode. Report this algorithmic difference explicitly.

## Baseline selection

Fixed channel: validate all eight choices. Random/cyclic/stay-switch: commit through retunes to the next actual attempt. On empty queues they wait. Window UCB: validate windows {50,200,500} and bonus {0.25,0.5,1}. UCB uses recent observed attempts only and scores optimistic success per service slot minus energy/switch costs. Predictor: fit context counts on 60 training episodes under random/cyclic/stay-switch behavior; validate lengths one and two. Counts use smoothed per-channel backoff. The predictor's context encodes channel/outcome and capped elapsed slots. Both adaptive policies consider packets that survive a candidate retune. The predictor's two-slot score is an approximation, not an optimal model-based planner.

Retain all candidate validation scores in `artifacts/baseline_settings.json`. Final test results must not influence any setting. Report all six baselines. No claim that RL is the only valid solution is made.

## Statistics and trace selection

Primary: episode on-time-delivery fraction, with all generated packets including overflow in the denominator. Compute per-family means per trained seed; macro-average four structured families equally; report all three seed values, mean, and range. Plot seed dots. This implementation uses ranges rather than a deceptively precise confidence interval from three seeds. Baseline columns represent one paired scenario-indexed exploration stream; they have no training-seed variability estimate.

Also report expiry, overflow, normalized energy per delivery, conditional latency, goodput, and switching. Zero-denominator metrics remain missing. Conditional latency is interpreted with loss rates. Distinguish percentage-point improvement from relative percentages.

The demo is the first switching record in the frozen validation manifest, selected independently of final performance. It shows all-channel truth explicitly as evaluator-only information. The final test is used only after the settings/seed budget freeze. Preserve per-episode results, checkpoint hashes, source hashes, and suite hash.

No recurrent model, online weight fine-tuning, additional RL algorithm, real radio experiment, or retrained zero-retune ablation is included in the core. Those are deferred extensions. A negative result against a strong conventional baseline remains reportable evidence.
