# Simulator specification

Version: `radio48-v1`. Full implementation: `src/radio_rl/env.py`. All quantities are dimensionless slots, packets, and normalized energy; no real RF hardware is used.

One transmitter/receiver pair tunes together across eight channels. Packets arrive with Bernoulli probability sampled uniformly from 0.35–0.65 per episode. Each arrival has a uniformly sampled relative deadline 4, 8, or 12. Eight queued packets are ordered by earliest deadline, then arrival and ID. Drop-tail overflow rejects the new packet. The controller chooses channels, not which packet to serve.

## Causal timing

Slot zero's arrival precedes the first observation. At each decision: apply the selected action; attempt service if permitted; register any detected transmission for a future jammer response; expire undelivered packets whose inclusive deadline is this slot; advance the interference or initialize a newly scheduled mode; admit the next slot's arrival; return observation and reward. Next-slot overflow is charged on this transition. There are 500 arrival slots and 12 drain slots. Remaining mission time is observed, so slot 511 ends with `terminated=True`, not an artificial truncation.

A packet arriving at `a` with relative deadline `D` may be served through `a+D-1`. Latency is `service_slot-a+1`. Every step enforces `generated = delivered + expired + overflow + queued`; the terminal queue must be empty. Reliable ACKs describe actual transmission success. No waveform, physical synchronization, ACK loss, or calibrated power model is claimed.

## Observation schema

| Zero-based indices | Meaning |
|---|---|
| 0:8 | Tuned-channel one-hot |
| 8:32 | Eight EDF rows: presence, age/12, remaining service opportunities/12 |
| 32 | Remaining mission fraction |
| 33 | Remaining arrival-phase fraction |
| 34:43 | Previous action, nine-way one-hot; zero at reset |
| 43:47 | Previous outcome: success, failure, retuned, idle; zero at reset |
| 47 | Previous-event validity |

The policy receives 48 float32 values. H=16 concatenates the last 16 observations into 768 values; H=1 uses 48. Histories are zero-padded only at real episode starts. H=1 still observes the previous event. The smaller network input is a parameter-count confound in the memory comparison.

## Actions and reward

Actions 0–7 choose a channel; 8 waits. A changed channel consumes one entire slot without emitting a packet. Staying on the tuned channel attempts one EDF packet if queued. Empty-queue retuning is permitted and costs energy. Waiting keeps the channel without transmission. All nine actions are always legal.

`r = delivered - 2*(expired + overflow) - 0.02*retuned - 0.01*energy`, where `energy = attempted + 0.2*retuned`. Discount is 0.99. Energy applies to failed attempts too. Reporting prioritizes on-time delivery over return. No hidden jammer-state bonus or future-information shaping is present.

## Interference

Success probability is `0.98` on a clean channel and `0.98*0.02` on a jammed channel. Each channel/slot has a pre-indexed success uniform variate; policies cannot access it.

- Sweep: contiguous width 2–4, direction ±1, movement probability 0.25–0.75, random initial phase.
- Burst: uniformly selected width-2–4 subset when an ON burst begins; initial OFF; OFF-to-ON probability 0.05–0.2; ON persistence 0.6–0.9.
- Follower: detects only actual packet emissions with probability 0.6–1; responds after 1–3 slots, jamming that channel for 2–5 slots. An event accepted at v occupies v through v+dwell-1. Latest detected emission wins among due events, resetting dwell. No current-action instant reaction is permitted.
- Switching: two hidden switch times in slots 100–399, at least 100 apart. Every segment uses a different family than the preceding segment. A mode change discards all old active/pending interference state.
- Controls: no interference; independent uniform K-of-8 interference with K=2 or 4.

Training chooses each structured single-family or switching scenario with probability 0.25. Other parameters are independently sampled within their documented ranges once per episode. A single episode's parameters are shared when revisiting that family; they do not change at every segment. Nominal/stress parameters and switches are saved in the scenario manifests.

## Information and pairing

Policy calls receive observations, actions, reward, and end flags. They never receive `info`, current masks, latent mode, future schedules, or random arrays. Debug traces are evaluator-only. Separate arrival, deadline, success, detection, interference, and initialization streams are derived from each seed. Changing a policy cannot change the offered arrivals or underlying uniforms. A reactive jammer's realized masks legitimately differ between policies because their emissions differ.

No-interference and IID controls have analytical checks. The IID per-attempt mean is `q*(1-K/N+rho*K/N)`, not an overall delivery bound when channel changes consume service time.
