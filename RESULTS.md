# neuromorphic-primitives — fixes, a demo, and benchmarks

## TL;DR

The test suite wasn't running at all (a broken import aborted collection), which hid
two real bugs: **the vectorized `Network` class could never actually receive external
input** (it could only ever propagate spikes it already had — i.e. never, from a cold
start), and STDP weight updates could silently exceed their own configured ceiling.
Both are fixed below, with tests that pin the fix. On top of the fixed engine, I built
a small reservoir-computing digit classifier and ran throughput/behavioral benchmarks.

## Bugs found and fixed

### 1. The whole test suite silently never ran
`tests/test_equivalence.py` imported `neuromorphic_original`, a module that does not
exist anywhere in the repo or its git history. Pytest aborts *collection* on any
import error, so `test_benchmark.py` and `test_properties.py` were never executing
either — not "passing", just never running. Rewrote it to compare the scalar `Neuron`
reference implementation (still in `neuron.py`) against the vectorized `Network`,
which is clearly what it was meant to check.

### 2. `Network.tick()`'s `external_pulses` never injected current (the big one)
```python
net = Network()
net.add_neuron(1); net.add_neuron(2)
net.connect(2, 1, 0.5)
for t in range(1, 6):
    net.tick(t, {1: [2]})   # "neuron 2 fires into neuron 1" -- every tick
# charge on neuron 1 stayed exactly 0.0 the entire time
```
`external_pulses` only ever updated the STDP timing clocks; the actual input-energy
calculation (`self.W @ self.prev_spike_vec`) only ever looked at spikes the network
had *already generated internally*. Since every network starts at `charge=0` and
nothing spikes spontaneously, **a freshly built `Network` had no documented way to
become active at all.** This is also why the repo's own benchmark was silently
measuring an idle network (see below).

Fixed by routing declared external pulses through the same weighted-input pathway the
scalar `Neuron` class already used correctly, plus a new direct-current-injection mode
for driving dedicated input/sensory neurons that have no incoming synapses of their
own (documented in the updated `tick()` docstring).

### 3. STDP weights could silently exceed `W_max` forever
The old merge-back into `W` rebuilt it via `sp.csr_matrix((data, (row, col)), ...)`,
which **sums duplicate `(row, col)` entries** instead of overwriting them — a real
overshoot hazard, caught by `test_properties.py`'s own invariant check. Rewrote
`_apply_stdp_learning_batch` and `_update_pre_spike_clocks` to be fully vectorized
(no more per-spiking-neuron Python loops, which were also the main scaling bottleneck)
and to write new weights directly into `W.data` in place — same positions, same
structure, no possibility of duplicate-summing.

### 4. Smaller fixes
- `connect()`/`connect_batch()` didn't clip initial weights to the destination
  neuron's `W_max`, so a weight passed in above the ceiling stayed there forever
  (only STDP-touched weights were ever bounded). Now clipped at connection time too.
- `Network.weights()` rounded to 5 decimals before returning, which could report a
  correctly-clipped weight as *larger* than `W_max` (`round(1.744775..., 5) ==
  1.74478 > 1.744775...`) — a display bug, not a real invariant violation, but it's
  what `test_properties.py` was actually catching on a second pass. Now returns full
  precision.
- `test_properties.py` used Hypothesis's default 200ms deadline on a test that runs
  a real numerical simulation; disabled the deadline (this is now more relevant since
  the network genuinely does work per the fix above).
- `numba` was a hard dependency (`requirements.txt`) with `njit` imported but never
  once applied anywhere in the codebase — dead weight. Added a real JIT kernel for
  the STDP hot path (see benchmark below) instead of leaving it inert.

All 5 tests pass, repeatably, across multiple fresh Hypothesis runs (cleared
`.hypothesis` cache and reran 4x with new random seeds).

## What this looked like in practice

![before vs after](before_after_activity.png)

Same 1000-neuron network, same stimulus pattern the repo's own
`tests/test_benchmark.py` used, 200 ticks: the original code produced **zero spikes,
ever**. The "benchmark" was timing an idle simulation the whole time.

## Something built on top of it: a reservoir-computing digit classifier

`examples/reservoir_digits.py` — this is only possible now that external input
actually works. Uses sklearn's built-in `digits` dataset (1797 8×8 images, 10 classes,
ships with scikit-learn, no download needed).

- 64 pixel intensities → rate-coded spike trains over 30 ticks, driving 64 dedicated
  input neurons (direct-current injection, the new list-form `external_pulses`).
- Input neurons project sparsely (15%) into a 150-neuron recurrent reservoir (5%
  recurrent connectivity), built from the same primitive.
- Each reservoir neuron's spike count over the window is a feature; a plain
  multinomial logistic regression readout is trained on those features. The
  reservoir itself is never trained by gradient descent — only, optionally, by the
  primitive's own built-in STDP.

Three conditions, 600-sample subset (420 train / 180 test), stratified:

| condition | train acc | test acc | mean reservoir weight | std |
|---|---|---|---|---|
| raw-pixel baseline (no reservoir) | 1.000 | **0.983** | n/a | n/a |
| frozen reservoir (STDP off) | 1.000 | **0.906** | 0.111 | 0.064 |
| STDP-adapted reservoir | 0.257 | **0.200** | 0.255 | 0.259 |

![accuracy comparison](reservoir_accuracy.png)

**The frozen random reservoir gets 90.6% test accuracy** — a random spiking
projection through this primitive preserves most of the class-discriminative signal
(unsurprising for this dataset, but a genuine confirmation the primitive's dynamics
aren't destroying information).

**Letting STDP self-organize the reservoir on the unlabeled training stream first
made things dramatically worse (20% test accuracy).** This isn't a pipeline bug — I
checked: after adaptation, **48.7% of reservoir neurons go permanently silent**
(never spike on any test image), versus healthy, varied activity in the frozen
case. This is a real and well-known failure mode of purely-Hebbian learning in an
all-excitatory recurrent network: without inhibitory synapses or synaptic
normalization to balance it, STDP drives a small subset of connections to the weight
ceiling while starving the rest — a winner-take-all collapse. It matches what the
before/after benchmark also hinted at (that dense/strongly-driven networks with this
primitive saturate quickly): **this primitive has no inhibitory (negative) synapses
at all**, so there's currently no way to build a network that self-stabilizes under
its own learning rule. That's the most useful concrete direction for future work.

## Benchmarks

### Throughput scaling (fixed implementation, genuinely active networks)

![throughput scaling](throughput_scaling.png)

| neurons | synapses (5% conn.) | elapsed | throughput | total spikes |
|---|---|---|---|---|
| 200 | 1,988 | 0.213s | 140,635 neuron-steps/s | 4,190 |
| 500 | 12,477 | 0.128s | 587,621 neuron-steps/s | 20,320 |
| 1,000 | 49,938 | 0.350s | 429,128 neuron-steps/s | 44,184 |
| 2,000 | 199,891 | 1.256s | 238,819 neuron-steps/s | 92,088 |
| 5,000 | 1,249,772 | 7.565s | 99,139 neuron-steps/s | 235,000 |

At fixed 5% connectivity, synapse count grows roughly with N², so per-neuron-step
throughput falls off at larger N — expected, since STDP and clock-update cost scale
with active synapse count, not neuron count alone.

### numba JIT kernel: does it actually help?

```
numba ON : 1.486s   201,930 neuron-steps/s
numba OFF: 1.867s   160,667 neuron-steps/s
speedup: 1.26x
```
Both paths produce **bit-identical spike counts** (90,150 either way) — a useful
correctness check in addition to the speed number. A real, if modest, win; worth
keeping now that it's actually wired in.

## Running it yourself

```bash
pip install -e ".[dev]" scikit-learn matplotlib
pytest tests/ -v
python benchmarks/before_after.py
python benchmarks/scaling_benchmark.py
python examples/reservoir_digits.py
```
