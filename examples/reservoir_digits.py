"""
Reservoir computing digit classifier built directly on top of the neuromorphic
primitive (neuron.py / network.py), using sklearn's built-in `digits` dataset
(1797 8x8 handwritten digit images, 10 classes, ships with scikit-learn --
no network access needed).

This is only possible now that Network.tick()'s external_pulses bug is fixed
(see network.py) -- previously there was no way to drive the network with real
input at all.

Pipeline
--------
1. Each 8x8 image (64 pixels, values 0-16) is rate-coded into a Poisson-ish
   spike train over T ticks: pixel intensity -> per-tick firing probability.
2. 64 dedicated input neurons are driven directly (list-form external_pulses =
   direct current injection) and project into a sparse recurrent reservoir of
   spiking neurons built from the same primitive.
3. The reservoir's per-neuron spike COUNT over the presentation window is used
   as a fixed-length feature vector (standard "rate readout" reservoir computing).
4. A simple linear classifier (multinomial logistic regression) is trained on
   those features -- the reservoir itself is never trained by gradient descent,
   only (optionally) by the primitive's own built-in STDP.

Three conditions are compared:
  (a) baseline    -- logistic regression directly on raw 64 pixel intensities.
  (b) frozen       -- fixed random reservoir (STDP disabled, alpha_plus=alpha_minus=0),
                      classic Echo State Network / Liquid State Machine style.
  (c) stdp-adapted -- same initial random reservoir, but STDP is left ON while a
                      block of training images is streamed through it once
                      (unsupervised self-organization), then frozen for readout.

This directly exercises the primitive's distinguishing feature (STDP-driven
synaptic adaptation) and asks a real empirical question: does letting the
reservoir self-organize on unlabeled data help a downstream linear classifier?
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuromorphic import Network
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


N_IN = 64
N_RES = 150
T_TICKS = 30
MAX_FIRE_PROB = 0.8
IN_TO_RES_CONNECTIVITY = 0.15
RES_RECURRENT_CONNECTIVITY = 0.05
SEED = 0


def build_reservoir(seed, plastic=False):
    """Build the input + reservoir topology. `plastic` toggles STDP on/off."""
    rng = np.random.default_rng(seed)
    net = Network()
    net.add_neurons(list(range(N_IN)))  # input layer: driven externally only, never
                                          # has incoming synapses, so its own
                                          # dynamics never fire it spontaneously
    stdp_kwargs = dict(alpha_plus=0.02, alpha_minus=0.015, tau_stdp=15.0) if plastic \
        else dict(alpha_plus=0.0, alpha_minus=0.0)
    net.add_neurons(
        list(range(N_IN, N_IN + N_RES)),
        R_min=2.0, lambda_leak=0.15, T_base=1.0, beta=0.5, gamma=0.02, W_max=1.0,
        **stdp_kwargs
    )

    conns = []
    for i in range(N_IN):
        targets = rng.choice(N_RES, size=int(N_RES * IN_TO_RES_CONNECTIVITY), replace=False)
        for r in targets:
            conns.append((i, N_IN + int(r), float(rng.uniform(0.05, 0.25))))
    for i in range(N_RES):
        targets = rng.choice(N_RES, size=int(N_RES * RES_RECURRENT_CONNECTIVITY), replace=False)
        for r in targets:
            if r != i:
                conns.append((N_IN + i, N_IN + int(r), float(rng.uniform(0.02, 0.1))))
    net.connect_batch(conns)
    return net


def present_sample(net, image01, rng, learn=False):
    """
    Run one image through the reservoir for T_TICKS and return the reservoir's
    per-neuron spike-count feature vector. `image01` is length-64, in [0, 1].
    Reservoir dynamic state (charge/H/U/spike history) is reset before and after
    so samples don't bleed into each other; synaptic weights (and hence any STDP
    learning) persist across calls unless the caller freezes them first.
    """
    net.reset()
    counts = np.zeros(N_RES)
    for t in range(1, T_TICKS + 1):
        active_inputs = [i for i in range(N_IN) if rng.random() < image01[i] * MAX_FIRE_PROB]
        result = net.tick(t, active_inputs)
        for r in range(N_RES):
            counts[r] += result[N_IN + r]
    return counts


def freeze(net):
    """Stop STDP from making any further changes (used after an adaptation phase)."""
    net.state.alpha_plus[N_IN:] = 0.0
    net.state.alpha_minus[N_IN:] = 0.0


def extract_features(net, images01, seed):
    rng = np.random.default_rng(seed)
    feats = np.zeros((len(images01), N_RES))
    for idx, img in enumerate(images01):
        feats[idx] = present_sample(net, img, rng)
    return feats


def run_condition(label, net, X_train01, X_test01, y_train, y_test):
    t0 = time.time()
    train_feats = extract_features(net, X_train01, seed=1)
    test_feats = extract_features(net, X_test01, seed=2)
    extract_time = time.time() - t0

    scaler = StandardScaler().fit(train_feats)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(train_feats), y_train)
    train_acc = accuracy_score(y_train, clf.predict(scaler.transform(train_feats)))
    test_acc = accuracy_score(y_test, clf.predict(scaler.transform(test_feats)))

    weights = net.weights()
    res_weights = [w for pre, post, w in weights if post >= N_IN]
    print(f"[{label}] train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
          f"feature_extract_time={extract_time:.2f}s  "
          f"mean_reservoir_weight={np.mean(res_weights):.4f}  "
          f"std={np.std(res_weights):.4f}")
    return dict(label=label, train_acc=train_acc, test_acc=test_acc,
                extract_time=extract_time, mean_weight=float(np.mean(res_weights)),
                std_weight=float(np.std(res_weights)))


def main(n_samples=600):
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    if n_samples < len(X):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_train01, X_test01, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    print(f"digits dataset subset: {len(X)} samples "
          f"({len(X_train01)} train / {len(X_test01)} test), 10 classes")
    print(f"reservoir: {N_RES} neurons, {T_TICKS} ticks/sample, "
          f"input connectivity={IN_TO_RES_CONNECTIVITY}, "
          f"recurrent connectivity={RES_RECURRENT_CONNECTIVITY}")
    print()

    results = []

    # (a) baseline: raw pixels, no reservoir at all
    t0 = time.time()
    scaler = StandardScaler().fit(X_train01)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(X_train01), y_train)
    base_train_acc = accuracy_score(y_train, clf.predict(scaler.transform(X_train01)))
    base_test_acc = accuracy_score(y_test, clf.predict(scaler.transform(X_test01)))
    print(f"[raw-pixel baseline] train_acc={base_train_acc:.4f}  "
          f"test_acc={base_test_acc:.4f}  time={time.time()-t0:.2f}s")
    results.append(dict(label="raw-pixel baseline", train_acc=base_train_acc,
                         test_acc=base_test_acc, extract_time=0.0,
                         mean_weight=None, std_weight=None))

    # (b) frozen random reservoir (STDP off)
    net_frozen = build_reservoir(seed=SEED, plastic=False)
    results.append(run_condition("frozen reservoir", net_frozen, X_train01, X_test01, y_train, y_test))

    # (c) STDP-adapted reservoir: let it self-organize on the training stream once
    # (unsupervised -- labels are never shown to the reservoir), then freeze it
    net_plastic = build_reservoir(seed=SEED, plastic=True)
    rng = np.random.default_rng(3)
    print("adapting reservoir with STDP over the training stream (unsupervised)...")
    t0 = time.time()
    for img in X_train01:
        present_sample(net_plastic, img, rng, learn=True)
    print(f"  adaptation pass: {time.time()-t0:.2f}s")
    freeze(net_plastic)
    results.append(run_condition("STDP-adapted reservoir", net_plastic, X_train01, X_test01, y_train, y_test))

    print()
    print("=" * 78)
    print(f"{'condition':<26}{'train_acc':>11}{'test_acc':>11}{'mean_w':>10}{'std_w':>10}")
    print("=" * 78)
    for r in results:
        mw = f"{r['mean_weight']:.4f}" if r['mean_weight'] is not None else "n/a"
        sw = f"{r['std_weight']:.4f}" if r['std_weight'] is not None else "n/a"
        print(f"{r['label']:<26}{r['train_acc']:>11.4f}{r['test_acc']:>11.4f}{mw:>10}{sw:>10}")

    return results


if __name__ == "__main__":
    main()
