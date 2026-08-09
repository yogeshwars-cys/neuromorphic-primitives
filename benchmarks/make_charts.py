import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Chart 1: throughput scaling (measured in scaling_benchmark.py) ---
sizes = [200, 500, 1000, 2000, 5000]
throughput = [140635, 587621, 429128, 238819, 99139]
synapses = [1988, 12477, 49938, 199891, 1249772]

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sizes, throughput, marker='o', color='#2b6cb0', linewidth=2, label='neuron-steps/sec')
ax1.set_xlabel('Network size (neurons)')
ax1.set_ylabel('Throughput (neuron-steps/sec)', color='#2b6cb0')
ax1.tick_params(axis='y', labelcolor='#2b6cb0')
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.plot(sizes, synapses, marker='s', color='#c05621', linewidth=2, linestyle='--', label='synapse count')
ax2.set_ylabel('Synapse count (5% connectivity)', color='#c05621')
ax2.tick_params(axis='y', labelcolor='#c05621')
ax2.set_yscale('log')
plt.title('Fixed Network: throughput vs. size (genuinely active, STDP running)')
fig.tight_layout()
plt.savefig('/mnt/user-data/outputs/throughput_scaling.png', dpi=140)
plt.close()

# --- Chart 2: reservoir computing accuracy comparison ---
labels = ['raw-pixel\nbaseline', 'frozen\nreservoir', 'STDP-adapted\nreservoir']
train_acc = [1.0000, 1.0000, 0.2571]
test_acc = [0.9833, 0.9056, 0.2000]

x = np.arange(len(labels))
width = 0.32
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.bar(x - width/2, train_acc, width, label='train accuracy', color='#4c51bf')
ax.bar(x + width/2, test_acc, width, label='test accuracy', color='#38a169')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel('Accuracy')
ax.set_title('Digit classification via spiking reservoir features\n(sklearn digits, 10 classes, 600-sample subset)')
ax.legend()
for i, (tr, te) in enumerate(zip(train_acc, test_acc)):
    ax.text(i - width/2, tr + 0.02, f"{tr:.2f}", ha='center', fontsize=9)
    ax.text(i + width/2, te + 0.02, f"{te:.2f}", ha='center', fontsize=9)
fig.tight_layout()
plt.savefig('/mnt/user-data/outputs/reservoir_accuracy.png', dpi=140)
plt.close()

# --- Chart 3: before/after activity comparison ---
labels2 = ['Original\n(pre-fix)', 'Fixed']
spikes = [0, 60100]
fig, ax = plt.subplots(figsize=(5, 4.2))
bars = ax.bar(labels2, spikes, color=['#a0aec0', '#2b6cb0'])
ax.set_ylabel('Total spikes over 200-tick benchmark run')
ax.set_title('external_pulses bug: before vs. after\n(1000-neuron network, identical stimulus)')
for b, s in zip(bars, spikes):
    ax.text(b.get_x() + b.get_width()/2, s + 800, f"{s:,}", ha='center', fontweight='bold')
fig.tight_layout()
plt.savefig('/mnt/user-data/outputs/before_after_activity.png', dpi=140)
plt.close()

print("charts written")
