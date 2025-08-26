import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_jsd_heatmap(jsd_map, out_path, title=None):
    names = list(jsd_map.keys())
    if not names:
        return
    T = max(len(v) for v in jsd_map.values())
    M = []
    for n in names:
        v = jsd_map[n]
        if len(v) < T:
            pad = np.zeros((T,))
            pad[:len(v)] = v
            M.append(pad)
        else:
            M.append(v)
    M = np.stack(M, axis=0)
    fig = plt.figure()
    plt.imshow(M, aspect="auto")
    plt.yticks(range(len(names)), names)
    plt.xlabel("generation step")
    if title:
        plt.title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_layer_mean_jsd(jsd_map, out_path, title=None):
    names = list(jsd_map.keys())
    if not names:
        return
    means = [float(np.mean(jsd_map[n])) for n in names]
    fig = plt.figure()
    plt.bar(range(len(names)), means)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel("mean JSD")
    if title:
        plt.title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
