"""
visualization.py
=================
Plotting utilities for the performance-analysis dashboard (Phase IV).
All functions save a PNG to the given path and also return the Figure,
so they can be used both from scripts and interactively.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "raw": "#9aa0a6",
    "mitigated": "#1a73e8",
    "ideal": "#188038",
}


def plot_accuracy_vs_noise(
    noise_strengths: Sequence[float],
    raw_errors: Sequence[float],
    mitigated_errors: Sequence[float],
    title: str,
    save_path: str,
    ylabel: str = "Absolute error vs. ideal",
):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(noise_strengths, raw_errors, "o-", color=PALETTE["raw"], label="Unmitigated (raw)")
    ax.plot(noise_strengths, mitigated_errors, "s-", color=PALETTE["mitigated"], label="Mitigated")
    ax.set_xlabel("Noise strength")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_error_reduction_bar(
    technique_names: Sequence[str],
    error_reductions: Sequence[float],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#1a73e8" if v >= 0 else "#d93025" for v in error_reductions]
    bars = ax.bar(technique_names, error_reductions, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Error reduction (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, error_reductions):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_distribution_comparison(
    ideal: Dict[str, float],
    raw: Dict[str, float],
    mitigated: Dict[str, float],
    title: str,
    save_path: str,
):
    keys = sorted(set(ideal) | set(raw) | set(mitigated))
    x = np.arange(len(keys))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.6), 4.2))
    ax.bar(x - width, [ideal.get(k, 0) for k in keys], width, label="Ideal", color=PALETTE["ideal"])
    ax.bar(x, [raw.get(k, 0) for k in keys], width, label="Raw (noisy)", color=PALETTE["raw"])
    ax.bar(x + width, [mitigated.get(k, 0) for k in keys], width, label="Mitigated", color=PALETTE["mitigated"])
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_overhead_vs_accuracy(
    technique_names: Sequence[str],
    overhead_factors: Sequence[float],
    error_reductions: Sequence[float],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(overhead_factors, error_reductions, s=90, c="#1a73e8", zorder=3)
    for name, x, y in zip(technique_names, overhead_factors, error_reductions):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sampling / circuit-evaluation overhead (x baseline)")
    ax.set_ylabel("Error reduction (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_scalability(
    qubit_counts: Sequence[int],
    runtimes: Dict[str, Sequence[float]],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, times in runtimes.items():
        ax.plot(qubit_counts, times, "o-", label=name)
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
