import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_wave(t, signal, title="波形图", xlabel="时间 (s)", ylabel="幅值"):
    fig, ax = plt.subplots(figsize=(10,4), dpi=150)
    ax.plot(t, signal, color="#1f77b4", linewidth=1.4, antialiased=True)
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color="#666666", lw=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    plt.tight_layout()
    return fig

def draw_spectrum(freq, amp, title="频谱图"):
    fig, ax = plt.subplots(figsize=(10,4), dpi=150)
    markerline, stemlines, baseline = ax.stem(freq, amp, basefmt=" ", linefmt="#ff6666", markerfmt="o")
    markerline.set_markersize(4)
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color="#666666", lw=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("频率 (Hz)", fontsize=8)
    ax.set_ylabel("幅度", fontsize=8)
    ax.set_xlim(0, 50)
    plt.tight_layout()
    return fig
