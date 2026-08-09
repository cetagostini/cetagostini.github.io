"""Chart 2 — Hourly concurrency heatmap."""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

COLORS = {
    "primary": "#778873", "secondary": "#A1BC98", "accent": "#DCCFC0",
    "bg": "#FDF6ED", "ink": "#2B2A26", "ink_muted": "#6B665C",
    "green_strong": "#4F6B4A", "line": "#E6DFD2", "brown": "#6B5A48",
}
mpl.rcParams.update({
    "figure.facecolor": COLORS["bg"], "axes.facecolor": COLORS["bg"],
    "axes.edgecolor": COLORS["line"], "axes.labelcolor": COLORS["ink"],
    "text.color": COLORS["ink"], "xtick.color": COLORS["ink_muted"],
    "ytick.color": COLORS["ink_muted"],
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.facecolor": COLORS["bg"],
    "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
})

DB = "/Users/carlostrujillo/Documents/GitHub/personal-ai-projects/ai-tracking/backend/tracking.db"
OUT = "/Users/carlostrujillo/Documents/github/_worktrees/cetagostini.github.io/diary-hourly-rate-ai/images"

conn = sqlite3.connect(DB)
hourly = pd.read_sql("""
    SELECT DATE(timestamp) as day,
        CAST(STRFTIME('%H', timestamp) AS INT) as hour_utc,
        COUNT(DISTINCT conversation_id) as concurrent
    FROM messages WHERE DATE(timestamp) >= '2026-08-01'
    GROUP BY DATE(timestamp), CAST(STRFTIME('%H', timestamp) AS INT)
    ORDER BY day, hour_utc
""", conn)
conn.close()
hourly["day"] = pd.to_datetime(hourly["day"])

days = sorted(hourly["day"].unique())
heat = np.zeros((len(days), 24))
for _, row in hourly.iterrows():
    di = days.index(row["day"])
    heat[di, int(row["hour_utc"])] = row["concurrent"]

fig, ax = plt.subplots(figsize=(10, 5))

im = ax.imshow(heat, aspect="auto", cmap="YlGn", origin="lower", vmin=0)
ax.set_yticks(range(len(days)))
ax.set_yticklabels([pd.Timestamp(d).strftime("%b %d") for d in days], fontsize=9)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=9)
ax.set_xlabel("Hour (UTC)", fontsize=12)

peak_idx = np.unravel_index(heat.argmax(), heat.shape)
peak_val = int(heat[peak_idx])
ax.annotate(f"Peak: {peak_val} simultaneous",
            xy=(peak_idx[1], peak_idx[0]),
            xytext=(peak_idx[1] - 5, peak_idx[0] - 0.8),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["brown"], lw=1.5,
                            connectionstyle="arc3,rad=0.2"),
            fontsize=10, fontweight="bold", color=COLORS["brown"],
            bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg"], ec=COLORS["line"], lw=0.8))

ax.set_title("Concurrent conversations by hour", fontsize=15, fontweight="bold",
             color=COLORS["brown"], loc="left", pad=12)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("Active conversations", fontsize=10)

plt.savefig(f"{OUT}/chart_heatmap.png", dpi=200, facecolor=COLORS["bg"],
            bbox_inches="tight", pad_inches=0.3)
print(f"Saved: {OUT}/chart_heatmap.png")
plt.close()
