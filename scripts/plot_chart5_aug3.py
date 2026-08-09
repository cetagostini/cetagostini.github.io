"""Aug 3 timeline — main sessions and the agents running inside them."""

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
    "ytick.color": COLORS["ink_muted"], "grid.color": COLORS["line"],
    "grid.alpha": 0.5, "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.facecolor": COLORS["bg"],
    "savefig.bbox": "tight", "savefig.pad_inches": 0.4,
})

DB = "/Users/carlostrujillo/Documents/GitHub/personal-ai-projects/ai-tracking/backend/tracking.db"
OUT = "/Users/carlostrujillo/Documents/github/_worktrees/cetagostini.github.io/diary-hourly-rate-ai/images"

conn = sqlite3.connect(DB)
aug3 = pd.read_sql("""
    SELECT 
        CAST(STRFTIME('%H', m.timestamp) AS INT) as hour_utc,
        COUNT(DISTINCT m.conversation_id) as total_concurrent,
        COUNT(DISTINCT CASE WHEN c.is_subagent = 0 THEN m.conversation_id END) as main_sessions,
        COUNT(*) as messages
    FROM messages m
    JOIN conversations c ON m.conversation_id = c.id
    WHERE DATE(m.timestamp) = '2026-08-03'
    GROUP BY CAST(STRFTIME('%H', m.timestamp) AS INT)
    ORDER BY hour_utc
""", conn)
conn.close()

hours = aug3["hour_utc"].values
main = aug3["main_sessions"].values
total = aug3["total_concurrent"].values
agents = total - main
messages = aug3["messages"].values

fig, ax = plt.subplots(figsize=(10, 5.5))

# Agents layer (on top of main sessions)
ax.fill_between(hours, main, total, color=COLORS["secondary"], alpha=0.6,
                label="Agents running inside sessions", zorder=3)
# Main sessions layer
ax.fill_between(hours, 0, main, color=COLORS["primary"], alpha=0.85,
                label="Main sessions (human-directed)", zorder=4)
ax.plot(hours, total, color=COLORS["green_strong"], linewidth=1.5, zorder=5, alpha=0.6)

# Mark the peak
peak_h = hours[np.argmax(total)]
peak_t = total.max()
peak_m = main[np.argmax(total)]
ax.scatter([peak_h], [peak_t], color=COLORS["brown"], s=80, zorder=6,
           edgecolors="white", linewidth=1.5)

ax.annotate(
    f"{peak_m} sessions\norchestrating {peak_t - peak_m} agents\n{messages[np.argmax(total)]:,} messages",
    xy=(peak_h, peak_t),
    xytext=(peak_h - 5, peak_t * 0.55),
    fontsize=10, fontweight="bold", color=COLORS["brown"],
    arrowprops=dict(arrowstyle="-|>", color=COLORS["brown"], lw=2,
                    connectionstyle="arc3,rad=-0.15"),
    bbox=dict(boxstyle="round,pad=0.5", fc=COLORS["bg"], ec=COLORS["brown"], lw=1.2),
    ha="center", va="center", linespacing=1.5,
)

ax.set_xlim(-0.5, 23.5)
ax.set_ylim(0, peak_t * 1.15)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=9)
ax.set_xlabel("Time (UTC) — August 3, 2026", fontsize=11)
ax.set_ylabel("Concurrent conversations", fontsize=11)
ax.set_title("One session, many agents — August 3", fontsize=15, fontweight="bold",
             color=COLORS["brown"], loc="left", pad=12)
ax.legend(loc="upper left", framealpha=0.9, edgecolor=COLORS["line"], fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.savefig(f"{OUT}/aug3_parallel_timeline.png", dpi=200, facecolor=COLORS["bg"],
            bbox_inches="tight", pad_inches=0.4)
print(f"Saved: {OUT}/aug3_parallel_timeline.png")
plt.close()
