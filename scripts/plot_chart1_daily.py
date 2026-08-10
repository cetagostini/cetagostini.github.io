"""Chart 1 — Daily main sessions with agent spawn counts."""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

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
    "grid.alpha": 0.6, "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.facecolor": COLORS["bg"],
    "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
})

DB = "/Users/carlostrujillo/Documents/GitHub/personal-ai-projects/ai-tracking/backend/tracking.db"
OUT = "/Users/carlostrujillo/Documents/github/_worktrees/cetagostini.github.io/diary-hourly-rate-ai/images"

conn = sqlite3.connect(DB)
daily = pd.read_sql("""
    SELECT DATE(c.start_time) as day,
        COUNT(*) as sessions,
        COALESCE(SUM(ch.child_count), 0) as agents_spawned
    FROM conversations c
    LEFT JOIN (
        SELECT parent_id, COUNT(*) as child_count 
        FROM conversations WHERE is_subagent = 1 
        GROUP BY parent_id
    ) ch ON c.id = ch.parent_id
    WHERE c.start_time IS NOT NULL AND DATE(c.start_time) >= '2026-08-01' AND c.is_subagent = 0
    GROUP BY DATE(c.start_time)
    ORDER BY day
""", conn)
conn.close()
daily["day"] = pd.to_datetime(daily["day"])

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(daily["day"], daily["sessions"], color=COLORS["primary"],
              width=0.7, zorder=3, edgecolor=COLORS["line"], linewidth=0.5)

for _, row in daily.iterrows():
    label = f'{int(row["sessions"])}'
    if row["agents_spawned"] > 0:
        label += f'\n({int(row["agents_spawned"])} agents)'
    ax.text(row["day"], row["sessions"] + 0.3, label, ha="center", va="bottom",
            fontsize=8, color=COLORS["ink_muted"], fontweight="medium", linespacing=1.3)

ax.set_title("Daily work sessions, August 2026", fontsize=15, fontweight="bold",
             color=COLORS["brown"], loc="left", pad=12)
ax.set_ylabel("Sessions", fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, daily["sessions"].max() * 1.6)

ax.text(0.98, 0.95, "Parentheses = agents spawned\ninside those sessions",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        color=COLORS["ink_muted"], style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg"], ec=COLORS["line"], lw=0.5))

plt.savefig(f"{OUT}/chart_daily_sessions.png", dpi=200, facecolor=COLORS["bg"],
            bbox_inches="tight", pad_inches=0.3)
print(f"Saved: {OUT}/chart_daily_sessions.png")
plt.close()
