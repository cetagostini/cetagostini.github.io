"""Chart 3 — Clock hours vs LLM work hours, diverging over time."""

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
    "grid.alpha": 0.5, "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.facecolor": COLORS["bg"],
    "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
})

DB = "/Users/carlostrujillo/Documents/GitHub/personal-ai-projects/ai-tracking/backend/tracking.db"
OUT = "/Users/carlostrujillo/Documents/github/_worktrees/cetagostini.github.io/diary-hourly-rate-ai/images"

conn = sqlite3.connect(DB)
data = pd.read_sql("""
    WITH hourly AS (
        SELECT DATE(timestamp) as day,
            CAST(STRFTIME('%H', timestamp) AS INT) as hour_utc,
            COUNT(DISTINCT conversation_id) as concurrent
        FROM messages WHERE DATE(timestamp) >= '2026-08-01'
        GROUP BY DATE(timestamp), CAST(STRFTIME('%H', timestamp) AS INT)
    )
    SELECT day,
        COUNT(*) as clock_hours,
        SUM(concurrent) as llm_work_hours
    FROM hourly
    GROUP BY day ORDER BY day
""", conn)
conn.close()
data["day"] = pd.to_datetime(data["day"])
data["multiplier"] = data["llm_work_hours"] / data["clock_hours"]

fig, ax = plt.subplots(figsize=(10, 5.5))

# LLM work hours — the climbing line
ax.plot(data["day"], data["llm_work_hours"], color=COLORS["primary"], linewidth=2.5,
        marker="o", markersize=7, zorder=4, label="LLM work-hours")
ax.fill_between(data["day"], 0, data["llm_work_hours"], color=COLORS["primary"], alpha=0.12, zorder=2)

# Clock hours — the flat line
ax.plot(data["day"], data["clock_hours"], color=COLORS["accent"], linewidth=2.5,
        marker="s", markersize=6, zorder=4, label="Clock hours")
ax.fill_between(data["day"], 0, data["clock_hours"], color=COLORS["accent"], alpha=0.2, zorder=3)

# Multiplier labels on the gap
for _, row in data.iterrows():
    mid_y = (row["clock_hours"] + row["llm_work_hours"]) / 2
    ax.annotate(f'{row["multiplier"]:.0f}×',
                xy=(row["day"], mid_y),
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=COLORS["green_strong"],
                bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["bg"], ec=COLORS["line"], lw=0.5))

# Value labels at endpoints
last = data.iloc[-1]
ax.text(last["day"] + pd.Timedelta(hours=6), last["llm_work_hours"],
        f'{int(last["llm_work_hours"])}h', va="center", fontsize=10,
        fontweight="bold", color=COLORS["primary"])
ax.text(last["day"] + pd.Timedelta(hours=6), last["clock_hours"],
        f'{int(last["clock_hours"])}h', va="center", fontsize=10,
        fontweight="bold", color=COLORS["ink_muted"])

ax.set_title("Clock hours vs LLM work-hours", fontsize=15, fontweight="bold",
             color=COLORS["brown"], loc="left", pad=12)
ax.set_ylabel("Hours", fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend(loc="upper left", framealpha=0.9, edgecolor=COLORS["line"], fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_xlim(data["day"].min() - pd.Timedelta(hours=12),
            data["day"].max() + pd.Timedelta(hours=18))

plt.savefig(f"{OUT}/chart_clock_vs_llm.png", dpi=200, facecolor=COLORS["bg"],
            bbox_inches="tight", pad_inches=0.3)
print(f"Saved: {OUT}/chart_clock_vs_llm.png")
plt.close()
