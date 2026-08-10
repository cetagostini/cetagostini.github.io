"""Chart 4 — The team: specialized agent types."""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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
agents = pd.read_sql("""
    SELECT agent_type, COUNT(*) as sessions, SUM(message_count) as messages
    FROM conversations
    WHERE start_time IS NOT NULL AND DATE(start_time) >= '2026-08-01' AND is_subagent = 1
    GROUP BY agent_type ORDER BY sessions DESC LIMIT 8
""", conn)
conn.close()

top = agents.sort_values("sessions", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))

norm = plt.Normalize(top["sessions"].min(), top["sessions"].max())
cmap = mpl.colors.LinearSegmentedColormap.from_list("", [COLORS["accent"], COLORS["primary"]])

bars = ax.barh(range(len(top)), top["sessions"], color=cmap(norm(top["sessions"].values)),
               zorder=3, height=0.55, edgecolor=COLORS["line"], linewidth=0.5)

for i, (_, row) in enumerate(top.iterrows()):
    msgs_k = f"{row['messages']/1000:.1f}k"
    ax.text(row["sessions"] + 2, i,
            f'{int(row["sessions"])} sessions · {msgs_k} msgs',
            va="center", fontsize=9, color=COLORS["ink_muted"])

ax.set_yticks(range(len(top)))
ax.set_yticklabels(top["agent_type"], fontsize=10)
ax.set_title("The team — specialized agents", fontsize=15, fontweight="bold",
             color=COLORS["brown"], loc="left", pad=12)
ax.set_xlabel("Sessions", fontsize=12)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_xlim(0, top["sessions"].max() * 1.55)

plt.savefig(f"{OUT}/chart_agent_types.png", dpi=200, facecolor=COLORS["bg"],
            bbox_inches="tight", pad_inches=0.3)
print(f"Saved: {OUT}/chart_agent_types.png")
plt.close()
