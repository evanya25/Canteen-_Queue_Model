"""
Makes the results table for the slide from the model's own output.

STEP 1: run  python canteen_model_two_counters.py   (creates outputs/results_by_session.csv)
STEP 2: run  python make_results_table.py
Creates: outputs/results_table.csv and outputs/results_table.png
Nothing is typed in by hand: every number is calculated from the model output.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
r = pd.read_csv(os.path.join(OUT, "results_by_session.csv"))

# System A: average waiting time over the sessions
wait_A = r[r.system == "A"].avg_wait.mean()

def cell(system, alpha):
    """Average wait of a System B arrangement and its % change versus System A."""
    w = r[(r.system == system) & (r.alpha == alpha)].avg_wait.mean()
    pct = 100 * (w - wait_A) / wait_A
    sign = "+" if pct >= 0 else "−"
    return f"{w:.1f} ({sign}{abs(pct):.0f}%)"

rows = []
for a in sorted(r[r.system != "A"].alpha.unique()):
    rows.append([f"{int(round(a * 100))}%", f"{wait_A:.1f}",
                 cell("B shared", a), cell("B shelf", a)])
table = pd.DataFrame(rows, columns=["α", "System A", "Shared staff", "Cook + shelf"])
print(table.to_string(index=False))
table.to_csv(os.path.join(OUT, "results_table.csv"), index=False)

# Picture of the table
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.axis("off")
t = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="left")
t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 1.8)
for (i, j), c in t.get_celld().items():
    if i == 0: c.set_facecolor("#e6e3da"); c.set_text_props(weight="bold")
ax.set_title("Average wait in minutes (demonstration data, 2 counters)", loc="left")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "results_table.png"), dpi=200)
print("Saved outputs/results_table.csv and outputs/results_table.png")
