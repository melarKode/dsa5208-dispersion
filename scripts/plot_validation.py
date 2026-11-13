"""Plot measured analytical convergence from the validation report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("report", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
data = json.loads(a.report.read_text())
fig, ax = plt.subplots(figsize=(7, 4.5))
for name in sorted({r["case"] for r in data["convergence"]}):
    entries = sorted((r for r in data["convergence"] if r["case"] == name), key=lambda r: r["n"])
    n = np.array([r["n"] for r in entries]); error = np.array([r["l1"] for r in entries])
    order = np.log(error[-2] / error[-1]) / np.log(n[-1] / n[-2])
    ax.loglog(n, error, "o-", label=f"{name}, final order {order:.2f}")
ax.set(xlabel="Cells per axis", ylabel="Volume-weighted L1 error", title="Analytical convergence: periodic transport")
ax.legend(); ax.grid(True, which="both", alpha=.2)
fig.tight_layout(); fig.savefig(a.output, dpi=180)
