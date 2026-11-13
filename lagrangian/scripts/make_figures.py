#!/usr/bin/env python3
"""Report figures from a Vanda scaling run (the REVIEW folder of scripts/vanda_scaling.pbs).

Usage: make_figures.py REVIEW_DIR OUT_DIR

Writes SVG figures (standard library only, so it runs anywhere):
  strong_speedup.svg     strong scaling: speedup vs cores per layout, with ideal
  strong_efficiency.svg  strong scaling: parallel efficiency vs cores
  weak_efficiency.svg    weak scaling: efficiency vs cores (1M particles per core)
  time_split.svg         time loop split into stepping / wait / migration,
                         balanced cloud vs compact plume, 16 cores
  plume_imbalance.svg    max/mean particles per rank over time for the plume

Colour follows the layout in every chart: threads-only = slot 1 (blue),
MPI-only = slot 2 (orange), hybrid = slot 3 (aqua) of the dataviz reference
palette (the first three slots are validated for all pairs).
"""

import csv
import math
import sys
from pathlib import Path

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
AXIS = "#b9b8b2"
IDEAL = "#8a8983"
SERIES = {"threads": "#2a78d6", "mpi": "#eb6834", "hybrid": "#1baf7a"}
SEGMENTS = [("stepping", "Stepping (physics)", "#2a78d6"), ("wait", "Wait (load imbalance)", "#eb6834"),
            ("migration", "Migration (exchange)", "#1baf7a")]
LAYOUT_NAME = {"threads": "Threads only", "mpi": "MPI only", "hybrid": "Hybrid (4 threads/rank)"}
FONT = "Helvetica, Arial, sans-serif"
SHORT_NAME = {"threads": "Threads", "mpi": "MPI", "hybrid": "Hybrid"}


def read_rows(path):
    with open(str(path), newline="") as f:
        return list(csv.DictReader(f))


def layout_of(ranks, threads):
    if ranks == 1 and threads == 1:
        return "base"
    if ranks == 1:
        return "threads"
    if threads == 1:
        return "mpi"
    return "hybrid"


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.parts = ['<rect width="{}" height="{}" fill="{}"/>'.format(width, height, SURFACE)]

    def text(self, x, y, s, size=12, color=TEXT_2, anchor="start", weight="normal"):
        self.parts.append('<text x="{:.1f}" y="{:.1f}" font-family="{}" font-size="{}" fill="{}" '
                          'text-anchor="{}" font-weight="{}">{}</text>'.format(x, y, FONT, size, color, anchor,
                                                                               weight, esc(s)))

    def line(self, x1, y1, x2, y2, color, width=1, dash=None):
        d = ' stroke-dasharray="{}"'.format(dash) if dash else ""
        self.parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="{}"{}/>'
                          .format(x1, y1, x2, y2, color, width, d))

    def polyline(self, points, color, width=2, dash=None):
        d = ' stroke-dasharray="{}"'.format(dash) if dash else ""
        pts = " ".join("{:.1f},{:.1f}".format(x, y) for x, y in points)
        self.parts.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="{}" stroke-linejoin="round" '
                          'stroke-linecap="round"{}/>'.format(pts, color, width, d))

    def marker(self, x, y, color):
        # 9 px dot with a 2 px surface ring so overlapping markers stay distinct.
        self.parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="4.5" fill="{}" stroke="{}" stroke-width="2"/>'
                          .format(x, y, color, SURFACE))

    def rect(self, x, y, w, h, color, rx=0):
        self.parts.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" rx="{}" fill="{}"/>'
                          .format(x, y, max(w, 0), h, rx, color))

    def path(self, d, color):
        self.parts.append('<path d="{}" fill="{}"/>'.format(d, color))

    def save(self, path):
        body = "\n".join(self.parts)
        Path(path).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'
                              '\n{2}\n</svg>\n'.format(self.w, self.h, body))


def legend(svg, x, y, entries):
    """entries: [(label, color, kind)] kind in {'line', 'dash', 'box'}."""
    for label, color, kind in entries:
        if kind == "box":
            svg.rect(x, y - 9, 12, 12, color, rx=2)
        elif kind == "dash":
            svg.line(x, y - 3, x + 18, y - 3, color, 2, dash="5 4")
        else:
            svg.line(x, y - 3, x + 18, y - 3, color, 2)
            svg.marker(x + 9, y - 3, color)
        svg.text(x + 24, y + 1, label, 12, TEXT_2)
        x += 24 + 7.0 * len(label) + 22


def spread_labels(items, min_gap=15):
    """items: [(y, label, color)] -> same, nudged apart vertically."""
    items = sorted(items, key=lambda t: t[0])
    for i in range(1, len(items)):
        if items[i][0] - items[i - 1][0] < min_gap:
            items[i] = (items[i - 1][0] + min_gap,) + items[i][1:]
    return items


def log2_axis_chart(svg, box, x_values, y_max, y_step, y_fmt, x_label, y_label):
    left, top, right, bottom = box
    lx = [math.log2(v) for v in x_values]
    x0, x1 = min(lx), max(lx)

    def X(v):
        return left + (math.log2(v) - x0) / (x1 - x0) * (right - left)

    def Y(v):
        return bottom - v / y_max * (bottom - top)

    v = 0.0
    while v <= y_max + 1e-9:
        svg.line(left, Y(v), right, Y(v), GRID if v > 0 else AXIS, 1)
        svg.text(left - 8, Y(v) + 4, y_fmt(v), 11, TEXT_2, "end")
        v += y_step
    for xv in x_values:
        svg.line(X(xv), bottom, X(xv), bottom + 4, AXIS, 1)
        svg.text(X(xv), bottom + 18, str(xv), 11, TEXT_2, "middle")
    svg.text((left + right) / 2, bottom + 38, x_label, 12, TEXT_2, "middle")
    svg.text(left - 46, top - 14, y_label, 12, TEXT_2, "start")
    return X, Y


def strong_figures(review, out):
    rows = read_rows(review / "strong_scaling.csv")
    base = next(r for r in rows if int(r["cores"]) == 1)
    series = {"threads": [], "mpi": [], "hybrid": []}
    for r in rows:
        kind = layout_of(int(r["ranks"]), int(r["threads"]))
        point = (int(r["cores"]), float(r["speedup"]), float(r["efficiency"]), r["dims"], int(r["ranks"]),
                 int(r["threads"]))
        if kind == "base":
            series["threads"].append(point)
            series["mpi"].append(point)
        else:
            series[kind].append(point)
    for s in series.values():
        s.sort()
    cores = sorted({int(r["cores"]) for r in rows})
    n = int(base["particles"])

    for figure in ("speedup", "efficiency"):
        svg = Svg(720, 440)
        if figure == "speedup":
            svg.text(24, 32, "Strong scaling: speedup of the time loop", 16, TEXT, weight="bold")
            svg.text(24, 52, "{:,} particles, 100 steps, one Vanda node (Xeon Gold 6438N); baseline {:.1f} s on 1 core"
                     .format(n, float(base["loop_s"])), 12, TEXT_2)
            X, Y = log2_axis_chart(svg, (80, 110, 520, 380), cores, 16, 4, lambda v: "{:.0f}×".format(v),
                                   "Cores (MPI ranks × threads per rank)", "Speedup")
            svg.polyline([(X(c), Y(c)) for c in cores], IDEAL, 2, dash="5 4")
            entries = [("Ideal", IDEAL, "dash")]
        else:
            svg.text(24, 32, "Strong scaling: parallel efficiency", 16, TEXT, weight="bold")
            svg.text(24, 52, "Speedup ÷ cores, same runs as the speedup figure", 12, TEXT_2)
            X, Y = log2_axis_chart(svg, (80, 110, 520, 380), cores, 100, 20, lambda v: "{:.0f}%".format(v),
                                   "Cores (MPI ranks × threads per rank)", "Efficiency")
            entries = []
        labels = []
        for kind in ("threads", "mpi", "hybrid"):
            pts = series[kind]
            color = SERIES[kind]
            value = (lambda p: p[1]) if figure == "speedup" else (lambda p: 100 * p[2])
            coords = [(X(p[0]), Y(value(p))) for p in pts]
            svg.polyline(coords, color, 2)
            for c in coords:
                svg.marker(c[0], c[1], color)
            last = pts[-1]
            value_text = "{:.1f}×".format(last[1]) if figure == "speedup" else "{:.0f}%".format(100 * last[2])
            labels.append((coords[-1][1], "{} {}×{}: {}".format(SHORT_NAME[kind], last[4], last[5], value_text),
                           color))
            entries.append((LAYOUT_NAME[kind], color, "line"))
        for y, text, color in spread_labels(labels):
            svg.line(530, y, 540, y, color, 2)
            svg.text(546, y + 4, text, 11, TEXT)
        legend(svg, 80, 82, entries)
        svg.save(out / "strong_{}.svg".format(figure))


def weak_figure(review, out):
    rows = read_rows(review / "weak_scaling.csv")
    series = {"mpi": [], "hybrid": []}
    for r in rows:
        kind = layout_of(int(r["ranks"]), int(r["threads"]))
        point = (int(r["cores"]), float(r["efficiency"]), int(r["ranks"]), int(r["threads"]), float(r["loop_s"]))
        if kind == "base":
            series["mpi"].append(point)
            base_point = point
        elif kind in series:
            series[kind].append(point)
    for s in series.values():
        s.sort()
    cores = sorted({int(r["cores"]) for r in rows})
    base = base_point
    svg = Svg(720, 440)
    svg.text(24, 32, "Weak scaling: 1 million particles per core", 16, TEXT, weight="bold")
    svg.text(24, 52, "Efficiency = loop time on 1 core ({:.1f} s) ÷ loop time on N cores with N× the particles"
             .format(base[4]), 12, TEXT_2)
    X, Y = log2_axis_chart(svg, (80, 110, 520, 380), cores, 100, 20, lambda v: "{:.0f}%".format(v),
                           "Cores (MPI ranks × threads per rank)", "Efficiency")
    labels, entries = [], []
    for kind in ("mpi", "hybrid"):
        pts = series[kind]
        coords = [(X(p[0]), Y(100 * p[1])) for p in pts]
        svg.polyline(coords, SERIES[kind], 2)
        for c in coords:
            svg.marker(c[0], c[1], SERIES[kind])
        last = pts[-1]
        labels.append((coords[-1][1], "{} {}×{}: {:.0f}%".format(SHORT_NAME[kind], last[2], last[3], 100 * last[1]),
                       SERIES[kind]))
        entries.append((LAYOUT_NAME[kind], SERIES[kind], "line"))
    for y, text, color in spread_labels(labels):
        svg.line(530, y, 540, y, color, 2)
        svg.text(546, y + 4, text, 11, TEXT)
    legend(svg, 80, 82, entries)
    svg.save(out / "weak_efficiency.svg")


def mean_split(run_dir):
    rows = read_rows(run_dir / "timings.csv")
    n = len(rows)
    return {key: sum(float(r[key + "_seconds"]) for r in rows) / n for key in ("stepping", "wait", "migration")}


def time_split_figure(review, out):
    runs = review / "runs"
    groups = [("Balanced cloud (spread)", "strong_c16"), ("Compact plume", "plume_c16")]
    layouts = [("threads", "r1_t16", "1×16"), ("hybrid", "r4_t4", "4×4"), ("mpi", "r16_t1", "16×1")]
    bars = []
    for title, prefix in groups:
        for kind, suffix, dims in layouts:
            split = mean_split(runs / "{}_{}".format(prefix, suffix))
            bars.append((title, "{} {}".format(LAYOUT_NAME[kind].split(" (")[0], dims), split))
    x_max = max(sum(b[2].values()) for b in bars)
    x_max = math.ceil(x_max / 10.0) * 10
    svg = Svg(760, 470)
    svg.text(24, 32, "Where the time goes on 16 cores: balanced vs compact plume", 16, TEXT, weight="bold")
    svg.text(24, 52, "Time loop per rank (average over ranks), 10 million particles, 100 steps", 12, TEXT_2)
    legend(svg, 190, 82, [(label, color, "box") for _, label, color in SEGMENTS])
    left, right, top = 190, 640, 110
    bar_h, gap = 22, 12

    def X(v):
        return left + v / x_max * (right - left)

    y = top
    current = None
    for title, label, split in bars:
        if title != current:
            if current is not None:
                y += 16
            svg.text(24, y + 12, title, 12, TEXT, weight="bold")
            y += 20
            current = title
        svg.text(left - 10, y + bar_h / 2 + 4, label, 12, TEXT_2, "end")
        x = left
        total = sum(split.values())
        visible = [(key, color) for key, _, color in SEGMENTS if split[key] > 0]
        for i, (key, color) in enumerate(visible):
            width = X(split[key]) - left
            seg_w = max(width - (2 if i < len(visible) - 1 else 0), 0)  # 2 px surface gap between segments
            if i == len(visible) - 1 and seg_w > 4:
                # Rounded data end (right), square at the baseline side.
                r = 4
                d = ("M{:.1f},{:.1f} h{:.1f} a{r},{r} 0 0 1 {r},{r} v{:.1f} a{r},{r} 0 0 1 -{r},{r} h-{:.1f} z"
                     .format(x, y, seg_w - r, bar_h - 2 * r, seg_w - r, r=r))
                svg.path(d, color)
            else:
                svg.rect(x, y, seg_w, bar_h, color)
            x += width
        svg.text(X(total) + 8, y + bar_h / 2 + 4, "{:.1f} s".format(total), 12, TEXT)
        y += bar_h + gap
    # Grid behind the bars (inserted first so bars paint over it), labels below.
    grid = []
    for v in range(0, int(x_max) + 1, 10):
        grid.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="{3}" stroke-width="1"/>'
                    .format(X(v), top, y - gap + 4, GRID if v else AXIS))
        svg.text(X(v), y + 10, "{} s".format(v), 11, TEXT_2, "middle")
    svg.parts[1:1] = grid
    svg.h = int(y + 30)
    svg.parts[0] = '<rect width="{}" height="{}" fill="{}"/>'.format(svg.w, svg.h, SURFACE)
    svg.save(out / "time_split.svg")


def plume_imbalance_figure(review, out):
    runs = review / "runs"
    svg = Svg(720, 440)
    svg.text(24, 32, "Compact plume: load imbalance over time (16 cores)", 16, TEXT, weight="bold")
    svg.text(24, 52, "Most-loaded rank ÷ average rank (active particles); 1 = perfectly balanced", 12, TEXT_2)
    left, top, right, bottom = 80, 110, 520, 380
    series = []
    for kind, suffix in (("hybrid", "r4_t4"), ("mpi", "r16_t1")):
        rows = read_rows(runs / "plume_c16_{}".format(suffix) / "rank_counts.csv")
        by_time = {}
        for r in rows:
            by_time.setdefault(float(r["time"]), []).append(int(r["active_particles"]))
        pts = sorted((t, max(v) / (sum(v) / len(v))) for t, v in by_time.items())
        series.append((kind, suffix, pts, len(next(iter(by_time.values())))))
    t_max = max(p[0] for s in series for p in s[2])
    y_max = 16

    def X(t):
        return left + t / t_max * (right - left)

    def Y(v):
        return bottom - v / y_max * (bottom - top)

    for v in range(0, y_max + 1, 4):
        svg.line(left, Y(v), right, Y(v), GRID if v else AXIS, 1)
        svg.text(left - 8, Y(v) + 4, "{}×".format(v), 11, TEXT_2, "end")
    for t in sorted({p[0] for s in series for p in s[2]}):
        svg.line(X(t), bottom, X(t), bottom + 4, AXIS, 1)
        svg.text(X(t), bottom + 18, "{:g}".format(t), 11, TEXT_2, "middle")
    svg.text((left + right) / 2, bottom + 38, "Simulated time (s)", 12, TEXT_2, "middle")
    svg.text(left - 46, top - 14, "Imbalance", 12, TEXT_2)
    svg.line(left, Y(1), right, Y(1), IDEAL, 2, dash="5 4")
    labels, entries = [], []
    for kind, suffix, pts, ranks in series:
        coords = [(X(t), Y(v)) for t, v in pts]
        svg.polyline(coords, SERIES[kind], 2)
        for c in coords:
            svg.marker(c[0], c[1], SERIES[kind])
        labels.append((coords[0][1], "{}: starts at {:.1f}× (max {}×)".format(
            LAYOUT_NAME[kind].split(" (")[0] + " " + suffix.replace("r", "").replace("_t", "×"), pts[0][1], ranks),
            SERIES[kind]))
        entries.append((LAYOUT_NAME[kind], SERIES[kind], "line"))
    entries.append(("Balanced", IDEAL, "dash"))
    for y, text, color in spread_labels(labels):
        svg.text(left + 12, y - 8, text, 11, TEXT)
    legend(svg, 80, 82, entries)
    svg.save(out / "plume_imbalance.svg")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    review, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    strong_figures(review, out)
    weak_figure(review, out)
    time_split_figure(review, out)
    plume_imbalance_figure(review, out)
    print("wrote", ", ".join(sorted(p.name for p in out.glob("*.svg"))))


if __name__ == "__main__":
    main()
