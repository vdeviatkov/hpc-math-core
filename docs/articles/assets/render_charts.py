"""Render the article's two charts from sme2-bstride.csv (standard library only).

sme2-bstride.csv holds the FP32 medians published in docs/benchmarks.md,
section "SME: B row stride": `bench_gemm --benchmark_filter=BStride/f32`,
Apple M4 Max, median of 3 interleaved repetitions. The strip-height chart
reads the raw samples in docs/experiments/sme-b-stride/height.csv, and the
ladder chart sme2-ladder.csv (docs/benchmarks.md, "The four SME steps").

    python3 render_charts.py    # rewrites the four SVG files
"""
import csv
import statistics
from collections import defaultdict
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLORS = {"in_place_gflops": "#245ea8", "packed_gflops": "#be4d14"}
LEGEND = {"in_place_gflops": "B read in place", "packed_gflops": "B packed"}
Y_MAX = 1600


def load(experiment):
    with (HERE / "sme2-bstride.csv").open() as f:
        return [r for r in csv.DictReader(f) if r["experiment"] == experiment]


def load_height():
    """Median GFLOP/s per strip height, keyed like the bstride rows."""
    samples = defaultdict(list)
    path = HERE.parent.parent / "experiments" / "sme-b-stride" / "height.csv"
    with path.open() as f:
        for r in csv.DictReader(f):
            samples[(int(r["Kc"]), int(r["ldb"]))].append(float(r["gflops"]))
    rows = []
    for kc in sorted({k for k, _ in samples}):
        rows.append({"Kc": kc,
                     "in_place_gflops": round(statistics.median(samples[(kc, 4096)])),
                     "packed_gflops": round(statistics.median(samples[(kc, 4128)]))})
    return rows


def text(x, y, s, size=14, anchor="middle"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(s))}</text>'


def chart(path, title, panels, subtitle=None, series=None, y_max=Y_MAX, reference=None):
    """panels: list of (subtitle, rows, x_key, x_label).
    series: {csv column: (color, legend label)}; reference: (value, label) drawn dashed."""
    subtitle = subtitle or "Apple M4 Max · FP32 · single thread · packing time included · median of 3"
    series = series or {k: (COLORS[k], LEGEND[k]) for k in COLORS}
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520" viewBox="0 0 1000 520" role="img">',
           f"<title>{escape(title)}</title>",
           '<rect width="1000" height="520" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#172635">',
           text(500, 34, title, 23),
           text(500, 61, subtitle, 14)]
    width = 365 if len(panels) == 2 else 790
    top, bottom = 118, 410
    for p, (subtitle, rows, x_key, x_label) in enumerate(panels):
        left = 82 + p * 485 if len(panels) == 2 else 100
        out.append(text(left + width / 2, 99, subtitle, 17))
        for tick in range(0, y_max + 1, 400):
            y = bottom - tick / y_max * (bottom - top)
            out.append(f'<path d="M {left} {y} H {left + width}" stroke="#dce2e8"/>')
            out.append(text(left - 10, y + 5, tick, 12, "end"))
        if reference:
            y = bottom - reference[0] / y_max * (bottom - top)
            out.append(f'<path d="M {left} {y} H {left + width}" stroke="#7a8794" stroke-dasharray="6 5"/>')
            out.append(text(left + width, y - 8, reference[1], 13, "end"))
        xs = [left + i * width / (len(rows) - 1) for i in range(len(rows))]
        for x, r in zip(xs, rows):
            out.append(text(x, bottom + 23, r[x_key], 12))
        for key, (color, _) in series.items():
            pts = [(x, bottom - int(r[key]) / y_max * (bottom - top)) for x, r in zip(xs, rows)]
            out.append(f'<polyline points="{" ".join(f"{x},{y}" for x, y in pts)}" '
                       f'fill="none" stroke="{color}" stroke-width="3"/>')
            for (x, y), r in zip(pts, rows):
                out.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
                if len(panels) == 1 and len(series) == 2:
                    out.append(text(x, y - 12 if key == "packed_gflops" else y + 21, r[key], 13))
        out.append(text(left + width / 2, 463, x_label, 14))
    out.append('<text transform="translate(23 270) rotate(-90)" text-anchor="middle" font-size="14">GFLOP/s</text>')
    step = 280 if len(series) == 2 else 215
    first = 270 if len(series) == 2 else 500 - step * len(series) / 2 + 20
    for i, (color, label) in enumerate(series.values()):
        x = first + i * step
        out.append(f'<path d="M {x} 494 h 35" stroke="{color}" stroke-width="3"/>')
        out.append(text(x + 45, 499, label, 14, "start"))
    out.append("</g></svg>")
    (HERE / path).write_text("\n".join(out) + "\n")


def main():
    sweep = load("size_sweep")
    chart("sme2-size-sweep.svg", "The power-of-two cliff",
          [("N near 2048", sweep[:4], "N", "N = ldb (categories, not to scale)"),
           ("N near 4096", sweep[4:], "N", "N = ldb (categories, not to scale)")])
    chart("sme2-padding.svg", "Same 4096 × 4096 GEMM, different B row stride",
          [("Only B's leading dimension changes", load("padding"), "ldb",
            "ldb in elements (categories, not to scale)")])
    chart("sme2-strip-height.svg", "Where the 16 KiB stride breaks: rows per B strip",
          [("N = 1024, B read in place, macro-kernel only", load_height(), "Kc",
            "rows per B strip, Kc (categories, not to scale)")],
          subtitle="Apple M4 Max · FP32 · single thread · median of 9",
          series={"in_place_gflops": (COLORS["in_place_gflops"], "rows 16,384 B apart"),
                  "packed_gflops": (COLORS["packed_gflops"], "rows 16,512 B apart")},
          y_max=2000)
    with (HERE / "sme2-ladder.csv").open() as f:
        ladder = list(csv.DictReader(f))
    chart("sme2-ladder.svg", "Four SME2 kernels, one optimisation each",
          [("", ladder, "N", "N (square matrices; categories, not to scale)")],
          subtitle="Apple M4 Max · FP32 · single thread · median of 3",
          series={"gemm_sme_1tile": ("#8c96a3", "one ZA tile"),
                  "gemm_sme_tiles": ("#245ea8", "all four tiles"),
                  "gemm_sme_blocked": ("#2f9e6e", "+ cache blocking"),
                  "gemm_sme": ("#be4d14", "+ packed B")},
          y_max=2000, reference=(1977, "FMOPA peak, 1,977"))


if __name__ == "__main__":
    main()
