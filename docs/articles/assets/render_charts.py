"""Render the articles' charts (standard library only).

sme2-bstride.csv holds the FP32 medians published in docs/benchmarks.md,
section "SME: B row stride": `bench_gemm --benchmark_filter=BStride/f32`,
Apple M4 Max, median of 3 interleaved repetitions. The strip-height chart
reads the raw samples in docs/experiments/sme-b-stride/height.csv, and the
ladder chart sme2-ladder.csv (docs/benchmarks.md, "The four SME steps").

    python3 render_charts.py    # rewrites every SVG file in this directory
"""
import csv
import statistics
from collections import defaultdict
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLORS = {"in_place_gflops": "#245ea8", "packed_gflops": "#be4d14"}
LEGEND = {"in_place_gflops": "B read in place", "packed_gflops": "B packed"}
Y_MAX = 1800


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


def bars(path, title, subtitle, rows, unit, x_max, tick, references=(), colors=None, left=250, ratio=False):
    """Horizontal bars. rows: (label, value, group); colors: {group: color};
    references: (value, label) drawn as dashed vertical lines."""
    colors = colors or {}
    right, top, row_h = 930, 100, 32
    height = top + row_h * len(rows) + 90
    scale = (right - left) / x_max
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" '
           f'viewBox="0 0 1000 {height}" role="img">',
           f"<title>{escape(title)}</title>",
           f'<rect width="1000" height="{height}" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#172635">',
           text(500, 34, title, 23), text(500, 61, subtitle, 14)]
    bottom = top + row_h * len(rows)
    for t in range(0, int(x_max) + 1, tick):
        x = left + t * scale
        out.append(f'<path d="M {x} {top - 8} V {bottom}" stroke="#dce2e8"/>')
        out.append(text(x, bottom + 20, t, 12))
    for i, (label, value, group) in enumerate(rows):
        y = top + i * row_h
        color = colors.get(group, "#245ea8")
        out.append(text(left - 12, y + 20, label, 14, "end"))
        out.append(f'<rect x="{left}" y="{y + 6}" width="{value * scale:.1f}" height="{row_h - 12}" fill="{color}"/>')
        out.append(text(left + value * scale + 8, y + 21, f"{value:.2f}×" if ratio else f"{value:,.1f}" if value < 100 else f"{value:,.0f}", 13, "start"))
    for value, label in references:
        x = left + value * scale
        out.append(f'<path d="M {x} {top - 8} V {bottom}" stroke="#7a8794" stroke-dasharray="6 5"/>')
        out.append(text(x - 6, top - 14, label, 12, "end"))
    out.append(text((left + right) / 2, bottom + 45, unit, 14))
    out.append("</g></svg>")
    (HERE / path).write_text("\n".join(out) + "\n")


def load_bars(name):
    with (HERE / name).open() as f:
        return [(r["label"], float(r["value"]), r["group"]) for r in csv.DictReader(f)]


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
          y_max=2000, reference=(1963, "FMOPA peak, 1,963"))
    bars("cuda-ladder.svg", "Nine CUDA GEMM levels on an RTX 5080",
         "N = 4096 · GPU time of the GEMM alone · levels 5–8 and cuBLAS FP16 multiply in fp16",
         load_bars("cuda-ladder.csv"), "TFLOP/s", 130, 20,
         references=[(59.7, "FP32 peak 59.7"), (124.4, "Tensor Core peak 124.4")],
         colors={"fma": "#245ea8", "tensor": "#be4d14", "ref": "#8c96a3"})
    simd = {"scalar": "#8c96a3", "simd": "#245ea8", "old": "#c9d3de"}
    bars("simd-zen5.svg", "FP32 GEMM on one Zen 5 core",
         "AMD Ryzen 9 9950X · N = 512 · GCC 13 · one thread",
         load_bars("simd-zen5.csv"), "GFLOP/s", 400, 50,
         references=[(182.3, "AVX2 peak 182"), (362, "AVX-512 peak 362")], colors=simd)
    bars("simd-m4.svg", "FP32 GEMM on one Apple M4 P-core",
         "Apple M4 Max · N = 512 · Apple Clang 17 · one thread",
         load_bars("simd-m4.csv"), "GFLOP/s", 140, 20,
         references=[(130, "NEON peak ~130")], colors=simd)
    bars("lowp-peaks.svg", "Peak multiply-add rate by input format",
         "Measured instruction throughput ÷ FP32 peak of the same unit (RTX 5080: FP32 FMA cores)",
         load_bars("lowp-peaks.csv"), "× the FP32 peak", 9, 1,
         references=[(1, "FP32 = 1×")], left=330, ratio=True,
         colors={"m4neon": "#8c96a3", "m4sme": "#2f9e6e", "zen5": "#245ea8", "gpu": "#be4d14"})


if __name__ == "__main__":
    main()
