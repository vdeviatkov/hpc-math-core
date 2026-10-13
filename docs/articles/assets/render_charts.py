"""Render the article's two FP32 charts from sme2-bstride.csv (standard library only).

sme2-bstride.csv holds the medians published in docs/benchmarks.md, section
"SME: B row stride": `bench_gemm --benchmark_filter=BStride`, Apple M4 Max,
median of 3 interleaved repetitions.

    python3 render_charts.py    # rewrites sme2-size-sweep.svg and sme2-padding.svg
"""
import csv
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLORS = {"in_place_gflops": "#245ea8", "packed_gflops": "#be4d14"}
LEGEND = {"in_place_gflops": "B read in place", "packed_gflops": "B packed"}
Y_MAX = 1600


def load(experiment):
    with (HERE / "sme2-bstride.csv").open() as f:
        return [r for r in csv.DictReader(f)
                if r["experiment"] == experiment and r["precision"] == "f32"]


def text(x, y, s, size=14, anchor="middle"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(s))}</text>'


def chart(path, title, panels):
    """panels: list of (subtitle, rows, x_key, x_label)."""
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520" viewBox="0 0 1000 520" role="img">',
           f"<title>{escape(title)}</title>",
           '<rect width="1000" height="520" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#172635">',
           text(500, 34, title, 23),
           text(500, 61, "Apple M4 Max · FP32 · single thread · packing time included · median of 3", 14)]
    width = 365 if len(panels) == 2 else 790
    top, bottom = 118, 410
    for p, (subtitle, rows, x_key, x_label) in enumerate(panels):
        left = 82 + p * 485 if len(panels) == 2 else 100
        out.append(text(left + width / 2, 99, subtitle, 17))
        for tick in range(0, Y_MAX + 1, 400):
            y = bottom - tick / Y_MAX * (bottom - top)
            out.append(f'<path d="M {left} {y} H {left + width}" stroke="#dce2e8"/>')
            out.append(text(left - 10, y + 5, tick, 12, "end"))
        xs = [left + i * width / (len(rows) - 1) for i in range(len(rows))]
        for x, r in zip(xs, rows):
            out.append(text(x, bottom + 23, r[x_key], 12))
        for key, color in COLORS.items():
            pts = [(x, bottom - int(r[key]) / Y_MAX * (bottom - top)) for x, r in zip(xs, rows)]
            out.append(f'<polyline points="{" ".join(f"{x},{y}" for x, y in pts)}" '
                       f'fill="none" stroke="{color}" stroke-width="3"/>')
            for (x, y), r in zip(pts, rows):
                out.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
                if len(panels) == 1:
                    out.append(text(x, y - 12 if key == "packed_gflops" else y + 21, r[key], 13))
        out.append(text(left + width / 2, 463, x_label, 14))
    out.append('<text transform="translate(23 270) rotate(-90)" text-anchor="middle" font-size="14">GFLOP/s</text>')
    for x, key in zip((270, 550), COLORS):
        out.append(f'<path d="M {x} 494 h 35" stroke="{COLORS[key]}" stroke-width="3"/>')
        out.append(text(x + 45, 499, LEGEND[key], 14, "start"))
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


if __name__ == "__main__":
    main()
