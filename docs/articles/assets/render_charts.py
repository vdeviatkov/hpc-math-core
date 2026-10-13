"""Render standalone SVG figures from documented medians; no dependencies."""
import csv
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
SIZES = [
    (1984, 1565, 1517, 407, 418), (2016, 1556, 1541, 410, 419),
    (2048, 1196, 1432, 281, 407), (2080, 1493, 1483, 398, 404),
    (4000, 1365, 1406, 380, 409), (4032, 1354, 1399, 380, 408),
    (4064, 1303, 1399, 364, 411), (4096, 608, 1344, 268, 398),
    (4128, 1260, 1304, 362, 395),
]
PADDING = [(4096, 608, 1344, 268, 398), (4097, 1145, 1365, 367, 407),
           (4128, 1339, 1386, 376, 407), (4160, 1332, 1390, 376, 406)]


def label(x, y, value, size=14, anchor="middle"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" '
            f'text-anchor="{anchor}">{escape(str(value))}</text>')


def figure(name, title, panels, maximum):
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520" viewBox="0 0 1000 520" role="img">',
           f'<title>{escape(title)}</title>',
           '<desc>Apple M4 Max, FP32, documented medians of three interleaved repetitions. Packing cost included. No error bars: individual samples unavailable.</desc>',
           '<rect width="1000" height="520" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#172635">', label(500, 34, title, 23),
           label(500, 61, 'Apple M4 Max · FP32 · packing included · median of 3 repetitions', 14)]
    colors = ['#245ea8', '#be4d14']
    for p, (subtitle, data, axis) in enumerate(panels):
        left = 82 + p * 485 if len(panels) == 2 else 100
        width = 365 if len(panels) == 2 else 790
        top, bottom = 118, 410
        out.append(label(left + width / 2, 99, subtitle, 17))
        for tick in range(0, maximum + 1, 400):
            y = bottom - tick / maximum * (bottom - top)
            out.append(f'<path d="M {left} {y} H {left+width}" stroke="#dce2e8"/>')
            out.append(label(left - 10, y + 5, tick, 12, 'end'))
        for i, row in enumerate(data):
            x = left + i * width / (len(data) - 1)
            out.append(label(x, bottom + 23, row[0], 12))
        for series, color in enumerate(colors, 1):
            points = [(left + i * width / (len(data)-1),
                       bottom - row[series] / maximum * (bottom-top))
                      for i, row in enumerate(data)]
            out.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="3"/>')
            for (x, y), row in zip(points, data):
                out.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
                if len(panels) == 1:
                    out.append(label(x, y - 12 if series == 2 else y + 21, row[series], 13))
        out.append(label(left + width / 2, 463, axis, 14))
    out.append('<text transform="translate(23 270) rotate(-90)" text-anchor="middle" font-size="14">Throughput (GFLOP/s)</text>')
    for x, color, text in [(270, colors[0], 'Original B'), (550, colors[1], 'Packed B')]:
        out.append(f'<path d="M {x} 494 h 35" stroke="{color}" stroke-width="3"/>')
        out.append(label(x+45, 499, text, 14, 'start'))
    out.append('</g></svg>')
    (ROOT / name).write_text('\n'.join(out) + '\n')


def main():
    with (ROOT / 'sme2-bstride-documented.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'N', 'ldb', 'precision', 'stride_bytes',
                         'in_place_gflops', 'packed_gflops', 'provenance'])
        for experiment, data in [('size_sweep', SIZES), ('padding', PADDING)]:
            for ld, f32i, f32p, f64i, f64p in data:
                for precision, size, original, packed in [('f32', 4, f32i, f32p), ('f64', 8, f64i, f64p)]:
                    writer.writerow([experiment, ld if experiment == 'size_sweep' else 4096,
                                     ld, precision, ld*size, original, packed,
                                     'src/gemm/README.md: rounded documented median, not raw samples'])
    figure('sme2-size-sweep.svg', 'B layout and the power-of-two performance cliff',
           [('Near 2048', SIZES[:4], 'N (equally spaced categories)'),
            ('Near 4096', SIZES[4:], 'N (equally spaced categories)')], 1600)
    figure('sme2-padding.svg', 'Same 4096 × 4096 GEMM, different B row stride',
           [('Only the physical leading dimension of B changes', PADDING,
             'Physical leading dimension ldb (elements; equally spaced categories)')], 1600)


if __name__ == '__main__':
    main()
