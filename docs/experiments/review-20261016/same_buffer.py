"""Compare FP64 SME kernels on identical buffers; see README.md for limits."""

import csv
import ctypes
from pathlib import Path
import random
import statistics
import time


def main():
    root = Path(__file__).resolve().parent
    libraries = [ctypes.CDLL(str(root / f"{phase}.dylib")) for phase in ("before", "after")]
    for library in libraries:
        library.macro_step.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        library.macro_step.restype = None
    libc = ctypes.CDLL(None)
    libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                   ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    rng = random.Random(31)
    n = 1024
    buffers = []
    try:
        for _ in range(3):
            pointer = ctypes.c_void_p()
            if libc.posix_memalign(ctypes.byref(pointer), 128, n * n * 8) != 0:
                raise MemoryError("same-buffer probe allocation failed")
            buffers.append(pointer)
            ctypes.memset(pointer, 0, n * n * 8)
        with (root / "same-buffer.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["kernel", "bits", "N", "repetitions", "before_ms", "after_ms",
                             "speedup", "paired_min", "paired_max"])
            for single in (0, 1):
                samples = [[], []]
                ratios = []
                for _ in range(31):
                    order = [0, 1]
                    rng.shuffle(order)
                    for phase in order:
                        libraries[phase].macro_step(single, 64, *buffers, n)
                        start = time.process_time_ns()
                        libraries[phase].macro_step(single, 64, *buffers, n)
                        samples[phase].append((time.process_time_ns() - start) / 1e6)
                    ratios.append(samples[0][-1] / samples[1][-1])
                before, after = map(statistics.median, samples)
                name = "one_tile_kernel" if single else "macro_kernel"
                writer.writerow([name, 64, n, 31, before, after, before / after,
                                 min(ratios), max(ratios)])
                output.flush()
                print(name, before, after, before / after, flush=True)
    finally:
        for pointer in buffers:
            libc.free(pointer)


if __name__ == "__main__":
    main()
