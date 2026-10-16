"""Review-only Mac comparison; see README.md for compilation and limits."""

import csv
import ctypes
import os
from pathlib import Path
import random
import statistics
import time


def main():
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    qos = ctypes.CDLL(None).pthread_set_qos_class_self_np
    qos.argtypes = [ctypes.c_uint, ctypes.c_int]
    if qos(0x21, 0) != 0:  # QOS_CLASS_USER_INTERACTIVE; not CPU affinity
        raise RuntimeError("could not request benchmark QoS")
    root = Path(__file__).resolve().parent
    libraries = [ctypes.CDLL(str(root / f"{phase}.dylib")) for phase in ("before", "after")]
    for library in libraries:
        library.setup.argtypes = library.step.argtypes = [ctypes.c_int, ctypes.c_int]
        library.setup.restype = library.step.restype = None
    names = ["Blocked", "NeonBlocked", "BlockedPf4", "NeonBlockedPf4", "Sme1Tile",
             "SmeTiles", "SmeBlocked", "Sme", "Amx"]
    rng = random.Random(20261016)
    configs = [(kernel, bits, n) for kernel in range(9) for bits in (32, 64)
               for n in (256, 1024)]
    configs += [(kernel, bits, 4096) for kernel in (6, 7, 8) for bits in (32, 64)]
    rng.shuffle(configs)
    with (root / "paired.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["kernel", "bits", "N", "repetitions", "before_ms", "after_ms",
                         "speedup", "median_paired_speedup", "paired_min", "paired_max"])
        for kernel, bits, n in configs:
            for library in libraries:
                library.setup(bits, n)
                library.step(kernel, bits)
            samples = [[], []]
            ratios = []
            clock = time.perf_counter_ns if kernel == 8 else time.process_time_ns
            for _ in range(9):
                order = [0, 1]
                rng.shuffle(order)
                for phase in order:
                    start = clock()
                    libraries[phase].step(kernel, bits)
                    elapsed = clock() - start
                    batch = max(1, min(100, int(20_000_000 / max(elapsed, 1))))
                    start = clock()
                    for _ in range(batch):
                        libraries[phase].step(kernel, bits)
                    samples[phase].append((clock() - start) / batch / 1e6)
                ratios.append(samples[0][-1] / samples[1][-1])
            before, after = map(statistics.median, samples)
            writer.writerow([names[kernel], bits, n, 9, before, after, before / after,
                             statistics.median(ratios), min(ratios), max(ratios)])
            output.flush()
            print(names[kernel], bits, n, round(before / after, 3), flush=True)


if __name__ == "__main__":
    main()
