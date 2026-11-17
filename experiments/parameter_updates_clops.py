"""
CLOPS for the parameter-updates experiment in ParameterUpdatesTest.py.

Uses train() from ParameterUpdatesTest.py (repo root) unchanged and times it. Each update runs
8 circuits (4 parameters x 2 parameter-shift evaluations), and each circuit is
one 2-qubit layer, so

    CLOPS = (8 * updates) circuits * shots * 1 layer / time
"""
import random
import sys
import time
import numpy as np
import pandas as pd
from paths import DATA, ROOT

sys.path.insert(0, str(ROOT))
import ParameterUpdatesTest as pu

UPDATE_COUNTS = [5, 10, 20, 40, 80]
SHOT_COUNTS = [100, 500, 1000, 5000]
REPEATS = 3
LAYERS = 1
CIRCUITS_PER_UPDATE = 8

rows = []
for shots in SHOT_COUNTS:
    pu.SHOTS = shots
    for n in UPDATE_COUNTS:
        times = []
        for r in range(REPEATS):
            random.seed(1000 * n + r)
            start = time.perf_counter()
            pu.train(n)
            times.append(time.perf_counter() - start)
        elapsed = float(np.median(times))
        clops = CIRCUITS_PER_UPDATE * n * shots * LAYERS / elapsed
        rows.append({"updates": n, "shots": shots, "time_s": elapsed, "clops": clops})
        print(f"shots {shots:>5} | updates {n:>3} | time {elapsed:6.2f} s | CLOPS {clops:,.0f}",
              flush=True)

pd.DataFrame(rows).to_csv(DATA / "parameter_updates_results.csv", index=False)
print("Saved parameter_updates_results.csv")
