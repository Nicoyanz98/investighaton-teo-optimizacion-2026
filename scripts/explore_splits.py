"""Exploración: ¿qué patrón de partición de los 4 clusters de C4 da la plantilla más densa?

Para cada patrón pieces = (k0, k1, k2, k3) (cuántas piezas tiene cada cuarto de vuelta) y un h fijo,
corre varias optimizaciones (Nelder–Mead sobre tamaños y desplazamientos, SDP interno en las
densidades) y guarda el mejor μ en data/explore_splits_h{h}.json.

Uso:  .venv/bin/python scripts/explore_splits.py 0.02 [restarts] [maxiter] [s7|mixed] [patrones, p.ej. 2221,3331]

Con arranques al azar ni siquiera se reencuentra S7 (cuencas angostas, O'Keeffe sec. VI), así que
el modo por defecto arranca desde un estado "estilo S7" adaptado a cada patrón.
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.search import SplitFamily, optimize_family, random_start, s7_like_start  # noqa: E402

PATTERNS = [
    (2, 2, 2, 1),  # S7 (O'Keeffe)
    (2, 2, 1, 1),  # dos vecinos
    (2, 1, 2, 1),  # dos opuestos
    (2, 2, 2, 2),  # los cuatro
    (3, 2, 2, 1),
    (2, 3, 2, 1),
    (3, 3, 3, 1),
    (2, 2, 3, 2),
    (3, 3, 3, 3),
    (2, 1, 1, 1),
    (3, 1, 1, 1),
    (3, 3, 1, 1),
]


def run(args):
    pieces, h, restarts, maxiter, seed, mode = args
    warnings.filterwarnings("ignore")
    fam = SplitFamily(pieces, h)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    results = []
    for r in range(restarts):
        # mitad de los reinicios desde el arranque estilo S7 (con ruido creciente), mitad al azar
        z0 = s7_like_start(fam, rng, noise=0.05 * r) if (mode == "s7" or r % 2 == 0) else random_start(fam, rng)
        mu, z = optimize_family(fam, z0, maxiter=maxiter)
        results.append(mu)
        if mu >= max(results):
            zbest = z
    x, phi = fam.decode(zbest)
    return {
        "pieces": list(pieces), "h": h, "best_mu": max(results), "all_mu": results,
        "x": x.tolist(), "s": ((phi - fam.quarter * np.pi / 2) / h).tolist(),
        "seconds": time.time() - t0,
    }


if __name__ == "__main__":
    h = float(sys.argv[1])
    restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    maxiter = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    mode = sys.argv[4] if len(sys.argv) > 4 else "s7"   # "s7": todos estilo S7; "mixed": mitad al azar
    patterns = PATTERNS
    if len(sys.argv) > 5:  # p.ej. "2221,3331" para correr solo esos patrones
        patterns = [tuple(int(c) for c in tok) for tok in sys.argv[5].split(",")]
    jobs = [(p, h, restarts, maxiter, 1000 + i, mode) for i, p in enumerate(patterns)]
    with Pool(min(6, len(jobs))) as pool:
        out = []
        for r in pool.imap_unordered(run, jobs):
            print(f"{tuple(r['pieces'])}: μ = {r['best_mu']:.5f}   ({r['seconds']:.0f}s)", flush=True)
            out.append(r)
    out.sort(key=lambda r: -r["best_mu"])
    path = os.path.join(os.path.dirname(__file__), "..", "data", f"explore_splits_h{h}_{mode}" + ("_" + sys.argv[5].replace(",", "-") if len(sys.argv) > 5 else "") + ".json")
    json.dump(out, open(path, "w"), indent=1)
    print("guardado en", os.path.abspath(path))
