"""Construye y certifica un miembro de la familia S7 (O'Keeffe, arXiv 2610.21271) de forma independiente.

Uso:  .venv/bin/python scripts/certify_s7.py t h log10D
Ejemplo (Proposición 1 del paper):  scripts/certify_s7.py 1e-5 1e-6 42

Guarda data/s7_t{t}_h{h}.json con los datos racionales exactos (numeradores sobre D), el centro de la
caja de fases, su radio y el resultado de cada paso del certificado.
"""
import json
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.certify import certify, newton_phases, rationalize  # noqa: E402
from src.s7 import NAMES, build  # noqa: E402

if __name__ == "__main__":
    t, h, logD = sys.argv[1], sys.argv[2], int(sys.argv[3])
    D = 2 * 10**logD
    dps = 2 * logD + 30
    radius = mp.mpf(10) ** (-(logD + 12))
    S = build(t, h, dps=dps)
    xq, Pq = rationalize(S["x"], S["P"], D)
    phi = newton_phases(xq, Pq, S["phi"], dps=dps + 20)
    print(f"S7 con t = {t}, h = {h}, D = 2·10^{logD}")
    res = certify(xq, Pq, phi, radius=radius, dps=dps)
    data = {
        "source": "Reconstrucción de S7 (O'Keeffe 2026, Tabla II + Apéndice A), certificado propio",
        "t": t, "h": h, "D": str(D), "clusters": NAMES,
        "X": [str(v.numerator * (D // v.denominator)) for v in xq],
        "Q": [[str(v.numerator * (D // v.denominator)) for v in row] for row in Pq],
        "phi_center": [mp.nstr(p, dps) for p in phi],
        "box_radius": mp.nstr(radius, 5),
        "mu_exact": f"{res['mu'].numerator}/{res['mu'].denominator}",
        "mu_float": float(res["mu"]),
        "certificate": {k: (v if isinstance(v, bool) else mp.nstr(v, 10)) for k, v in res.items() if k != "mu"},
    }
    path = os.path.join(os.path.dirname(__file__), "..", "data", f"s7_t{t}_h{h}.json")
    json.dump(data, open(path, "w"), indent=1)
    print("guardado en", os.path.abspath(path))
    sys.exit(0 if res["ok"] else 1)
