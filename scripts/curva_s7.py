"""Fase 0 del grafo explícito: frontera entre μ y el margen de los modos de cluster en la familia S7.

Para cada h (separación de las mitades), cada margen ε del SDP (L/h² ⪰ ε) y cada par de tamaños
racionales (k_mitad, k_W), optimiza los desplazamientos s (Nelder–Mead) y las densidades P (SDP),
y guarda μ, los autovalores normalizados de L, min c, la condición del Jacobiano del equilibrio y
el menor n con el que el redondeo ingenuo de los grados sigue siendo estricto.

Salida: data/s7_curva.json.

Ejemplos:
  .venv/bin/python scripts/curva_s7.py --procesos 8                 # local
  OMP_NUM_THREADS=1 python scripts/curva_s7.py --procesos 12        # server (12 núcleos)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.enteros import (  # noqa: E402
    espectro_s7, fases_s7, n_minimo_ingenuo, optimizar_s7, pulir, tamanos_exactos, tamanos_racionales,
)
from src.experimentos import correr_en_paralelo, guardar_json  # noqa: E402

TAMANOS = [(3, 5), (11, 18), (20, 33)]


def un_punto(trabajo):
    """Optimiza S7 para (h, margen, tamaños) y mide su espectro y el redondeo ingenuo."""
    h, margen, (k_mitad, k_w), con_ingenuo = trabajo
    inicio = time.time()
    x = tamanos_racionales(k_mitad, k_w)
    mu, s, P = optimizar_s7(x, h, margen)
    fila = {"h": h, "margen": margen, "k": [k_mitad, k_w], "mu": mu}
    if mu is None:
        return fila
    phi = fases_s7(h, s)
    fila.update(espectro_s7(x, phi, P))
    fila["s"] = s.tolist()
    fila["P"] = np.asarray(P).tolist()
    if con_ingenuo:
        k = [k_mitad] * 6 + [k_w]
        x_exacto, phi_exacto, P_exacto = pulir(tamanos_exactos(k), phi, P)
        n_min, _ = n_minimo_ingenuo(x_exacto, P_exacto, phi_exacto, h, k, exponentes=np.arange(4, 16.01, 1))
        fila["n_min_ingenuo"] = n_min
    fila["segundos"] = time.time() - inicio
    return fila


def main():
    """Barre (h, margen, tamaños) en paralelo y guarda la curva."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hs", default="0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--margenes", default="1e-5,1e-4,3e-4,1e-3,2e-3,3e-3,5e-3,1e-2")
    parser.add_argument("--sin-ingenuo", action="store_true", help="no medir el n del redondeo ingenuo")
    parser.add_argument("--procesos", type=int, default=8)
    args = parser.parse_args()

    trabajos = []
    for h in [float(v) for v in args.hs.split(",")]:
        for margen in [float(v) for v in args.margenes.split(",")]:
            for tamanos in TAMANOS:
                trabajos.append((h, margen, tamanos, not args.sin_ingenuo))

    resultados = []
    for fila in correr_en_paralelo(un_punto, trabajos, args.procesos):
        if fila["mu"] is None:
            print(f"h = {fila['h']}  ε = {fila['margen']:.0e}  k = {fila['k']}: infactible", flush=True)
            continue
        print(f"h = {fila['h']}  ε = {fila['margen']:.0e}  k = {fila['k']}:  μ = {fila['mu']:.5f}"
              f"  λ_min = {fila['autovalores'][0]:.2e}  n ingenuo = {fila.get('n_min_ingenuo')}", flush=True)
        resultados.append(fila)
    resultados.sort(key=lambda fila: (fila["h"], fila["margen"], fila["k"]))
    print("guardado en", guardar_json("s7_curva.json", resultados))


if __name__ == "__main__":
    main()
