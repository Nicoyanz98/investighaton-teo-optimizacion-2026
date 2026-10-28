"""Fase 1 del grafo explícito: grados enteros para plantillas S7 de la curva (data/s7_curva.json).

Para cada candidato (h, ε, k_mitad, k_W) y cada n de la grilla, compara el redondeo ingenuo de los
grados con el redondeo por reticulado (Newton entero sobre el residuo blando, src/enteros.py) y
guarda si el equilibrio exacto de los datos enteros es estricto, su μ exacto y los grados.

Salida: data/s7_enteros.json.

Ejemplos:
  .venv/bin/python scripts/s7_entero.py --procesos 8
  OMP_NUM_THREADS=1 python scripts/s7_entero.py --ns 1e6,3e6,1e7,3e7,1e8,3e8 --procesos 12   # server
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desarrollo.src.enteros import (  # noqa: E402
    evaluar_enteros, fases_s7, grados_ingenuos, mu_entero, pulir, redondeo_reticulado, tamanos_exactos,
)
from desarrollo.src.experimentos import cargar_json, correr_en_paralelo, guardar_json  # noqa: E402

CANDIDATOS = "0.03:1e-3:3:5,0.02:1e-3:3:5,0.01:1e-3:3:5"


def leer_candidatos(texto):
    """'h:ε:k_mitad:k_W,…' → lista de (h, ε, (k_mitad, k_W))."""
    candidatos = []
    for token in texto.split(","):
        h, margen, k_mitad, k_w = token.split(":")
        candidatos.append((float(h), float(margen), (int(k_mitad), int(k_w))))
    return candidatos


def plantilla_de_la_curva(curva, h, margen, tamanos):
    """Fila de data/s7_curva.json para (h, ε, tamaños)."""
    for fila in curva:
        if fila["h"] == h and fila["margen"] == margen and tuple(fila["k"]) == tamanos:
            return fila
    raise KeyError(f"no está en s7_curva.json: h = {h}, ε = {margen}, k = {tamanos}")


def un_n(trabajo):
    """Redondeo ingenuo y por reticulado para un candidato y un n."""
    fila, n = trabajo
    inicio = time.time()
    h = fila["h"]
    k_mitad, k_w = fila["k"]
    k = [k_mitad] * 6 + [k_w]
    g = int(round(n / sum(k)))
    phi = fases_s7(h, np.array(fila["s"]))
    x, phi, P = pulir(tamanos_exactos(k), phi, np.array(fila["P"]))
    ingenuo = evaluar_enteros(k, g, grados_ingenuos(x, P, k, g), phi, h)
    d, reticulado = redondeo_reticulado(x, P, phi, h, k, g)
    mu = mu_entero(k, g, d)
    return {
        "h": h, "margen": fila["margen"], "k": fila["k"], "mu_plantilla": fila["mu"],
        "n": g * sum(k), "g": g,
        "ingenuo_ok": ingenuo["ok"],
        "reticulado_ok": reticulado["ok"],
        "menor_autovalor": reticulado["menor autovalor"],
        "mu": float(mu), "mu_exacto": f"{mu.numerator}/{mu.denominator}",
        "grados": d, "fases": reticulado["phi"],
        "segundos": time.time() - inicio,
    }


def main():
    """Corre la grilla (candidato, n) en paralelo y guarda los resultados."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidatos", default=CANDIDATOS, help="h:ε:k_mitad:k_W separados por comas")
    parser.add_argument("--ns", default="1e6,3e6,1e7,3e7,1e8,3e8,1e9")
    parser.add_argument("--procesos", type=int, default=8)
    args = parser.parse_args()

    curva = cargar_json("s7_curva.json")
    trabajos = []
    for h, margen, tamanos in leer_candidatos(args.candidatos):
        fila = plantilla_de_la_curva(curva, h, margen, tamanos)
        for n in [float(v) for v in args.ns.split(",")]:
            trabajos.append((fila, n))

    resultados = []
    for r in correr_en_paralelo(un_n, trabajos, args.procesos):
        print(f"h = {r['h']}  ε = {r['margen']:.0e}  n = {r['n']:.1e}:  reticulado {r['reticulado_ok']}"
              f" (λ = {r['menor_autovalor']:.2e}, μ = {r['mu']:.6f})  ingenuo {r['ingenuo_ok']}", flush=True)
        resultados.append(r)
    resultados.sort(key=lambda r: (r["h"], r["margen"], r["n"]))
    print("guardado en", guardar_json("s7_enteros.json", resultados))


if __name__ == "__main__":
    main()
