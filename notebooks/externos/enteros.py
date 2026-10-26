"""Fase 1 (sección 7.6): redondeo ingenuo y por reticulado para 3 plantillas y una grilla de n.

Plantillas h ∈ {0.03, 0.02, 0.01} con ε = 1e-3 y tamaños (3, 5), leídas de la curva de la sección 6
(s7_curva.json si está, si no s7_curva_reducida.json). n ∈ {1e6, 3e6, …, 1e9}.
Uso: python externos/enteros.py [--procesos N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biblioteca_s7 import *  # noqa: E402,F403


def main():
    """Corre la grilla (plantilla, n) en paralelo y guarda resultados/s7_enteros.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", default="1e6,3e6,1e7,3e7,1e8,3e8,1e9")
    parser.add_argument("--procesos", type=int, default=8)
    args = parser.parse_args()
    archivo = "s7_curva.json" if existe_resultado("s7_curva.json") else "s7_curva_reducida.json"
    curva = cargar_json(archivo)
    trabajos = []
    for h in [0.03, 0.02, 0.01]:
        fila = plantilla_de_la_curva(curva, h, 1e-3, (3, 5))
        for n in [float(v) for v in args.ns.split(",")]:
            trabajos.append((fila, n))
    resultados = []
    for r in correr_en_paralelo(un_n, trabajos, args.procesos):
        print(f"h = {r['h']}  n = {r['n']:.1e}:  reticulado {r['reticulado_ok']}"
              f" (λ = {r['menor_autovalor']:.2e}, μ = {r['mu']:.6f})  ingenuo {r['ingenuo_ok']}"
              f"  ({r['segundos']:.0f} s)", flush=True)
        resultados.append(r)
    resultados.sort(key=lambda r: (r["h"], r["margen"], r["n"]))
    print("guardado en", guardar_json("s7_enteros.json", resultados))


if __name__ == "__main__":
    main()
