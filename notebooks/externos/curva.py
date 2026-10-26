"""Curva μ contra margen de S7 (sección 6.3): optimiza S7 para cada (h, ε, tamaños) en paralelo.

Completo: h ∈ {0.005, 0.01, 0.02, 0.03, 0.05}, 8 márgenes y 3 pares de tamaños (120 puntos).
Reducido: h ∈ {0.01, 0.02, 0.03}, ε ∈ {1e-4, 1e-3, 3e-3, 1e-2} y tamaños (3, 5) (12 puntos).
Con --con-ingenuo mide además el n mínimo del redondeo ingenuo (necesita la biblioteca de la sección 7).
Uso: python externos/curva.py [--reducido] [--con-ingenuo] [--procesos N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biblioteca_s7 import *  # noqa: E402,F403


def main():
    """Barre (h, margen, tamaños) en paralelo y guarda resultados/s7_curva[_reducida].json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reducido", action="store_true")
    parser.add_argument("--con-ingenuo", action="store_true")
    parser.add_argument("--procesos", type=int, default=8)
    args = parser.parse_args()
    if args.reducido:
        hs = [0.01, 0.02, 0.03]
        margenes = [1e-4, 1e-3, 3e-3, 1e-2]
        tamanos = [(3, 5)]
    else:
        hs = [0.005, 0.01, 0.02, 0.03, 0.05]
        margenes = [1e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
        tamanos = TAMANOS
    trabajos = []
    for h in hs:
        for margen in margenes:
            for par in tamanos:
                trabajos.append((h, margen, par, args.con_ingenuo))
    resultados = []
    for fila in correr_en_paralelo(un_punto, trabajos, args.procesos):
        if fila["mu"] is None:
            print(f"h = {fila['h']}  ε = {fila['margen']:.0e}  k = {fila['k']}: infactible", flush=True)
            continue
        print(f"h = {fila['h']}  ε = {fila['margen']:.0e}  k = {fila['k']}:  μ = {fila['mu']:.5f}"
              f"  λ_min = {fila['autovalores'][0]:.2e}  ({fila['segundos']:.0f} s)", flush=True)
        resultados.append(fila)
    resultados.sort(key=lambda fila: (fila["h"], fila["margen"], fila["k"]))
    nombre = "s7_curva_reducida.json" if args.reducido else "s7_curva.json"
    print("guardado en", guardar_json(nombre, resultados))


if __name__ == "__main__":
    main()
