"""Barrido de semillas (vía E): μ de borde (L ⪰ 0, sin brecha) de geometrías de clusters.

Prueba una red de ℓ clusters equiespaciados y la unión de dos redes de ℓ puntos rotadas β entre sí
(con ℓ = 4 es la geometría de Canale). Para cada una reporta el mejor μ de borde y cuántos modos
de cluster quedan neutros. Guarda data/semillas.json.

Ejemplo:  .venv/bin/python scripts/barrer_semillas.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desarrollo.src.busqueda import dos_redes, semilla, una_red  # noqa: E402
from desarrollo.src.experimentos import guardar_json, silenciar_avisos  # noqa: E402


def barrer_una_red(ls):
    """Semilla de cada red de ℓ clusters."""
    resultados = []
    for l in ls:
        mu, neutros = semilla(*una_red(l))
        resultados.append({"l": l, "mu": mu, "neutros": neutros})
    return resultados


def barrer_dos_redes(ls, rs, cantidad_betas):
    """Mejores semillas de dos redes rotadas, sobre una grilla de β y pesos r."""
    resultados = []
    for l in ls:
        betas = np.linspace(0.02, 2 * np.pi / l - 0.02, cantidad_betas)
        for r in rs:
            for beta in betas:
                mu, neutros = semilla(*dos_redes(l, r, beta))
                if mu is not None:
                    resultados.append({"l": l, "r": r, "beta_grados": float(np.degrees(beta)),
                                       "mu": mu, "neutros": neutros})
    resultados.sort(key=lambda fila: -fila["mu"])
    return resultados


def main():
    """Barre una red y dos redes, imprime los mejores y guarda data/semillas.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--betas", type=int, default=40, help="puntos de la grilla en β")
    args = parser.parse_args()
    silenciar_avisos()

    una = barrer_una_red(range(3, 13))
    print("una red (L ⪰ 0):")
    for fila in una:
        mu = "—" if fila["mu"] is None else f"{fila['mu']:.4f}"
        print(f"  ℓ = {fila['l']:2d}   μ = {mu}   modos neutros = {fila['neutros']}")

    dos = barrer_dos_redes([3, 4, 5, 6, 8], [0.3, 0.4, 0.5], args.betas)
    print("\ndos redes rotadas β (mejor por ℓ):")
    vistos = set()
    for fila in dos:
        if fila["l"] in vistos:
            continue
        vistos.add(fila["l"])
        print(f"  ℓ = {fila['l']}   μ = {fila['mu']:.4f}   neutros = {fila['neutros']}"
              f"   r = {fila['r']}   β = {fila['beta_grados']:.1f}°")

    print("\nguardado en", guardar_json("semillas.json", {"una_red": una, "dos_redes": dos}))


if __name__ == "__main__":
    main()
