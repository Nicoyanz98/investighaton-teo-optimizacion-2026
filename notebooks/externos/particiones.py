"""Patrones de partición de C4 (sección 4.2): el mejor μ de cada patrón, con Nelder–Mead afuera del SDP.

Completo: 12 patrones con h = 0.02 (arranques estilo S7 y al azar) y S7 contra (3,3,3,1) con h = 0.01
y 0.005, 8 reinicios de hasta 1000 iteraciones. Reducido: 4 patrones con h = 0.02 y 2 con h = 0.01, un
solo arranque de 300 iteraciones. Uso: python externos/particiones.py [--reducido] [--procesos N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biblioteca_s7 import *  # noqa: E402,F403


def trabajos_completos():
    """(base, piezas, h, reinicios, iteraciones, semilla, arranque) de la exploración completa."""
    trabajos = []
    for arranque in ["estructurado", "aleatorio"]:
        for i, piezas in enumerate(PATRONES_C4):
            trabajos.append(("c4", piezas, 0.02, 8, 1000, 1000 + i, arranque))
    for h in [0.01, 0.005]:
        for i, piezas in enumerate([(2, 2, 2, 1), (3, 3, 3, 1)]):
            trabajos.append(("c4", piezas, h, 8, 1000, 1000 + i, "estructurado"))
    return trabajos


def trabajos_reducidos():
    """Versión chica: pocos patrones, un arranque estilo S7 y 300 iteraciones."""
    trabajos = []
    for i, piezas in enumerate([(2, 2, 2, 1), (3, 3, 3, 1), (2, 2, 2, 2), (2, 2, 1, 1)]):
        trabajos.append(("c4", piezas, 0.02, 1, 300, 1000 + i, "estructurado"))
    for i, piezas in enumerate([(2, 2, 2, 1), (3, 3, 3, 1)]):
        trabajos.append(("c4", piezas, 0.01, 1, 300, 1000 + i, "estructurado"))
    return trabajos


def main():
    """Corre los patrones en paralelo y guarda resultados/particiones[_reducido].json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reducido", action="store_true")
    parser.add_argument("--procesos", type=int, default=8)
    args = parser.parse_args()
    trabajos = trabajos_reducidos() if args.reducido else trabajos_completos()
    resultados = []
    for r in correr_en_paralelo(explorar_patron, trabajos, args.procesos):
        print(f"h = {r['h']}  {r['arranque']:12s} {tuple(r['piezas'])}: μ = {r['mejor_mu']:.5f}"
              f"   ({r['segundos']:.0f} s)", flush=True)
        resultados.append(r)
    resultados.sort(key=lambda r: (r["h"], r["arranque"], -r["mejor_mu"]))
    nombre = "particiones_reducido.json" if args.reducido else "particiones.json"
    print("guardado en", guardar_json(nombre, resultados))


if __name__ == "__main__":
    main()
