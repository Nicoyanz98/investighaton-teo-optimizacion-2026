"""Explora patrones de partición de una base (C4 o Canale): ¿cuál da la plantilla más densa?

Para cada patrón (piezas por sector) y un h fijo corre varias optimizaciones (Nelder–Mead sobre
tamaños y desplazamientos, SDP interno en las densidades) y guarda el mejor μ en
data/particiones_<base>_h<h>_<arranque>[_<patrones>].json.

Con arranques al azar ni siquiera se reencuentra S7 (cuencas angostas, O'Keeffe sec. VI); por eso
el arranque por defecto es el estructurado (ansatz tipo S7 adaptado a cada patrón).

Ejemplos:
  .venv/bin/python scripts/explorar_particiones.py --h 0.02
  .venv/bin/python scripts/explorar_particiones.py --h 0.01 --patrones 2221,3331 --reinicios 6
  .venv/bin/python scripts/explorar_particiones.py --base canale --h 0.02 --patrones 21212121
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desarrollo.src.busqueda import (  # noqa: E402
    FamiliaParticiones, arranque_aleatorio, arranque_estructurado, base_c4, base_canale, optimizar_familia,
)
from desarrollo.src.experimentos import correr_en_paralelo, guardar_json  # noqa: E402

BASES = {"c4": base_c4, "canale": base_canale}

PATRONES_C4 = [
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


def explorar_patron(trabajo):
    """Corre los reinicios de un patrón y devuelve el mejor resultado."""
    nombre_base, piezas, h, reinicios, max_iteraciones, semilla_rng, arranque = trabajo
    familia = FamiliaParticiones(BASES[nombre_base](), piezas, h)
    rng = np.random.default_rng(semilla_rng)
    inicio = time.time()
    mus = []
    mejor_z = None
    for reinicio in range(reinicios):
        usar_estructurado = arranque == "estructurado" or reinicio % 2 == 0
        if usar_estructurado:
            z0 = arranque_estructurado(familia, rng, ruido=0.05 * reinicio)
        else:
            z0 = arranque_aleatorio(familia, rng)
        mu, z = optimizar_familia(familia, z0, max_iteraciones=max_iteraciones)
        mus.append(mu)
        if mu >= max(mus):
            mejor_z = z
    x, phi = familia.decodificar(mejor_z)
    return {
        "base": nombre_base,
        "piezas": list(piezas),
        "h": h,
        "mejor_mu": max(mus),
        "todos_mu": mus,
        "x": x.tolist(),
        "s": familia.desplazamientos(phi).tolist(),
        "segundos": time.time() - inicio,
    }


def leer_patrones(texto):
    """'2221,3331' → [(2, 2, 2, 1), (3, 3, 3, 1)]."""
    patrones = []
    for token in texto.split(","):
        patrones.append(tuple(int(caracter) for caracter in token))
    return patrones


def main():
    """Lee los argumentos, corre los patrones en paralelo y guarda los resultados."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", choices=sorted(BASES), default="c4")
    parser.add_argument("--h", type=float, required=True, help="separación de las piezas")
    parser.add_argument("--reinicios", type=int, default=8)
    parser.add_argument("--max-iteraciones", type=int, default=1000)
    parser.add_argument("--arranque", choices=["estructurado", "mixto"], default="estructurado",
                        help="mixto: la mitad de los reinicios al azar")
    parser.add_argument("--patrones", default=None, help="p. ej. 2221,3331 (por defecto, la lista para C4)")
    parser.add_argument("--procesos", type=int, default=6)
    args = parser.parse_args()

    patrones = leer_patrones(args.patrones) if args.patrones else PATRONES_C4
    trabajos = []
    for i, piezas in enumerate(patrones):
        trabajos.append((args.base, piezas, args.h, args.reinicios, args.max_iteraciones, 1000 + i, args.arranque))

    resultados = []
    for resultado in correr_en_paralelo(explorar_patron, trabajos, args.procesos):
        print(f"{tuple(resultado['piezas'])}: μ = {resultado['mejor_mu']:.5f}   ({resultado['segundos']:.0f}s)", flush=True)
        resultados.append(resultado)
    resultados.sort(key=lambda r: -r["mejor_mu"])

    sufijo = "_" + args.patrones.replace(",", "-") if args.patrones else ""
    nombre = f"particiones_{args.base}_h{args.h}_{args.arranque}{sufijo}.json"
    print("guardado en", guardar_json(nombre, resultados))


if __name__ == "__main__":
    main()
