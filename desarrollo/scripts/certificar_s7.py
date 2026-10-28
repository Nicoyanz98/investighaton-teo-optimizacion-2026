"""Construye y certifica un miembro de la familia S7 (O'Keeffe, arXiv 2610.21271) de forma independiente.

Guarda data/s7_t<t>_h<h>.json con los datos racionales exactos (numeradores sobre D), el centro de
la caja de fases, su radio y el resultado de cada paso. Sale con código 0 si el certificado pasa.

Ejemplos:
  .venv/bin/python scripts/certificar_s7.py 1e-5 1e-6 42      # Proposición 1 del paper
  .venv/bin/python scripts/certificar_s7.py -0.01 1e-4 30     # control negativo (q > q*): falla
"""
import argparse
import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desarrollo.src.certificado import certificar, fases_newton, racionalizar  # noqa: E402
from desarrollo.src.experimentos import guardar_json  # noqa: E402
from desarrollo.src.s7 import NOMBRES, construir  # noqa: E402


def a_numeradores(fracciones, D):
    """Numeradores (como texto) de fracciones con denominador común D."""
    numeradores = []
    for fraccion in fracciones:
        numeradores.append(str(fraccion.numerator * (D // fraccion.denominator)))
    return numeradores


def main():
    """Construye S7 para (t, h), lo certifica y guarda los datos racionales."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("t", help="q = q* − t")
    parser.add_argument("h", help="separación de las mitades")
    parser.add_argument("log_d", type=int, help="D = 2·10^log_d")
    args = parser.parse_args()

    D = 2 * 10**args.log_d
    dps = 2 * args.log_d + 30
    radio = mp.mpf(10) ** (-(args.log_d + 12))
    S = construir(args.t, args.h, dps=dps)
    xq, Pq = racionalizar(S["x"], S["P"], D)
    phi = fases_newton(xq, Pq, S["phi"], dps=dps + 20)
    print(f"S7 con t = {args.t}, h = {args.h}, D = 2·10^{args.log_d}")
    resultado = certificar(xq, Pq, phi, radio=radio, dps=dps)

    pasos = {}
    for clave, valor in resultado.items():
        if clave != "mu":
            pasos[clave] = valor if isinstance(valor, bool) else mp.nstr(valor, 10)
    Q = []
    for fila in Pq:
        Q.append(a_numeradores(fila, D))
    datos = {
        "fuente": "Reconstrucción de S7 (O'Keeffe 2026, Tabla II + Apéndice A), certificado propio",
        "t": args.t,
        "h": args.h,
        "D": str(D),
        "clusters": NOMBRES,
        "X": a_numeradores(xq, D),
        "Q": Q,
        "centro_fases": [mp.nstr(p, dps) for p in phi],
        "radio_caja": mp.nstr(radio, 5),
        "mu_exacto": f"{resultado['mu'].numerator}/{resultado['mu'].denominator}",
        "mu": float(resultado["mu"]),
        "certificado": pasos,
    }
    print("guardado en", guardar_json(f"s7_t{args.t}_h{args.h}.json", datos))
    sys.exit(0 if resultado["ok"] else 1)


if __name__ == "__main__":
    main()
