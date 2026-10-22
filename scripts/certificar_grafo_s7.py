"""Fase 3 del grafo explícito: certificado riguroso de los blow-ups circulantes de data/s7_enteros.json.

Para cada solución entera estricta (reticulado_ok) arma el grafo de src/blowup.py y prueba, con
aritmética de intervalos, que tiene un equilibrio estricto no sincronizado:
  1. Cociente (modos constantes por cluster): `certificado.certificar` sobre x = n_a/n, P = d_ab/n_b
     (Krawczyk + c > 0 + Cholesky por intervalos de L + fases distintas). Devuelve la caja de fases.
  2. Resto de los modos (H_m con m ≠ 0, y la parte de H_0 ortogonal a los modos de cluster):
     margen de Gershgorin > 0 con los cos evaluados en intervalos sobre esa caja
     (`blowup.margenes_gershgorin`).
Todo eso lo hace `blowup.certificar_blowup`.
  3. μ = (δ + 1)/n exacto y > 11/16 en aritmética racional.
Con --control-negativo agrega el redondeo ingenuo del mismo n, que tiene que fallar.

Salida: data/grafos_s7.json (descripción compacta de cada grafo + resultado del certificado).

Ejemplo (tarda menos de un minuto, CPU):
  .venv/bin/python scripts/certificar_grafo_s7.py --control-negativo
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blowup import certificar_blowup, descripcion_compacta  # noqa: E402
from src.enteros import fases_s7, grados_ingenuos, pulir, tamanos_exactos  # noqa: E402
from src.experimentos import cargar_json, guardar_json  # noqa: E402


def control_ingenuo(r):
    """Grados del redondeo ingenuo para el mismo (h, ε, k, n) que la fila r de s7_enteros.json."""
    fila = [f for f in cargar_json("s7_curva.json") if f["h"] == r["h"] and f["margen"] == r["margen"]
            and f["k"] == r["k"]][0]
    k = [r["k"][0]] * 6 + [r["k"][1]]
    phi = fases_s7(r["h"], np.array(fila["s"]))
    x, phi, P = pulir(tamanos_exactos(k), phi, np.array(fila["P"]))
    return grados_ingenuos(x, P, k, r["g"]), [float(p) for p in phi]


def imprimir(etiqueta, resultado):
    """Una línea por grafo."""
    if "krawczyk" not in resultado:
        print(f"{etiqueta}: falla ({resultado['motivo']})", flush=True)
        return
    print(f"{etiqueta}: ok {resultado['ok']}  (Krawczyk {resultado['krawczyk']}, Cholesky {resultado['cholesky']},"
          f" pivote {resultado['min pivote L (cota inf)']:.2e}, margen Fourier {resultado['margen fourier / n (cota inf)']:.3f} n,"
          f" μ = {resultado['mu']:.6f})", flush=True)


def main():
    """Certifica cada solución entera estricta y guarda las descripciones compactas."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control-negativo", action="store_true")
    parser.add_argument("--dps", type=int, default=100)
    args = parser.parse_args()

    salida = []
    for r in sorted(cargar_json("s7_enteros.json"), key=lambda r: (r["n"], -r["h"])):
        if not r["reticulado_ok"]:
            continue
        k = [r["k"][0]] * 6 + [r["k"][1]]
        inicio = time.time()
        resultado = certificar_blowup(k, r["g"], r["grados"], r["fases"], dps=args.dps)
        etiqueta = f"h = {r['h']}  n = {r['n']:.2e}  reticulado"
        imprimir(etiqueta, resultado)
        fila = {"h": r["h"], "margen": r["margen"], "redondeo": "reticulado", "segundos": time.time() - inicio,
                "certificado": resultado, "grafo": descripcion_compacta(k, r["g"], r["grados"], resultado.get("fases", r["fases"]))}
        salida.append(fila)
        if args.control_negativo:
            d_ingenuo, phi = control_ingenuo(r)
            negativo = certificar_blowup(k, r["g"], d_ingenuo, phi, dps=args.dps)
            imprimir(f"h = {r['h']}  n = {r['n']:.2e}  ingenuo   ", negativo)
            salida.append({"h": r["h"], "margen": r["margen"], "redondeo": "ingenuo", "certificado": negativo,
                           "grafo": descripcion_compacta(k, r["g"], d_ingenuo, negativo.get("fases", phi))})
    print("guardado en", guardar_json("grafos_s7.json", salida))


if __name__ == "__main__":
    main()
