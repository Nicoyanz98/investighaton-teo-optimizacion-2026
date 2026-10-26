"""Fase 3 (sección 9.4): certifica cada solución estricta de la Fase 1 y su control ingenuo.

Guarda la descripción compacta de cada grafo con su certificado en resultados/grafos_s7.json.
Uso: python externos/certificar_grafos.py [--dps 100]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biblioteca_s7 import *  # noqa: E402,F403


def main():
    """Certifica cada solución entera estricta (y su control ingenuo) y guarda las descripciones compactas."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dps", type=int, default=100)
    args = parser.parse_args()
    curva = cargar_json("s7_curva.json" if existe_resultado("s7_curva.json") else "s7_curva_reducida.json")
    salida = []
    for r in sorted(cargar_json("s7_enteros.json"), key=lambda r: (r["n"], -r["h"])):
        if not r["reticulado_ok"]:
            continue
        k = [r["k"][0]] * 6 + [r["k"][1]]
        inicio = time.time()
        resultado = certificar_blowup(k, r["g"], r["grados"], r["fases"], dps=args.dps)
        imprimir(f"h = {r['h']}  n = {r['n']:.2e}  reticulado", resultado)
        fila = {"h": r["h"], "margen": r["margen"], "redondeo": "reticulado", "segundos": time.time() - inicio,
                "certificado": resultado,
                "grafo": descripcion_compacta(k, r["g"], r["grados"], resultado.get("fases", r["fases"]))}
        salida.append(fila)
        d_ingenuo, phi = control_ingenuo(r, curva)
        negativo = certificar_blowup(k, r["g"], d_ingenuo, phi, dps=args.dps)
        imprimir(f"h = {r['h']}  n = {r['n']:.2e}  ingenuo   ", negativo)
        salida.append({"h": r["h"], "margen": r["margen"], "redondeo": "ingenuo", "certificado": negativo,
                       "grafo": descripcion_compacta(k, r["g"], d_ingenuo, negativo.get("fases", phi))})
    print("guardado en", guardar_json("grafos_s7.json", salida))


if __name__ == "__main__":
    main()
