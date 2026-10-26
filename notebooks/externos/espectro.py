"""Espectro completo por bloques de Fourier (sección 10.4), comparado con las cotas del certificado.

Uso: python externos/espectro.py [--archivo grafo_simetrico_s7.json]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biblioteca_s7 import *  # noqa: E402,F403


def main():
    """Calcula el espectro, lo compara con el certificado y guarda resultados/espectro_s7_n<n>.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archivo", default="grafos_s7.json")
    parser.add_argument("--lote", type=int, default=4096, help="frecuencias por lote")
    parser.add_argument("--puntos", type=int, default=2000, help="puntos de la curva guardada")
    args = parser.parse_args()

    fila = elegir_grafo(cargar_json(args.archivo))
    descripcion = fila["grafo"]
    certificado = fila["certificado"]
    inicio = time.time()
    grafo = GrafoImplicito(descripcion, Arreglos())
    n = grafo.n
    print(f"grafo de {args.archivo}: n = {n}, k = {grafo.k}, g = {grafo.g}", flush=True)

    error = chequear_producto(descripcion, grafo, [0, 1], args.lote)
    print(f"1. H·x por Fourier contra el implícito: error relativo {error:.1e}", flush=True)

    espectro = espectro_fourier(descripcion, lote=args.lote, informar=informar_espectro(time.time(), args.lote))
    cluster = espectro["cluster"]
    formula = np.linalg.eigvalsh(cociente_formula(descripcion))
    rotacion = int(np.argmin(np.abs(cluster)))
    menor_cluster = float(np.delete(cluster, rotacion).min())
    menor_cero = float(espectro["cero"].min())
    posicion = int(np.argmin(espectro["minimos"]))
    menor_fourier = float(espectro["minimos"][posicion])
    cota_cero = n * certificado["margen cero / n (cota inf)"]
    cota_fourier = n * certificado["margen fourier / n (cota inf)"]
    # la cota de m = 0 puede ser exacta (cluster sin acople), así que se compara con tolerancia de redondeo
    tolerancia = 1e-12
    respeta_cotas = (menor_cero >= cota_cero * (1 - tolerancia)
                     and menor_fourier >= cota_fourier * (1 - tolerancia))
    estable = menor_cluster > 0 and menor_cero > 0 and menor_fourier > 0
    abscisas, valores = curva_minimos(espectro["frecuencias"], espectro["minimos"], grafo.g, args.puntos)
    salida = {
        "archivo": args.archivo, "n": n, "g": grafo.g, "mu_exacto": certificado["mu_exacto"],
        "error_producto": error, "acople_cluster": espectro["acople"],
        "autovalores_cluster": cluster.tolist(), "autovalores_Q_grados": formula.tolist(),
        "diferencia_Q": float(np.abs(cluster - formula).max()),
        "menor_cluster": menor_cluster, "autovalores_cero": espectro["cero"].tolist(),
        "menor_cero": menor_cero, "cota_cero": cota_cero,
        "menor_fourier": menor_fourier, "m_menor_fourier": int(espectro["frecuencias"][posicion]),
        "cota_fourier": cota_fourier,
        "maximo": espectro["maximo"], "estable": bool(estable), "respeta_cotas": bool(respeta_cotas),
        "curva_m_sobre_g": abscisas, "curva_menor_autovalor": valores,
        "segundos": time.time() - inicio,
    }
    print(f"λ_min = {min(menor_cluster, menor_cero, menor_fourier):.6e}, respeta las cotas: {respeta_cotas}"
          f"   ({time.time() - inicio:.0f} s)", flush=True)
    print("guardado en", guardar_json(f"espectro_s7_n{n}.json", salida))


if __name__ == "__main__":
    main()
