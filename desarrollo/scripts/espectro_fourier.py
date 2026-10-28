"""Espectro completo del Hessiano de un grafo certificado, por bloques de Fourier (src/espectro.py).

Lanczos (scripts/validar_grafo.py) da una cota superior del menor autovalor; acá se calculan todos:
H se parte en g bloques hermíticos H_m de K × K, uno por frecuencia, y se diagonalizan todos.
  1. Chequeo de la descomposición: H·x por bloques (FFT) contra el producto implícito de
     src/validacion.py, que solo usa la regla de adyacencia, en vectores al azar.
  2. Modos de cluster: autovalores de H_0 en la base de clusters, contra la Q de la fórmula de grados.
  3. Resto de m = 0 y m ≠ 0: el menor autovalor verdadero, contra la cota de Gershgorin certificada.
     Si el menor autovalor quedara debajo de la cota, el certificado tendría un error.

Salida: data/espectro_s7_n<n>.json (incluye el menor autovalor de cada m, reducido a ~2000 puntos).

Comandos (desde la raíz del repo; numpy en CPU, no hace falta GPU):
  python scripts/espectro_fourier.py --archivo grafo_simetrico_s7.json      (n = 299 989, segundos)
  python scripts/espectro_fourier.py                                        (n = 30 000 004, minutos)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desarrollo.src.espectro import curva_minimos, espectro_fourier, producto_fourier  # noqa: E402
from desarrollo.src.experimentos import cargar_json, guardar_json  # noqa: E402
from desarrollo.src.validacion import Arreglos, GrafoImplicito, cociente_formula, elegir_grafo  # noqa: E402


def chequear_producto(descripcion, grafo, semillas, lote):
    """Máximo error relativo ‖H x (Fourier) − H x (implícito)‖ / ‖H x‖ sobre vectores gaussianos."""
    errores = []
    for semilla in semillas:
        x = np.random.default_rng(semilla).standard_normal(grafo.n)
        implicito = grafo.arreglos.a_numpy(grafo.producto_hessiano(grafo.arreglos.desde(x)))
        fourier = producto_fourier(descripcion, x, lote=lote)
        errores.append(float(np.linalg.norm(fourier - implicito) / np.linalg.norm(implicito)))
    return max(errores)


def informar(inicio, lote):
    """Función que imprime el progreso de la diagonalización (cada 50 lotes)."""
    def imprimir(hechas, total):
        if hechas == total or (hechas // lote) % 50 == 0:
            print(f"  {hechas}/{total} frecuencias ({time.time() - inicio:.0f} s)", flush=True)
    return imprimir


def main():
    """Calcula el espectro, lo compara con el certificado y guarda el resultado."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archivo", default="grafos_s7.json",
                        help="JSON de data/ con el grafo (grafos_s7.json o grafo_simetrico_s7.json)")
    parser.add_argument("--n", type=int, default=None, help="n del grafo (por defecto el certificado más chico)")
    parser.add_argument("--dispositivo", default="numpy", help="para el producto implícito: numpy, cpu o cuda")
    parser.add_argument("--lote", type=int, default=4096, help="frecuencias por lote")
    parser.add_argument("--puntos", type=int, default=2000, help="puntos de la curva guardada")
    args = parser.parse_args()

    fila = elegir_grafo(cargar_json(args.archivo), args.n)
    descripcion = fila["grafo"]
    certificado = fila["certificado"]
    inicio = time.time()
    grafo = GrafoImplicito(descripcion, Arreglos(None if args.dispositivo == "numpy" else args.dispositivo))
    n = grafo.n
    print(f"grafo de {args.archivo}: n = {n}, k = {grafo.k}, g = {grafo.g}, μ = {certificado['mu_exacto']}",
          flush=True)

    error = chequear_producto(descripcion, grafo, [0, 1], args.lote)
    print(f"1. H·x por Fourier contra el implícito: error relativo {error:.1e}", flush=True)

    espectro = espectro_fourier(descripcion, lote=args.lote, informar=informar(time.time(), args.lote))
    cluster = espectro["cluster"]
    formula = np.linalg.eigvalsh(cociente_formula(descripcion))
    rotacion = int(np.argmin(np.abs(cluster)))
    menor_cluster = float(np.delete(cluster, rotacion).min())
    print(f"2. modos de cluster (acople con el resto: {espectro['acople']:.1e}):", flush=True)
    print("   Fourier:", "  ".join(f"{v:.10e}" for v in cluster), flush=True)
    print(f"   |Fourier − Q de los grados| máx = {np.abs(cluster - formula).max():.1e}", flush=True)

    menor_cero = float(espectro["cero"].min())
    posicion = int(np.argmin(espectro["minimos"]))
    menor_fourier = float(espectro["minimos"][posicion])
    m_menor = int(espectro["frecuencias"][posicion])
    cota_cero = n * certificado["margen cero / n (cota inf)"]
    cota_fourier = n * certificado["margen fourier / n (cota inf)"]
    print(f"3. resto de m = 0: λ_min = {menor_cero:.10e}  cota certificada {cota_cero:.6e}"
          f"  (holgura {menor_cero / cota_cero - 1:.2e})", flush=True)
    print(f"   m ≠ 0: λ_min = {menor_fourier:.10e} en m = {m_menor}  cota certificada {cota_fourier:.6e}"
          f"  (holgura {menor_fourier / cota_fourier - 1:.2e})", flush=True)

    # la cota de m = 0 puede ser exacta (cluster sin acople: autovalor = Σ_b d_ab cos_ab + 1), así que
    # se compara con una tolerancia del orden del redondeo de float64
    tolerancia = 1e-12
    respeta_cotas = (menor_cero >= cota_cero * (1 - tolerancia)
                     and menor_fourier >= cota_fourier * (1 - tolerancia))
    estable = menor_cluster > 0 and menor_cero > 0 and menor_fourier > 0
    print(f"\nλ_2(H) = {min(menor_cluster, menor_cero, menor_fourier):.6e} > 0: {estable}"
          f"   respeta las cotas del certificado: {respeta_cotas}   λ_max = {espectro['maximo']:.6e}"
          f"   ({time.time() - inicio:.0f} s en total)", flush=True)

    abscisas, valores = curva_minimos(espectro["frecuencias"], espectro["minimos"], grafo.g, args.puntos)
    salida = {
        "archivo": args.archivo, "n": n, "g": grafo.g, "mu_exacto": certificado["mu_exacto"],
        "error_producto": error, "acople_cluster": espectro["acople"],
        "autovalores_cluster": cluster.tolist(), "autovalores_Q_grados": formula.tolist(),
        "menor_cluster": menor_cluster, "autovalores_cero": espectro["cero"].tolist(),
        "menor_cero": menor_cero, "cota_cero": cota_cero,
        "menor_fourier": menor_fourier, "m_menor_fourier": m_menor, "cota_fourier": cota_fourier,
        "maximo": espectro["maximo"], "estable": bool(estable), "respeta_cotas": bool(respeta_cotas),
        "curva_m_sobre_g": abscisas, "curva_menor_autovalor": valores,
        "segundos": time.time() - inicio,
    }
    print("guardado en", guardar_json(f"espectro_s7_n{n}.json", salida))


if __name__ == "__main__":
    main()
