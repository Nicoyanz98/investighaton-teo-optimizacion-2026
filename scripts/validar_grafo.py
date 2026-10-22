"""Fase 4 del grafo explícito: validación ciega, en float64, de un grafo certificado de data/grafos_s7.json.

Reconstruye el grafo a partir de su descripción compacta (k, g, tablas t, fases) y, sin usar Fourier
ni Gershgorin, calcula con productos implícitos W·y y H·x (src/validacion.py):
  1. Grados: cuenta con W·1_b los vecinos de cada vértice en cada cluster; tienen que ser d_ab exactos.
  2. Velocidad: max_u |θ̇_u| en las fases certificadas (tiene que ser ~ error de redondeo).
  3. Modos de cluster: Q_ab = 1_aᵀ H 1_b / √(n_a n_b) con H implícito, contra la fórmula de los grados.
     Sus autovalores son los de H en los modos de cluster (uno es la rotación, ≈ 0).
  4. Resto de los modos: Lanczos de H en el complemento de los modos de cluster. El certificado
     garantiza λ ≥ (cota de Gershgorin); acá se mide λ_min de verdad.
  5. Con --completo: Lanczos de H en 1^⊥ desde un vector al azar, sin separar los modos de cluster.
     Tiene que dar el menor autovalor de Q (con n = 3·10⁷ convergió en 50 pasos: el espectro son
     pocos autovalores aislados más una banda lejana).
Como los modos de cluster forman un subespacio invariante (por 1), λ_min(H en 1^⊥) = min(3, 4).

Salida: data/validacion_s7_n<n>.json.

Comandos (en el server: GTX 1080 Ti, torch + CUDA):
  .venv/bin/python scripts/validar_grafo.py --dispositivo cuda --completo
  (CPU sin torch: --dispositivo numpy; el grafo de 3·10⁷ nodos tarda varios segundos por producto)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimentos import cargar_json, guardar_json  # noqa: E402
from src.validacion import (  # noqa: E402
    Arreglos, GrafoImplicito, cociente_formula, cociente_implicito, grados_medidos, lanczos,
)


def elegir_grafo(n):
    """Fila de grafos_s7.json: el reticulado certificado con ese n (o el más chico si n es None)."""
    filas = [f for f in cargar_json("grafos_s7.json") if f["redondeo"] == "reticulado" and f["certificado"]["ok"]]
    if n is not None:
        filas = [f for f in filas if f["grafo"]["n"] == n]
    return min(filas, key=lambda f: (f["grafo"]["n"], -f["h"]))


def informar(etiqueta, inicio):
    """Función que imprime el progreso de Lanczos."""
    def imprimir(fila):
        print(f"  [{etiqueta}] paso {fila['paso']:6d}  θ_min = {fila['menor']:.10e}  residuo = {fila['residuo']:.2e}"
              f"  θ_max = {fila['mayor']:.4e}  ({time.time() - inicio:.0f} s)", flush=True)
    return imprimir


def correr_lanczos(grafo, proyectar, etiqueta, args, semilla):
    """Lanczos de H con la proyección dada, desde un vector gaussiano."""
    inicio = time.time()
    resultado = lanczos(grafo.producto_hessiano, proyectar, grafo.arreglos.aleatorio(grafo.n, semilla),
                        grafo.arreglos, args.iteraciones, cada=args.cada, tolerancia=args.tolerancia,
                        informar=informar(etiqueta, inicio))
    resultado["segundos"] = time.time() - inicio
    return resultado


def main():
    """Corre los chequeos y guarda el resultado."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=None, help="n del grafo (por defecto el certificado más chico)")
    parser.add_argument("--dispositivo", default="cuda", help="numpy, cpu o cuda")
    parser.add_argument("--iteraciones", type=int, default=30000)
    parser.add_argument("--cada", type=int, default=50)
    parser.add_argument("--tolerancia", type=float, default=1e-6, help="residuo relativo para cortar Lanczos")
    parser.add_argument("--completo", action="store_true", help="también Lanczos en 1^⊥ desde un vector al azar")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    fila = elegir_grafo(args.n)
    descripcion = fila["grafo"]
    certificado = fila["certificado"]
    arreglos = Arreglos(None if args.dispositivo == "numpy" else args.dispositivo)
    inicio = time.time()
    grafo = GrafoImplicito(descripcion, arreglos)
    n = grafo.n
    print(f"grafo h = {fila['h']}, n = {n}, k = {grafo.k}, g = {grafo.g}, μ = {certificado['mu_exacto']}"
          f" ({args.dispositivo})", flush=True)

    t0 = time.time()
    grafo.producto_hessiano(grafo.arreglos.aleatorio(n, 1))
    print(f"un producto H·x: {time.time() - t0:.2f} s", flush=True)

    minimos, maximos = grados_medidos(grafo)
    esperados = np.array(descripcion["grados"])
    grados_ok = bool((minimos == esperados).all() and (maximos == esperados).all())
    grado_min = int((esperados.sum(axis=1)).min())
    print(f"1. grados exactos: {grados_ok}   grado mínimo δ = {grado_min}   μ = (δ + 1)/n = {(grado_min + 1) / n:.8f}",
          flush=True)

    velocidad = np.abs(arreglos.a_numpy(grafo.velocidad())).max()
    print(f"2. max |θ̇| = {velocidad:.3e}   (= {velocidad / n:.2e}·n)", flush=True)

    Q = cociente_implicito(grafo)
    diferencia = float(np.abs(Q - cociente_formula(descripcion)).max())
    autovalores_Q = np.linalg.eigvalsh((Q + Q.T) / 2)
    print(f"3. modos de cluster: |Q implícita − fórmula| = {diferencia:.2e}", flush=True)
    print("   autovalores de Q:", "  ".join(f"{v:.6e}" for v in autovalores_Q), flush=True)
    rotacion = int(np.argmin(np.abs(autovalores_Q)))
    menor_cluster = float(np.delete(autovalores_Q, rotacion).min())

    cota = n * min(certificado["margen fourier / n (cota inf)"], certificado["margen cero / n (cota inf)"])
    print(f"4. resto de los modos (cota de Gershgorin certificada: λ ≥ {cota:.6e})", flush=True)
    resto = correr_lanczos(grafo, grafo.quitar_medias_cluster, "resto", args, args.semilla)
    print(f"   λ_min = {resto['menor']:.10e}  (residuo {resto['residuo']:.1e}, {resto['pasos']} pasos)"
          f"  ≥ cota: {resto['menor'] >= cota}", flush=True)

    completo = None
    if args.completo:
        print(f"5. Lanczos en 1^⊥ (esperado: menor autovalor de Q = {menor_cluster:.10e})", flush=True)
        completo = correr_lanczos(grafo, grafo.quitar_media, "1^⊥", args, args.semilla + 1)
        relativo = abs(completo["menor"] - menor_cluster) / abs(menor_cluster)
        print(f"   λ_min = {completo['menor']:.10e}  (residuo {completo['residuo']:.1e}, {completo['pasos']} pasos)"
              f"  diferencia relativa con Q: {relativo:.1e}", flush=True)

    estable = menor_cluster > 0 and resto["menor"] > 0
    print(f"\nλ_min(H en 1^⊥) = {min(menor_cluster, resto['menor']):.6e} > 0: {estable}"
          f"   ({time.time() - inicio:.0f} s en total)", flush=True)
    salida = {
        "h": fila["h"], "n": n, "dispositivo": args.dispositivo, "mu_exacto": certificado["mu_exacto"],
        "grados_ok": grados_ok, "grado_min": grado_min, "max_velocidad": float(velocidad),
        "Q_diferencia": diferencia, "autovalores_Q": autovalores_Q.tolist(),
        "menor_cluster": menor_cluster, "cota_gershgorin": cota, "resto": resto, "completo": completo,
        "estable": bool(estable), "segundos": time.time() - inicio,
    }
    print("guardado en", guardar_json(f"validacion_s7_n{n}.json", salida))


if __name__ == "__main__":
    main()
