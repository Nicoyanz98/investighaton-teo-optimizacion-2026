"""Validación ciega (sección 10.2): grados, velocidad, modos de cluster y Lanczos del grafo real.

Lee la descripción compacta de resultados/<archivo> (grafos_s7.json: el general; grafo_simetrico_s7.json: el
simétrico) y guarda resultados/validacion_s7_n<n>.json.
Uso: python externos/validar_grafo.py [--archivo grafo_simetrico_s7.json] [--dispositivo numpy|cpu|cuda]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biblioteca_s7 import *  # noqa: E402,F403


def main():
    """Corre los cinco chequeos y guarda el resultado."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archivo", default="grafos_s7.json")
    parser.add_argument("--dispositivo", default="numpy", help="numpy, cpu o cuda")
    parser.add_argument("--iteraciones", type=int, default=30000)
    parser.add_argument("--cada", type=int, default=50)
    parser.add_argument("--tolerancia", type=float, default=1e-6, help="residuo relativo para cortar Lanczos")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    fila = elegir_grafo(cargar_json(args.archivo))
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
    segundos_producto = time.time() - t0
    print(f"un producto H·x: {segundos_producto:.2f} s", flush=True)

    minimos, maximos = grados_medidos(grafo)
    esperados = np.array(descripcion["grados"])
    grados_ok = bool((minimos == esperados).all() and (maximos == esperados).all())
    grado_min = int((esperados.sum(axis=1)).min())
    print(f"1. grados exactos: {grados_ok}   grado mínimo δ = {grado_min}", flush=True)

    velocidad = np.abs(arreglos.a_numpy(grafo.velocidad())).max()
    print(f"2. max |θ̇| = {velocidad:.3e}", flush=True)

    Q = cociente_implicito(grafo)
    diferencia = float(np.abs(Q - cociente_formula(descripcion)).max())
    autovalores_Q = np.linalg.eigvalsh((Q + Q.T) / 2)
    print(f"3. modos de cluster: |Q implícita − fórmula| = {diferencia:.2e}", flush=True)
    rotacion = int(np.argmin(np.abs(autovalores_Q)))
    menor_cluster = float(np.delete(autovalores_Q, rotacion).min())

    cota = n * min(certificado["margen fourier / n (cota inf)"], certificado["margen cero / n (cota inf)"])
    print(f"4. resto de los modos (cota de Gershgorin certificada: λ ≥ {cota:.6e})", flush=True)
    resto = correr_lanczos(grafo, grafo.quitar_medias_cluster, "resto", args, args.semilla)
    print(f"5. Lanczos en 1^⊥ (esperado: menor autovalor de Q = {menor_cluster:.10e})", flush=True)
    completo = correr_lanczos(grafo, grafo.quitar_media, "1^⊥", args, args.semilla + 1)

    estable = menor_cluster > 0 and resto["menor"] > 0
    salida = {
        "h": fila["h"], "n": n, "dispositivo": args.dispositivo, "mu_exacto": certificado["mu_exacto"],
        "segundos_producto": segundos_producto,
        "grados_ok": grados_ok, "grado_min": grado_min, "max_velocidad": float(velocidad),
        "Q_diferencia": diferencia, "autovalores_Q": autovalores_Q.tolist(),
        "menor_cluster": menor_cluster, "cota_gershgorin": cota, "resto": resto, "completo": completo,
        "estable": bool(estable), "segundos": time.time() - inicio,
    }
    print("guardado en", guardar_json(f"validacion_s7_n{n}.json", salida))


if __name__ == "__main__":
    main()
