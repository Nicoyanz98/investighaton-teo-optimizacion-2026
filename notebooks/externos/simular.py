"""Simulación de Kuramoto sobre el grafo completo (secciones 11.2 y 13.9), seguida por la reducción a clusters.

Modos: perturbado (estado certificado + ruido con media 0 en cada cluster) y azar (fases uniformes).
El grafo completo se integra con RK4 hasta --tiempo (desde --continuar-desde sigue en paralelo la reducción
exacta a clusters, hasta --tiempo-clusters). Guarda resultados/simulacion_n<n>.json (una clave por corrida)
y las figuras en resultados/figuras/.
Uso: python externos/simular.py [--archivo ...] [--modos perturbado,azar] [--ruido 0.05] [--tiempo 60]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biblioteca_s7 import *  # noqa: E402,F403


def main():
    """Corre las simulaciones pedidas y guarda series y figuras."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archivo", default="grafos_s7.json")
    parser.add_argument("--modos", default="perturbado", help="perturbado y/o azar, separados por comas")
    parser.add_argument("--dispositivo", default="numpy", help="numpy, cpu o cuda")
    parser.add_argument("--tiempo", type=float, default=60.0)
    parser.add_argument("--continuar-desde", type=float, default=None)
    parser.add_argument("--fotos", default=None, help="tiempos de las fotos (por defecto 0, T/20, T/4, T)")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--ruido", type=float, default=0.05)
    parser.add_argument("--tiempo-clusters", type=float, default=1e7)
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    fila = elegir_grafo(cargar_json(args.archivo))
    descripcion = fila["grafo"]
    n = descripcion["n"]
    nombre_json = f"simulacion_n{n}.json"
    resumen = cargar_json(nombre_json) if existe_resultado(nombre_json) else {"n": n, "origen": args.archivo}
    simulacion = Simulacion(descripcion, Arreglos(None if args.dispositivo == "numpy" else args.dispositivo),
                            semilla=args.semilla)
    fotos = [0, args.tiempo / 20, args.tiempo / 4, args.tiempo]
    if args.fotos:
        fotos = [float(t) for t in args.fotos.split(",")]
    desde = args.tiempo if args.continuar_desde is None else args.continuar_desde
    fases = np.array([float(p) for p in descripcion["fases"]])
    for modo in args.modos.split(","):
        print(f"simulación '{modo}' (grafo completo, n = {n:,}, {args.dispositivo}):", flush=True)
        if modo == "perturbado":
            theta = simulacion.inicial_perturbado(args.ruido, args.semilla)
            titulo = f"Desde el estado certificado + ruido de {args.ruido} rad dentro de cada cluster (n = {n:,})"
            sufijo = f"perturbado_ruido{args.ruido:g}_t{args.tiempo:g}"
        else:
            theta = simulacion.inicial_azar(args.semilla)
            titulo = f"Desde fases uniformes al azar (n = {n:,})"
            sufijo = f"azar_t{args.tiempo:g}"
        serie, instantaneas, theta = simulacion.correr(theta, desde, args.dt, fotos)
        # en `desde` cada cluster está casi concentrado: la reducción a sus fases medias sigue desde ahí
        y0 = fases + envolver(simulacion.medias_cluster(theta) - fases)
        tiempos, distancias, radios = integrar_clusters(descripcion, y0, args.tiempo_clusters)
        continuacion = {"desde": desde, "t": (desde + tiempos[1:]).tolist(), "distancia": distancias[1:],
                        "r": radios[1:]}
        if args.tiempo > desde:
            resto, fotos_resto, theta = simulacion.correr(theta, args.tiempo, args.dt, fotos, t0=desde)
            for clave in serie:
                serie[clave].extend(resto[clave][1:])
            instantaneas.extend(fotos_resto[1:] if fotos_resto and fotos_resto[0][0] == desde else fotos_resto)
        archivo = CARPETA_RESULTADOS / "figuras" / f"grafo_n{n}_sim_{sufijo}.png"
        archivo.parent.mkdir(parents=True, exist_ok=True)
        dibujar_simulacion(serie, continuacion, instantaneas, simulacion.cluster_muestra, descripcion, titulo,
                           archivo)
        print(f"  fin: r = {serie['r'][-1]:.4f}, distancia = {serie['distancia'][-1]:.2e};"
              f" reducción hasta t = {continuacion['t'][-1]:.0e}: r = {radios[-1]:.4f},"
              f" distancia = {distancias[-1]:.2e} → {archivo}", flush=True)
        resumen[sufijo] = {"grafo completo": serie, "continuacion": continuacion}
        guardar_json(nombre_json, resumen)
    print("series guardadas en", CARPETA_RESULTADOS / nombre_json)


if __name__ == "__main__":
    main()
