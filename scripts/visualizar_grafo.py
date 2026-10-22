"""Dibuja un grafo certificado (data/grafos_s7.json o data/s7_subpiezas.json) y simula Kuramoto sobre él.

Figuras, en data/figuras/:
  grafo_n<n>_estructura.png
    (a) grafo de clusters: cada cluster en su fase del estado certificado (las separaciones dentro de
        un sector, de ~0.03 rad, están exageradas para que se vean), área ∝ n_a, y una arista por par de
        clusters con su densidad d_ab/n_b (los opuestos no tienen links);
    (b) la adyacencia a escala: cada bloque Z_g se parte en franjas y cada píxel es la fracción de
        pares unidos. Se ven los cliques, los pares vacíos y las bandas circulantes (intervalos).
  grafo_n<n>_sim_<modo>.png, con --simular:
    perturbado  grafo completo desde el estado certificado más ruido gaussiano dentro de cada cluster
                (se le resta la media por cluster): tiene que volver al estado, no sincronizar.
    azar        grafo completo desde fases uniformes al azar: lo esperable es que sincronice (la trampa
                tiene una cuenca chica).
    clusters    dinámica exacta de los estados constantes por cluster (una ecuación por cluster, porque
                los grados son exactos), por tiempos largos: ahí viven los modos lentos (λ/n ~ 5·10⁻⁶),
                que la simulación del grafo completo no llega a ver.

Dinámica (tiempo normalizado): θ̇_u = (1/n) Σ_v W_uv sin(θ_v − θ_u). El grafo completo se integra con
RK4 y los productos implícitos W·y de src/validacion.py (numpy, o torch en cpu/cuda); los estados por
cluster, con LSODA (scipy).

Comandos:
  .venv/bin/python scripts/visualizar_grafo.py                                     # solo las figuras
  .venv/bin/python scripts/visualizar_grafo.py --simular perturbado,azar,clusters  # grafo B, CPU, ~minutos
  python scripts/visualizar_grafo.py --n 30000004 --simular perturbado,azar --dispositivo cuda   # server
"""
import argparse
import colorsys
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimentos import CARPETA_DATOS, cargar_json, guardar_json  # noqa: E402
from src.validacion import Arreglos, GrafoImplicito  # noqa: E402

LETRAS = ["A", "B", "C", "W"]
TONOS = [0.58, 0.07, 0.40, 0.80]     # azul, naranja, verde, violeta (uno por sector)
TINTA = "#1f2328"
GRIS = "#8c959f"


# ---------------------------------------------------------------- el grafo

def grafos_certificados():
    """Todas las descripciones compactas certificadas: [(origen, fila)]."""
    filas = []
    for fila in cargar_json("grafos_s7.json"):
        if fila["redondeo"] == "reticulado" and fila["certificado"]["ok"]:
            filas.append(("grafos_s7.json", fila))
    for fila in cargar_json("s7_subpiezas.json"):
        if fila.get("certificado") and fila["certificado"]["ok"]:
            filas.append(("s7_subpiezas.json", fila))
    return filas


def elegir_grafo(n):
    """Descripción compacta del grafo con ese n (o del más chico si n es None) y su μ exacto."""
    candidatos = grafos_certificados()
    if n is not None:
        candidatos = [c for c in candidatos if c[1]["grafo"]["n"] == n]
        if not candidatos:
            raise SystemExit(f"no hay un grafo certificado con n = {n}")
    origen, fila = min(candidatos, key=lambda c: c[1]["grafo"]["n"])
    return fila["grafo"], fila["certificado"]["mu_exacto"], origen


def sectores(fases):
    """Sector de C4 de cada cluster (0: A, 1: B, 2: C, 3: W) y su desvío respecto de k·π/2."""
    sector = np.round(np.asarray(fases) / (np.pi / 2)).astype(int) % 4
    desvio = np.asarray(fases) - sector * np.pi / 2
    return sector, desvio


def nombres_y_colores(fases):
    """Nombre (Am, Ap, … o W1, W2 si las piezas tienen la misma fase) y color de cada cluster."""
    sector, desvio = sectores(fases)
    nombres = [""] * len(fases)
    colores = [None] * len(fases)
    for s in range(4):
        piezas = sorted(np.flatnonzero(sector == s), key=lambda a: desvio[a])
        separadas = len(piezas) == 2 and abs(desvio[piezas[0]] - desvio[piezas[1]]) > 1e-3
        for orden, a in enumerate(piezas):
            if len(piezas) == 1:
                nombres[a] = LETRAS[s]
            elif separadas:
                nombres[a] = LETRAS[s] + ("m" if orden == 0 else "p")
            else:
                nombres[a] = f"{LETRAS[s]}{orden + 1}"
            luz = 0.42 + 0.2 * orden / max(len(piezas) - 1, 1)
            colores[a] = colorsys.hls_to_rgb(TONOS[s], luz, 0.65)
    return nombres, colores


def fases_vertices(descripcion):
    """θ del estado certificado, vértice por vértice."""
    fases = [float(p) for p in descripcion["fases"]]
    return np.repeat(fases, np.array(descripcion["k"]) * descripcion["g"])


# ---------------------------------------------------------------- figura de la estructura

def densidad_circulante(t, g, franjas):
    """Densidad por píxel (franjas × franjas) del bloque circulante v ~ v + s, s < t, en Z_g.

    Solo depende de la diferencia de franjas q − p; los pares con w − v = δ son L − |δ − (q − p)L|.
    """
    largo = g // franjas
    desfase = np.arange(-largo + 1, largo)
    peso = largo - np.abs(desfase)
    por_diferencia = np.zeros(franjas)
    for diferencia in range(franjas):
        delta = diferencia * largo + desfase
        por_diferencia[diferencia] = (peso * ((delta % g) < t)).sum() / largo**2
    p, q = np.meshgrid(np.arange(franjas), np.arange(franjas), indexing="ij")
    return por_diferencia[(q - p) % franjas]


def imagen_adyacencia(descripcion, franjas):
    """Matriz de densidades (K·franjas)² del grafo: bloque por bloque."""
    k = descripcion["k"]
    g = descripcion["g"]
    inicios = np.concatenate([[0], np.cumsum(k)]) * franjas
    lado = sum(k) * franjas
    imagen = np.zeros((lado, lado))
    for a in range(len(k)):
        imagen[inicios[a]: inicios[a + 1], inicios[a]: inicios[a + 1]] = 1
    for clave, tabla in descripcion["tablas"].items():
        a, b = (int(c) for c in clave.split(","))
        for i in range(k[a]):
            for j in range(k[b]):
                if tabla[i][j] == 0:
                    continue
                bloque = densidad_circulante(tabla[i][j], g, franjas)
                filas = slice(inicios[a] + i * franjas, inicios[a] + (i + 1) * franjas)
                columnas = slice(inicios[b] + j * franjas, inicios[b] + (j + 1) * franjas)
                imagen[filas, columnas] = bloque
                imagen[columnas, filas] = bloque.T
    return imagen, inicios


def posiciones_clusters(fases, exageracion):
    """Posición de cada cluster: ángulo del sector más el desvío exagerado; las piezas con la misma fase, hacia adentro."""
    sector, desvio = sectores(fases)
    angulos = sector * np.pi / 2 + exageracion * desvio
    radios = np.ones(len(fases))
    for a in range(len(fases)):
        for b in range(a):
            if sector[a] == sector[b] and abs(desvio[a] - desvio[b]) < 1e-3:
                radios[a] = radios[b] - 0.45
    return np.column_stack([radios * np.cos(angulos), radios * np.sin(angulos)])


def dibujar_grafo_clusters(eje, descripcion, nombres, colores):
    """Panel (a): clusters en sus fases y aristas con la densidad de cada par."""
    fases = [float(p) for p in descripcion["fases"]]
    k = descripcion["k"]
    g = descripcion["g"]
    n_a = np.array(k) * g
    _, desvio = sectores(fases)
    exageracion = 0.45 / np.abs(desvio).max()
    posiciones = posiciones_clusters(fases, exageracion)
    circulo = np.linspace(0, 2 * np.pi, 200)
    eje.plot(np.cos(circulo), np.sin(circulo), color="#d0d7de", lw=1, zorder=0)
    for a in range(len(k)):
        for b in range(a + 1, len(k)):
            densidad = descripcion["grados"][a][b] / n_a[b]
            if densidad == 0:
                continue
            (x0, y0), (x1, y1) = posiciones[a], posiciones[b]
            completo = densidad > 0.999
            eje.plot([x0, x1], [y0, y1], color=GRIS if completo else TINTA, lw=0.6 + 2.4 * densidad,
                     alpha=0.35 if completo else 0.8, zorder=1)
            if densidad < 0.95:
                # etiquetas a 40% o 60% del camino, alternadas, para que no se encimen las que salen del mismo cluster
                fraccion = 0.4 if (a + b) % 2 == 0 else 0.6
                eje.text(x0 + fraccion * (x1 - x0), y0 + fraccion * (y1 - y0), f"{100 * densidad:.1f}%", fontsize=7,
                         color=TINTA,
                         ha="center", va="center", zorder=3,
                         bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
    areas = 1300 * n_a / n_a.max()
    eje.scatter(posiciones[:, 0], posiciones[:, 1], s=areas, c=colores, edgecolors="white", linewidths=2, zorder=2)
    for a in range(len(k)):
        x, y = posiciones[a]
        radio = np.hypot(x, y)
        # los clusters del borde llevan la etiqueta afuera; los de adentro (misma fase que otro), hacia el centro
        escala = 1.36 / radio if radio > 0.99 else (radio - 0.36) / radio
        eje.text(x * escala, y * escala, f"{nombres[a]}\nφ = {fases[a]:.3f}\nn = {n_a[a]:,}", fontsize=7.5,
                 color=TINTA, ha="center", va="center")
    eje.set(xlim=(-1.75, 1.75), ylim=(-1.75, 1.75), aspect="equal", xticks=[], yticks=[])
    for borde in eje.spines.values():
        borde.set_visible(False)
    eje.set_title(f"(a) clusters en su fase (desvíos ×{exageracion:.0f}; misma fase: uno adentro del otro)\n"
                  "gris: unión completa; negro: parcial (etiqueta si < 95%; el resto, en (b))", fontsize=9,
                  color=TINTA)


def dibujar_estructura(descripcion, mu, archivo):
    """Figura con el grafo de clusters y la adyacencia a escala."""
    fases = [float(p) for p in descripcion["fases"]]
    nombres, colores = nombres_y_colores(fases)
    fig, ejes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.1]})
    dibujar_grafo_clusters(ejes[0], descripcion, nombres, colores)

    franjas = max(4, 330 // sum(descripcion["k"]))
    imagen, inicios = imagen_adyacencia(descripcion, franjas)
    figura = ejes[1].imshow(imagen, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    for corte in inicios[1:-1]:
        ejes[1].axhline(corte - 0.5, color=TINTA, lw=0.7)
        ejes[1].axvline(corte - 0.5, color=TINTA, lw=0.7)
    centros = (inicios[:-1] + inicios[1:]) / 2 - 0.5
    ejes[1].set_xticks(centros, nombres, fontsize=8)
    ejes[1].set_yticks(centros, nombres, fontsize=8)
    ejes[1].tick_params(length=0)
    for etiqueta, color in zip(ejes[1].get_xticklabels(), colores):
        etiqueta.set_color(color)
    for etiqueta, color in zip(ejes[1].get_yticklabels(), colores):
        etiqueta.set_color(color)
    ejes[1].set_title(f"(b) adyacencia a escala: {franjas} franjas por bloque de g = {descripcion['g']:,} vértices",
                      fontsize=9, color=TINTA)
    barra = fig.colorbar(figura, ax=ejes[1], fraction=0.046, pad=0.03)
    barra.set_label("fracción de pares unidos", fontsize=8)
    fig.suptitle(f"Grafo certificado: n = {descripcion['n']:,} vértices, k = {descripcion['k']}, "
                 f"μ = {mu} ≈ {eval(mu):.6f}", fontsize=11, color=TINTA)
    fig.tight_layout()
    fig.savefig(archivo, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- simulación del grafo completo

class Simulacion:
    """Kuramoto normalizado sobre el grafo implícito: θ̇ = (1/n) Σ_v W_uv sin(θ_v − θ_u)."""

    def __init__(self, descripcion, arreglos, muestra=4000, semilla=0):
        self.grafo = GrafoImplicito(descripcion, arreglos)
        self.arreglos = arreglos
        self.estrella = arreglos.desde(fases_vertices(descripcion))
        rng = np.random.default_rng(semilla)
        self.muestra = np.sort(rng.choice(self.grafo.n, size=min(muestra, self.grafo.n), replace=False))
        self.cluster_muestra = np.searchsorted(self.grafo.inicios, self.muestra, side="right") - 1

    def campo(self, theta):
        """θ̇ normalizado por n."""
        return self.grafo.velocidad_en(theta) / self.grafo.n

    def paso_rk4(self, theta, dt):
        """Un paso de Runge–Kutta 4."""
        k1 = self.campo(theta)
        k2 = self.campo(theta + 0.5 * dt * k1)
        k3 = self.campo(theta + 0.5 * dt * k2)
        k4 = self.campo(theta + dt * k3)
        return theta + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def medir(self, theta):
        """r = |⟨e^{iθ}⟩| y distancia (máxima, sin la rotación global) al estado certificado."""
        coseno, seno = self.arreglos.coseno_seno(theta)
        r = (float(coseno.mean()) ** 2 + float(seno.mean()) ** 2) ** 0.5
        diferencia = theta - self.estrella
        coseno, seno = self.arreglos.coseno_seno(diferencia)
        giro = np.arctan2(float(seno.mean()), float(coseno.mean()))
        envuelta = (diferencia - giro + np.pi) % (2 * np.pi) - np.pi
        return r, float(abs(envuelta).max())

    def correr(self, theta, tiempo, dt, fotos):
        """Integra hasta `tiempo`; guarda r y la distancia en cada paso y las fases de la muestra en `fotos`."""
        pasos = int(round(tiempo / dt))
        tiempos_foto = set()
        for t in fotos:
            tiempos_foto.add(int(round(t / dt)))
        serie = {"t": [], "r": [], "distancia": []}
        instantaneas = []
        inicio = time.time()
        for paso in range(pasos + 1):
            r, distancia = self.medir(theta)
            serie["t"].append(paso * dt)
            serie["r"].append(r)
            serie["distancia"].append(distancia)
            if paso in tiempos_foto:
                instantaneas.append((paso * dt, self.arreglos.tomar(theta, self.muestra)))
            if paso % max(1, pasos // 10) == 0:
                print(f"  t = {paso * dt:7.1f}  r = {r:.4f}  distancia al estado = {distancia:.2e}"
                      f"  ({time.time() - inicio:.0f} s)", flush=True)
            if paso < pasos:
                theta = self.paso_rk4(theta, dt)
        return serie, instantaneas

    def inicial_perturbado(self, ruido, semilla):
        """Estado certificado + ruido gaussiano con media 0 en cada cluster (solo modos dentro de los clusters)."""
        perturbacion = self.arreglos.aleatorio(self.grafo.n, semilla) * ruido
        return self.estrella + self.grafo.quitar_medias_cluster(perturbacion)

    def inicial_azar(self, semilla):
        """Fases uniformes en [0, 2π)."""
        return self.arreglos.desde(np.random.default_rng(semilla).uniform(0, 2 * np.pi, self.grafo.n))


def dibujar_simulacion(serie, instantaneas, cluster_muestra, descripcion, titulo, archivo):
    """r(t), distancia al estado certificado y fotos de las fases de una muestra de vértices."""
    nombres, colores = nombres_y_colores([float(p) for p in descripcion["fases"]])
    fig = plt.figure(figsize=(13, 6.4))
    grilla = fig.add_gridspec(2, len(instantaneas), height_ratios=[1, 1.15])
    eje_r = fig.add_subplot(grilla[0, : len(instantaneas) // 2])
    eje_d = fig.add_subplot(grilla[0, len(instantaneas) // 2:])
    eje_r.plot(serie["t"], serie["r"], color="#2a78d6", lw=2)
    eje_r.set(xlabel="tiempo (normalizado)", ylabel="r = |⟨e^{iθ}⟩|", ylim=(0, 1.03),
              title="parámetro de orden (1 = sincronizado)")
    eje_d.plot(serie["t"], serie["distancia"], color="#eb6834", lw=2)
    eje_d.set(xlabel="tiempo (normalizado)", ylabel="max |θ − θ*| (rad)", yscale="log",
              title="distancia al estado certificado θ*")
    for eje in (eje_r, eje_d):
        eje.grid(alpha=0.25)
    rng = np.random.default_rng(1)
    color_vertices = np.array([colores[c] for c in cluster_muestra])
    for columna, (t, fases) in enumerate(instantaneas):
        eje = fig.add_subplot(grilla[1, columna], projection="polar")
        radio = 1 + 0.08 * rng.standard_normal(len(fases))
        eje.scatter(fases, radio, s=4, c=color_vertices, alpha=0.6, linewidths=0)
        eje.set(ylim=(0, 1.35), yticks=[], title=f"t = {t:g}")
        eje.set_xticks(np.arange(4) * np.pi / 2, ["0", "π/2", "π", "3π/2"], fontsize=7)
    manijas = []
    for nombre, color in zip(nombres, colores):
        manijas.append(plt.Line2D([], [], marker="o", ls="", color=color, label=nombre))
    fig.legend(handles=manijas, loc="lower center", ncol=len(nombres), fontsize=8, frameon=False)
    fig.suptitle(titulo, fontsize=11, color=TINTA)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(archivo, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- dinámica exacta por clusters

def campo_clusters(descripcion):
    """(f, jac) de ẏ_a = Σ_b (d_ab/n) sin(y_b − y_a): Kuramoto normalizado restringido a estados constantes por cluster."""
    A = np.array(descripcion["grados"], dtype=float) / descripcion["n"]

    def f(t, y):
        return (A * np.sin(y[None, :] - y[:, None])).sum(axis=1)

    def jac(t, y):
        C = A * np.cos(y[None, :] - y[:, None])
        np.fill_diagonal(C, 0)
        return C - np.diag(C.sum(axis=1))

    return f, jac


def simular_clusters(descripcion, amplitudes, tiempo, semilla):
    """Para cada amplitud δ: perturba las fases de los clusters (al azar y en el modo más lento) e integra."""
    fases = np.array([float(p) for p in descripcion["fases"]])
    x = np.array(descripcion["k"], dtype=float) / sum(descripcion["k"])
    f, jac = campo_clusters(descripcion)
    autovalores, vectores = np.linalg.eig(jac(0, fases))
    orden = np.argsort(-autovalores.real)
    direcciones = {"modo más lento": vectores[:, orden[1]].real,
                   "al azar": np.random.default_rng(semilla).standard_normal(len(fases))}
    tiempos = np.concatenate([[0], np.logspace(-1, np.log10(tiempo), 120)])
    corridas = []
    for nombre, direccion in direcciones.items():
        direccion = direccion - np.sum(x * direccion)
        direccion = direccion / np.abs(direccion).max()
        for delta in amplitudes:
            inicio = time.time()
            sol = solve_ivp(f, (0, tiempo), fases + delta * direccion, method="LSODA", jac=jac,
                            rtol=1e-10, atol=1e-13, t_eval=tiempos)
            distancias = []
            for y in sol.y.T:
                diferencia = y - fases
                diferencia = diferencia - np.sum(x * diferencia)
                distancias.append(float(np.abs((diferencia + np.pi) % (2 * np.pi) - np.pi).max()))
            r_final = float(abs(np.sum(x * np.exp(1j * sol.y[:, -1]))))
            print(f"  {nombre:15s} δ = {delta:.0e}: distancia final {distancias[-1]:.1e},  r final = {r_final:.4f}"
                  f"  ({time.time() - inicio:.1f} s)", flush=True)
            corridas.append({"direccion": nombre, "delta": delta, "t": sol.t.tolist(), "distancia": distancias,
                             "r_final": r_final})
    return corridas, -np.sort(autovalores.real)[::-1]


def dibujar_clusters(corridas, tasas, descripcion, archivo):
    """Distancia al estado certificado contra el tiempo, en escala log-log, para cada perturbación."""
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    paleta = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e87ba4", "#6e40c9"]
    for eje, direccion in zip(ejes, ["modo más lento", "al azar"]):
        propias = [c for c in corridas if c["direccion"] == direccion]
        for color, corrida in zip(paleta, propias):
            t = np.array(corrida["t"][1:])
            final = "vuelve" if corrida["distancia"][-1] < 1e-6 else f"se va (r final = {corrida['r_final']:.2f})"
            eje.plot(t, corrida["distancia"][1:], color=color, lw=2, label=f"δ = {corrida['delta']:.0e}: {final}")
        for tasa in tasas[1:3]:
            eje.axvline(1 / tasa, color=GRIS, lw=1, ls=":")
        eje.set(xscale="log", yscale="log", xlabel="tiempo (normalizado)", title=f"perturbación: {direccion}")
        eje.grid(alpha=0.25)
        eje.legend(fontsize=8)
    ejes[0].set_ylabel("max |y − φ| (rad), sin la rotación")
    fig.suptitle(f"Estados constantes por cluster (dinámica exacta del grafo de n = {descripcion['n']:,}); "
                 f"punteadas: 1/λ de los dos modos lentos", fontsize=10, color=TINTA)
    fig.tight_layout()
    fig.savefig(archivo, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    """Figuras de la estructura y, si se pide, las simulaciones."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=None, help="n del grafo (por defecto el certificado más chico)")
    parser.add_argument("--simular", default="", help="perturbado, azar y/o clusters, separados por comas")
    parser.add_argument("--dispositivo", default="numpy", help="numpy, cpu o cuda")
    parser.add_argument("--tiempo", type=float, default=60.0, help="tiempo de las simulaciones del grafo completo")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--ruido", type=float, default=0.5, help="desvío del ruido de 'perturbado' (rad)")
    parser.add_argument("--amplitudes", default="1e-4,3e-4,1e-3,1e-2,1e-1", help="δ de 'clusters'")
    parser.add_argument("--tiempo-clusters", type=float, default=1e7)
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    descripcion, mu, origen = elegir_grafo(args.n)
    n = descripcion["n"]
    carpeta = CARPETA_DATOS / "figuras"
    carpeta.mkdir(exist_ok=True)
    print(f"grafo de {origen}: n = {n:,}, k = {descripcion['k']}, g = {descripcion['g']:,}, μ = {mu}", flush=True)
    archivo = carpeta / f"grafo_n{n}_estructura.png"
    dibujar_estructura(descripcion, mu, archivo)
    print("estructura:", archivo, flush=True)

    modos = [m for m in args.simular.split(",") if m]
    resumen = {"n": n, "origen": origen}
    if "perturbado" in modos or "azar" in modos:
        arreglos = Arreglos(None if args.dispositivo == "numpy" else args.dispositivo)
        simulacion = Simulacion(descripcion, arreglos, semilla=args.semilla)
        fotos = [0, args.tiempo / 20, args.tiempo / 4, args.tiempo]
        for modo in ["perturbado", "azar"]:
            if modo not in modos:
                continue
            print(f"simulación '{modo}' (grafo completo, {args.dispositivo}):", flush=True)
            if modo == "perturbado":
                theta = simulacion.inicial_perturbado(args.ruido, args.semilla)
                titulo = (f"Desde el estado certificado + ruido de {args.ruido} rad dentro de cada cluster "
                          f"(n = {n:,})")
            else:
                theta = simulacion.inicial_azar(args.semilla)
                titulo = f"Desde fases uniformes al azar (n = {n:,})"
            serie, instantaneas = simulacion.correr(theta, args.tiempo, args.dt, fotos)
            archivo = carpeta / f"grafo_n{n}_sim_{modo}.png"
            dibujar_simulacion(serie, instantaneas, simulacion.cluster_muestra, descripcion, titulo, archivo)
            print(f"  r final = {serie['r'][-1]:.4f}, distancia final = {serie['distancia'][-1]:.2e} → {archivo}",
                  flush=True)
            resumen[modo] = serie
    if "clusters" in modos:
        print("simulación 'clusters' (estados constantes por cluster, LSODA):", flush=True)
        amplitudes = [float(v) for v in args.amplitudes.split(",")]
        corridas, tasas = simular_clusters(descripcion, amplitudes, args.tiempo_clusters, args.semilla)
        archivo = carpeta / f"grafo_n{n}_sim_clusters.png"
        dibujar_clusters(corridas, tasas, descripcion, archivo)
        print("  →", archivo, flush=True)
        resumen["clusters"] = {"corridas": corridas, "tasas": tasas.tolist()}
    if modos:
        print("series guardadas en", guardar_json(f"simulacion_n{n}.json", resumen))


if __name__ == "__main__":
    main()
