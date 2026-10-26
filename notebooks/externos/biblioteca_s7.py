# biblioteca
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import colorsys
import itertools
import json
import sys
import time
import warnings
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from multiprocessing import Pool
from pathlib import Path

import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import mpmath as mp
import networkx as nx
import numpy as np
from mpmath import iv
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import linprog, minimize


# biblioteca
CARPETA_RESULTADOS = Path("resultados")
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NOMBRES = ["Am", "Ap", "Bm", "Bp", "Cm", "Cp", "W"]
TINTA = "#1f2328"
BORDE = "#d0d7de"
FONDO_ENCABEZADO = "#e8ecf0"
FONDO_ALTERNO = "#f6f8fa"
FONDO_RESALTE = "#dbeafe"


def silenciar_avisos():
    """Oculta los avisos de precisión de cvxpy y scipy (los resultados se verifican aparte)."""
    warnings.filterwarnings("ignore")


def correr_en_paralelo(funcion, trabajos, procesos=6):
    """Aplica funcion a cada trabajo en varios procesos; va devolviendo resultados a medida que terminan."""
    with Pool(min(procesos, len(trabajos)), initializer=silenciar_avisos) as pool:
        for resultado in pool.imap_unordered(funcion, trabajos):
            yield resultado


def guardar_json(nombre, datos):
    """Guarda datos en resultados/<nombre> y devuelve la ruta."""
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    ruta = CARPETA_RESULTADOS / nombre
    with open(ruta, "w") as archivo:
        json.dump(datos, archivo, indent=1)
    return ruta


def cargar_json(nombre):
    """Lee resultados/<nombre>."""
    with open(CARPETA_RESULTADOS / nombre) as archivo:
        return json.load(archivo)


def existe_resultado(nombre):
    """¿Ya está resultados/<nombre>?"""
    return (CARPETA_RESULTADOS / nombre).exists()


def mostrar_tabla(columnas, filas, titulo=None, resaltar=()):
    """Dibuja una tabla de resultados con matplotlib: encabezado gris, filas alternadas y filas resaltadas."""
    textos = []
    for fila in filas:
        textos.append([str(valor) for valor in fila])
    fig, eje = plt.subplots(figsize=(1, 0.22 * (len(filas) + 1)))
    eje.axis("off")
    tabla = eje.table(cellText=textos, colLabels=columnas, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.auto_set_column_width(list(range(len(columnas))))
    tabla.scale(1, 1.3)
    for (i, j), celda in tabla.get_celld().items():
        celda.set_edgecolor(BORDE)
        celda.set_linewidth(0.6)
        celda.PAD = 0.15
        celda.get_text().set_color(TINTA)
        if i == 0:
            celda.set_facecolor(FONDO_ENCABEZADO)
            celda.get_text().set_fontweight("bold")
        elif i - 1 in resaltar:
            celda.set_facecolor(FONDO_RESALTE)
            celda.get_text().set_fontweight("bold")
        elif i % 2 == 0:
            celda.set_facecolor(FONDO_ALTERNO)
    if titulo:
        # el título va arriba a la izquierda de la tabla (no de los ejes, que son más chicos)
        fig.canvas.draw()
        caja = tabla.get_window_extent(fig.canvas.get_renderer())
        x0, y1 = eje.transAxes.inverted().transform((caja.x0, caja.y1))
        eje.annotate(titulo, xy=(x0, y1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                     fontsize=10, color=TINTA)
    plt.show()


def si_no(valor):
    """Texto para un booleano en las tablas."""
    return "sí" if valor else "no"


# biblioteca
def diferencias_fase(theta):
    """Matriz D con D[v, u] = θ_u − θ_v."""
    return theta[None, :] - theta[:, None]


def velocidad(theta, W):
    """θ̇_v = Σ_u w_uv sin(θ_u − θ_v), que es −∇E."""
    return (W * np.sin(diferencias_fase(theta))).sum(axis=1)


def energia(theta, W):
    """E = ½ Σ_{u,v} w_uv (1 − cos(θ_u − θ_v))."""
    return 0.5 * (W * (1 - np.cos(diferencias_fase(theta)))).sum()


def parametro_orden(theta):
    """r = (1/n) Σ e^{iθ_v}: |r| mide cuán agrupadas están las fases y arg r hacia dónde."""
    return np.exp(1j * theta).mean()


def hessiano(theta, W):
    """Hessiano de E: Laplaciano con pesos w_uv cos(θ_u − θ_v)."""
    C = W * np.cos(diferencias_fase(theta))
    return np.diag(C.sum(axis=1)) - C


def simular(theta0, W, h=None, tol=1e-10, max_pasos=200_000, guardar=True):
    """Descenso de gradiente (Euler) sobre E desde theta0, hasta que ‖θ̇‖ < tol.

    El paso por defecto h = 1/(2·grado máximo) garantiza que la energía no suba.
    Devuelve el estado final y, si guardar=True, la historia con t, E y r.
    """
    if h is None:
        h = 1.0 / (2 * W.sum(axis=1).max())
    theta = np.array(theta0, dtype=float)
    historia = [theta.copy()]
    convergio = False
    for paso in range(1, max_pasos + 1):
        v = velocidad(theta, W)
        if np.linalg.norm(v) < tol:
            convergio = True
            break
        theta = theta + h * v
        if guardar:
            historia.append(theta.copy())
    salida = {"theta": theta, "pasos": paso, "convergio": convergio, "h": h}
    if guardar:
        historia = np.array(historia)
        salida["t"] = h * np.arange(len(historia))
        salida["historia"] = historia
        salida["E"] = np.array([energia(th, W) for th in historia])
        salida["r"] = np.array([parametro_orden(th) for th in historia])
    return salida


def clasificar(theta, W, tol=1e-6):
    """Clasifica un equilibrio por los autovalores del Hessiano (el 0 de la rotación global siempre está)."""
    autovalores = np.linalg.eigvalsh(hessiano(theta, W))
    escala = max(1.0, np.abs(autovalores).max())
    negativos = int(np.sum(autovalores < -tol * escala))
    positivos = int(np.sum(autovalores > tol * escala))
    nulos = len(autovalores) - negativos - positivos
    if negativos == 0 and nulos == 1:
        tipo = "mínimo estricto"
    elif positivos == 0 and nulos == 1:
        tipo = "máximo estricto"
    elif negativos > 0:
        tipo = "silla" if positivos > 0 else "máximo (degenerado)"
    else:
        tipo = "degenerado"
    return tipo, autovalores


def twisted_state(n, q):
    """θ_v = 2π q v / n."""
    return 2 * np.pi * q * np.arange(n) / n


# biblioteca
def grafo_circulante(n, k):
    """C_{n,k}: cada v conectado con v ± 1, …, v ± k (módulo n)."""
    W = np.zeros((n, n))
    v = np.arange(n)
    for j in range(1, k + 1):
        W[v, (v + j) % n] = 1
        W[(v + j) % n, v] = 1
    return W


def blowup_lexicografico(W, m):
    """G[K_m]: cada vértice pasa a ser un clique de m copias unido a todas las copias de sus vecinos."""
    unos = np.ones((m, m))
    identidad = np.eye(m)
    n = len(W)
    return np.kron(unos, W) + np.kron(unos - identidad, np.eye(n))


def blowup_cartesiano(W, m):
    """G □ K_m: m copias de G, y las copias de un mismo vértice forman un clique."""
    unos = np.ones((m, m))
    identidad = np.eye(m)
    n = len(W)
    return np.kron(identidad, W) + np.kron(unos - identidad, np.eye(n))


def conectividad(W):
    """μ = (δ + 1)/n, con δ el grado mínimo."""
    return (W.sum(axis=1).min() + 1) / len(W)


# biblioteca
def grados(x, P):
    """d_a = Σ_b P_ab x_b (grado normalizado de un vértice del cluster a)."""
    m = len(x)
    d = []
    for a in range(m):
        d.append(sum(P[a][b] * x[b] for b in range(m)))
    return d


def fuerzas(x, phi, P, lib=np):
    """F_a = Σ_{b≠a} P_ab x_b sin(φ_a − φ_b); el equilibrio es F = 0."""
    m = len(x)
    F = []
    for a in range(m):
        total = 0
        for b in range(m):
            if b != a:
                total = total + P[a][b] * x[b] * lib.sin(phi[a] - phi[b])
        F.append(total)
    return F


def splitting(x, phi, P, lib=np):
    """c_a = Σ_b P_ab x_b cos(φ_a − φ_b): cuánto tira el resto hacia el cluster a si un oscilador se separa."""
    m = len(x)
    c = []
    for a in range(m):
        total = P[a][a] * x[a]
        for b in range(m):
            if b != a:
                total = total + P[a][b] * x[b] * lib.cos(phi[a] - phi[b])
        c.append(total)
    return c


def matriz_clusters(x, phi, P, lib=np):
    """Matriz de L(y): Laplaciano con pesos x_a x_b P_ab cos(φ_a − φ_b), como lista de listas."""
    m = len(x)
    L = []
    for a in range(m):
        fila = [0] * m
        for b in range(m):
            if b != a:
                fila[b] = -x[a] * x[b] * P[a][b] * lib.cos(phi[a] - phi[b])
        diagonal = 0
        for b in range(m):
            if b != a:
                diagonal = diagonal - fila[b]
        fila[a] = diagonal
        L.append(fila)
    return L


def jacobiano_fuerzas(x, phi, P, lib=np):
    """∂F_a/∂φ_b para a, b = 1..m−1 (la fase φ_0 queda fija para sacar la rotación global)."""
    m = len(x)
    J = []
    for a in range(1, m):
        fila = []
        for b in range(1, m):
            if a == b:
                diagonal = 0
                for k in range(m):
                    if k != a:
                        diagonal = diagonal + P[a][k] * x[k] * lib.cos(phi[a] - phi[k])
                fila.append(diagonal)
            else:
                fila.append(-P[a][b] * x[b] * lib.cos(phi[a] - phi[b]))
        J.append(fila)
    return J


def base_perpendicular(m):
    """Base ortonormal (m × (m−1)) del ortogonal de (1, …, 1)."""
    Q, _ = np.linalg.qr(np.eye(m) - 1.0 / m)
    return Q[:, : m - 1]


def analizar(x, phi, P, lib=np, dps=60):
    """μ, max|F|, min c y autovalores de L fuera de la rotación global, en float (np) o alta precisión (mp).

    Con mp se trabaja con dps dígitos y se usa M = X^{-1/2} L X^{-1/2}, cuyo autovalor 0 corresponde
    a la rotación y se descarta.
    """
    if lib is mp:
        with mp.workdps(dps):
            return _analizar(x, phi, P, lib)
    return _analizar(x, phi, P, lib)


def _analizar(x, phi, P, lib):
    """Cuerpo de analizar, con la precisión ya fijada."""
    d = grados(x, P)
    F = fuerzas(x, phi, P, lib)
    c = splitting(x, phi, P, lib)
    L = matriz_clusters(x, phi, P, lib)
    if lib is np:
        Q = base_perpendicular(len(x))
        autovalores = list(np.linalg.eigvalsh(Q.T @ np.array(L, dtype=float) @ Q))
    else:
        m = len(x)
        raices = [mp.sqrt(xa) for xa in x]
        M = mp.matrix(m, m)
        for a in range(m):
            for b in range(m):
                M[a, b] = L[a][b] / (raices[a] * raices[b])
        autovalores = sorted(mp.eigsy(M)[0])
        indice_rotacion = min(range(m), key=lambda i: abs(autovalores[i]))
        autovalores.pop(indice_rotacion)
    return {
        "mu": min(d),
        "max|F|": max(abs(f) for f in F),
        "min c": min(c),
        "autovalores L": autovalores,
    }


@dataclass
class Plantilla:
    """Plantilla (x, φ, P) en float, con sus cantidades y el test de estrictez."""

    x: np.ndarray
    phi: np.ndarray
    P: np.ndarray
    nombre: str = ""

    def __post_init__(self):
        """Convierte a arrays de numpy y valida simetría de P y Σ x = 1."""
        self.x = np.asarray(self.x, dtype=float)
        self.phi = np.asarray(self.phi, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        assert np.allclose(self.P, self.P.T), "P tiene que ser simétrica"
        assert abs(self.x.sum() - 1) < 1e-12, "Σ x_a tiene que ser 1"

    @property
    def m(self):
        """Cantidad de clusters."""
        return len(self.x)

    def grados(self):
        """d_a para cada cluster."""
        return np.array(grados(self.x, self.P))

    def mu(self):
        """μ(T) = min_a d_a."""
        return self.grados().min()

    def fuerzas(self):
        """F_a para cada cluster (cero en equilibrio)."""
        return np.array(fuerzas(self.x, self.phi, self.P))

    def splitting(self):
        """c_a para cada cluster."""
        return np.array(splitting(self.x, self.phi, self.P))

    def matriz_clusters(self):
        """Matriz de L(y) como array."""
        return np.array(matriz_clusters(self.x, self.phi, self.P))

    def autovalores_clusters(self):
        """Autovalores de L en el ortogonal de 1 (sin el 0 de la rotación global)."""
        Q = base_perpendicular(self.m)
        return np.linalg.eigvalsh(Q.T @ self.matriz_clusters() @ Q)

    def reporte(self):
        """μ, max|F|, min c y el menor autovalor de L fuera de la rotación."""
        resultado = analizar(self.x, self.phi, self.P)
        resultado["min autovalor L"] = min(resultado.pop("autovalores L"))
        return resultado

    def es_estricta(self, tol=1e-12):
        """¿Equilibrio, c > 0 y L ≻ 0 fuera de la rotación, con tolerancia tol?"""
        r = self.reporte()
        return r["max|F|"] < tol and r["min c"] > tol and r["min autovalor L"] > tol


# biblioteca
def realizar(T, g, rng=None, tipo="aleatorio"):
    """Blow-up de una plantilla con clusters de igual tamaño en anillos Z_g (Fig. 4 de O'Keeffe).

    El vértice (a, i) se une con (b, j) si j − i ∈ S_ab, con |S_ab| = P_ab·g (tiene que ser entero),
    S_ba = −S_ab y S_aa simétrico sin el 0. Cada vértice de a tiene exactamente P_ab·g vecinos en b,
    así que las fases de la plantilla son un equilibrio exacto. Con tipo="intervalo",
    S_ab = {0, …, k−1}, como el k-spinning de Canale. Devuelve (W, theta).
    """
    assert np.allclose(T.x, 1 / T.m), "realizar solo soporta clusters de igual tamaño"
    rng = np.random.default_rng(rng)
    m = T.m
    W = np.zeros((m * g, m * g))
    for a in range(m):
        for b in range(a, m):
            k = T.P[a, b] * g
            assert abs(k - round(k)) < 1e-9, f"P[{a},{b}]·g = {k} no es entero"
            k = int(round(k))
            if a == b:
                k = min(k, g - 1)  # P_aa = 1: clique, g − 1 vecinos (sin self-loop)
                S = _conjunto_simetrico(g, k, rng, tipo)
            elif tipo == "intervalo":
                S = list(range(k))
            else:
                S = list(rng.choice(g, size=k, replace=False))
            for i in range(g):
                for s in S:
                    origen = a * g + i
                    destino = b * g + (i + s) % g
                    W[origen, destino] = 1
                    W[destino, origen] = 1
    theta = np.repeat(T.phi, g)
    return W, theta


def _conjunto_simetrico(g, k, rng, tipo):
    """Subconjunto simétrico de Z_g sin el 0, de tamaño k (pares {s, −s} más g/2 si k es impar)."""
    assert k <= g - 1
    pares = [(s, g - s) for s in range(1, (g + 1) // 2)]
    if k % 2 == 1:
        assert g % 2 == 0, "con g impar el tamaño de S_aa tiene que ser par"
    cantidad_pares = k // 2
    if tipo == "intervalo":
        elegidos = pares[:cantidad_pares]
    else:
        indices = rng.choice(len(pares), cantidad_pares, replace=False)
        elegidos = [pares[i] for i in indices]
    S = []
    for par in elegidos:
        S.extend(par)
    if k % 2 == 1:
        S.append(g // 2)
    return S


# biblioteca
def plantilla_c4():
    """Estado de 4 clusters de Kassabov–Strogatz–Townsend: μ = 3/4, L ≡ 0 (degenerado)."""
    P = np.array([
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
    ], dtype=float)
    return Plantilla(np.full(4, 0.25), np.arange(4) * np.pi / 2, P, "C4")


def plantilla_canale(a=1.0, b=1.0, c=1.0, d=0.5, interna=1.0):
    """Plantilla de Canale (arXiv 2209.06362, sec. V): 4 pares de clusters (i, ±) en πi/2 ± β/2.

    Los conjuntos E_t del paper tienen erratas; usamos la geometría de su Fig. 2 y del Lema 1:
      c: (i,+)–(i,−)       diferencia β          a: (i,+)–(i+1,−)   diferencia α
      b: (i,s)–(i+1,s)     diferencia π/2        d: (i,−)–(i+1,+)   diferencia α + 2β
    con α + β = π/2 y equilibrio (a − d) tan α = c. Estable si c² > 2ad.
    Con a = b = c = 1 y d → 1/2⁻, μ → 11/16. Índice del cluster (i, s): 2i + (s == +1).
    """
    def indice(i, s):
        """Índice del cluster (i, s) en la plantilla."""
        return 2 * (i % 4) + (s == 1)

    P = np.zeros((8, 8))
    for i in range(4):
        P[indice(i, 1), indice(i, -1)] = c
        P[indice(i, 1), indice(i + 1, -1)] = a
        P[indice(i, 1), indice(i + 1, 1)] = b
        P[indice(i, -1), indice(i + 1, -1)] = b
        P[indice(i, -1), indice(i + 1, 1)] = d
    P = P + P.T
    np.fill_diagonal(P, interna)
    alpha = np.arctan2(c, a - d)
    beta = np.pi / 2 - alpha
    phi = np.zeros(8)
    for k in range(8):
        signo = 1 if k % 2 else -1
        phi[k] = np.pi * (k // 2) / 2 + signo * beta / 2
    return Plantilla(np.full(8, 1 / 8), phi, P, f"Canale (a,b,c,d)=({a},{b},{c},{d})")


# biblioteca
def twisted_lp(m, q=1, margen=1e-9):
    """Mayor μ de una plantilla twisted con m clusters y q vueltas (0 si es infactible)."""
    distancias = np.arange(1, m // 2 + 1)
    # j y −j son la misma distancia salvo j = m/2, que aparece una sola vez
    multiplicidad = np.full(len(distancias), 2)
    if m % 2 == 0:
        multiplicidad[-1] = 1
    paso = 2 * np.pi * q / m
    cosenos = multiplicidad * np.cos(distancias * paso)

    filas, cotas = [], []
    for k in range(1, m):
        autovalor_k = cosenos * (2 - 2 * np.cos(2 * np.pi * distancias * k / m)) / m**2
        filas.append(-autovalor_k)
        cotas.append(-margen)
    # splitting: c_a = (1 + Σ_j p_j cos(j·paso))/m > 0
    filas.append(-cosenos / m)
    cotas.append(1 / m - margen)

    objetivo = -multiplicidad / m
    limites = [(0, 1)] * len(distancias)
    resultado = linprog(objetivo, A_ub=filas, b_ub=cotas, bounds=limites, method="highs")
    if not resultado.success:
        return 0.0
    return 1 / m - resultado.fun


def mejor_twisted(m):
    """Mejor μ sobre q para m clusters, sin q = m/4 (que es el estado degenerado C4)."""
    mejor = 0.0
    for q in range(1, (m - 1) // 4 + 1):
        if 4 * q == m:
            continue
        mejor = max(mejor, twisted_lp(m, q))
    return mejor


# biblioteca
INDICE = {nombre: i for i, nombre in enumerate(NOMBRES)}


CUARTOS = [0, 0, 1, 1, 2, 2, 3]


CAMINO = [("Am", "Bm"), ("Bm", "Cm"), ("Cm", "W"), ("W", "Cp"), ("Cp", "Bp"), ("Bp", "Ap")]


def f_e(q, w):
    """Las dos condiciones de neutralidad del modelo límite (Apéndice A, (I5))."""
    f = -2 * q**2 * w**2 - 4 * q**2 * w - 3 * q**2 - 2 * q + 4 * w + 1
    e = -q**2 * w**2 + q**2 - 2 * q * w**2 + 6 * q - w**2 + 1
    return f, e


def u13(q, w):
    """Parámetro u13(q, w) del Apéndice A, ec. (A3)."""
    return (2 * q**2 * w**2 + 2 * q**2 * w - q**2 - 6 * q + 4 * w**2 - 2 * w + 3) / 2


def polinomio_3(q):
    """Polinomio (3) de O'Keeffe, cuya única raíz en (0, 1) es q*."""
    return (9 * q**8 + 72 * q**7 + 288 * q**6 + 448 * q**5 + 470 * q**4
            + 88 * q**3 - 208 * q**2 - 128 * q - 15)


def esquina(dps=50):
    """(q*, w*, μ0): punto donde los dos dipolos del modelo límite son exactamente neutros."""
    with mp.workdps(dps):
        q, w = mp.findroot(f_e, (mp.mpf("0.65"), mp.mpf("1.4")))
        return q, w, (4 + 2 * q) / (7 + q)


def matriz_cero(m=7):
    """Matriz m × m de ceros como lista de listas."""
    return [[mp.mpf(0)] * m for _ in range(m)]


def _poner(P, nombre1, nombre2, valor):
    """Escribe un valor simétrico en P para el par de clusters (nombre1, nombre2)."""
    P[INDICE[nombre1]][INDICE[nombre2]] = valor
    P[INDICE[nombre2]][INDICE[nombre1]] = valor


def patron_limite(q):
    """Patrón límite de la Tabla II: 1 completo, q adelgazado, 0 vacío."""
    filas = [
        [1, 1, 1, q, 0, 0, 1],
        [1, 1, q, 1, 0, 0, 1],
        [1, q, 1, 1, 1, q, 0],
        [q, 1, 1, 1, q, 1, 0],
        [0, 0, 1, q, 1, 1, 1],
        [0, 0, q, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1],
    ]
    P = []
    for fila in filas:
        P.append([mp.mpf(v) for v in fila])
    return P


def primer_orden(q, w, a, xW, kappa=mp.mpf(19) / 10):
    """Corrección de primer orden p′ (Apéndice A, después de la ec. (A1))."""
    u = u13(q, w)
    A1 = w * (w + 1) - u / 2
    A2 = u / 2 - w * (w - 1)
    P = matriz_cero()
    _poner(P, "Am", "Bm", A1 - kappa)
    _poner(P, "Bp", "Cp", A1 - kappa)
    _poner(P, "Ap", "Bm", A2 - kappa)
    _poner(P, "Bp", "Cm", A2 - kappa)
    for nombre1, nombre2 in [("Ap", "Bp"), ("Bm", "Cm"), ("Am", "Bp"), ("Bm", "Cp")]:
        _poner(P, nombre1, nombre2, -kappa)
    _poner(P, "Am", "W", a / xW * (A1 + 2 - 2 * kappa))
    _poner(P, "Cp", "W", a / xW * (A1 + 2 - 2 * kappa))
    _poner(P, "Ap", "W", a / xW * (A2 - 2 - 2 * kappa))
    _poner(P, "Cm", "W", a / xW * (A2 - 2 - 2 * kappa))
    return P


def _corregir_equilibrio(x, phi, P):
    """Resuelve Δ sobre el CAMINO para que F = 0 exacto (F es afín en P).

    Se descarta la ecuación del cluster W, que se deduce de Σ x_a F_a = 0. Devuelve Δ y P corregida.
    """
    F_actual = fuerzas(x, phi, P, mp)
    columnas = []
    for nombre1, nombre2 in CAMINO:
        E = matriz_cero()
        _poner(E, nombre1, nombre2, mp.mpf(1))
        columnas.append(fuerzas(x, phi, E, mp))
    J = mp.matrix(6, 6)
    for fila in range(6):
        for columna in range(6):
            J[fila, columna] = columnas[columna][fila]
    lado_derecho = mp.matrix([-F_actual[fila] for fila in range(6)])
    delta = mp.lu_solve(J, lado_derecho)
    for k, (nombre1, nombre2) in enumerate(CAMINO):
        i, j = INDICE[nombre1], INDICE[nombre2]
        _poner(P, nombre1, nombre2, P[i][j] + delta[k])
    return delta, P


def construir(t, h, dps=60, corregir=True):
    """Plantilla S7 para (t, h): dict con x, phi, P (mpmath, listas), q, w, F_primer_orden y delta."""
    with mp.workdps(dps):
        t = mp.mpf(t)
        h = mp.mpf(h)
        q_estrella, w, _ = esquina(dps)
        q = q_estrella - t
        a = 1 / (7 + q)
        xW = 1 - 6 * a
        x = [a] * 6 + [xW]
        desplazamientos = [-1, 1, -w, w, -1, 1, 0]
        phi = []
        for c in range(7):
            phi.append(CUARTOS[c] * mp.pi / 2 + h * desplazamientos[c])

        limite = patron_limite(q)
        correccion = primer_orden(q, w, a, xW)
        P = matriz_cero()
        for i in range(7):
            for j in range(7):
                P[i][j] = limite[i][j] + h * correccion[i][j]

        F_primer_orden = fuerzas(x, phi, P, mp)
        delta = None
        if corregir:
            delta, P = _corregir_equilibrio(x, phi, P)
        return {"x": x, "phi": phi, "P": P, "q": q, "w": w, "t": t, "h": h,
                "F_primer_orden": F_primer_orden, "delta": delta}


# biblioteca
def racionalizar(x, P, D, diagonal_maxima=True):
    """Redondea (x, P) a denominador común D (par), con Σ x = 1 exacto ajustando el último cluster.

    Si diagonal_maxima, las diagonales quedan en (D − 1)/D (clique completo, como pide el Lema 2).
    """
    assert D % 2 == 0
    m = len(x)
    with mp.workdps(len(str(D)) + 30):  # precisión suficiente para redondear a 1/D
        X = []
        for xa in x[:-1]:
            X.append(int(mp.nint(xa * D)))
        X.append(D - sum(X))
        Q = [[0] * m for _ in range(m)]
        for a in range(m):
            for b in range(a, m):
                if a == b:
                    valor = D - 1 if diagonal_maxima else int(mp.floor(P[a][b] * D))
                else:
                    valor = int(mp.nint(P[a][b] * D))
                tope = D - 1 if a == b else D
                valor = max(0, min(tope, valor))
                Q[a][b] = valor
                Q[b][a] = valor
    xq = [Fraction(Xa, D) for Xa in X]
    Pq = []
    for fila in Q:
        Pq.append([Fraction(valor, D) for valor in fila])
    return xq, Pq


def _a_mpf(fraccion):
    """Fraction → mpf."""
    return mp.mpf(fraccion.numerator) / fraccion.denominator


def _a_intervalo(fraccion):
    """Fraction → intervalo de mpmath.iv que la contiene."""
    return iv.mpf(fraccion.numerator) / fraccion.denominator


def fases_newton(xq, Pq, phi0, dps=120, iteraciones=60):
    """Refina las fases por Newton (φ_0 fijo) para los datos racionales, en mpmath de alta precisión."""
    with mp.workdps(dps):
        x = [_a_mpf(v) for v in xq]
        P = []
        for fila in Pq:
            P.append([_a_mpf(v) for v in fila])
        phi = [mp.mpf(p) for p in phi0]
        m = len(x)
        for _ in range(iteraciones):
            F = mp.matrix(fuerzas(x, phi, P, mp)[1:])
            if mp.norm(F) < mp.mpf(10) ** (-dps + 10):
                break
            J = mp.matrix(jacobiano_fuerzas(x, phi, P, mp))
            paso = mp.lu_solve(J, F)
            for a in range(1, m):
                phi[a] -= paso[a - 1]
        return phi


def _cota_inferior(intervalo):
    """Extremo inferior de un intervalo de mpmath.iv, como mpf."""
    return mp.mpf(intervalo.a.a)


def _krawczyk(x, P, centro, caja):
    """¿K(X) = m − Y F(m) + (I − Y J(X))(X − m) cae en el interior de la caja X?"""
    n = len(caja) - 1
    F_centro = fuerzas(x, centro, P, iv)[1:]
    J_centro = jacobiano_fuerzas(x, centro, P, iv)
    J_punto = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            J_punto[i, j] = mp.mpf(J_centro[i][j].mid)
    Y = J_punto ** -1
    J_caja = jacobiano_fuerzas(x, caja, P, iv)

    for i in range(n):
        K_i = centro[i + 1]
        for k in range(n):
            K_i = K_i - iv.mpf(Y[i, k]) * F_centro[k]
        for j in range(n):
            coeficiente = 1 if i == j else 0
            for k in range(n):
                coeficiente = coeficiente - iv.mpf(Y[i, k]) * J_caja[k][j]
            K_i = K_i + coeficiente * (caja[j + 1] - centro[j + 1])
        dentro = K_i.a > caja[i + 1].a and K_i.b < caja[i + 1].b
        if not dentro:
            return False
    return True


def _cholesky_intervalos(A):
    """Cholesky de una matriz de intervalos; devuelve (éxito, cota inferior del menor pivote)."""
    n = len(A)
    L = [[iv.mpf(0)] * n for _ in range(n)]
    menor_pivote = None
    for j in range(n):
        pivote = A[j][j]
        for k in range(j):
            pivote = pivote - L[j][k] ** 2
        cota = _cota_inferior(pivote)
        menor_pivote = cota if menor_pivote is None else min(menor_pivote, cota)
        if not cota > 0:
            return False, menor_pivote
        L[j][j] = iv.sqrt(pivote)
        for i in range(j + 1, n):
            suma = A[i][j]
            for k in range(j):
                suma = suma - L[i][k] * L[j][k]
            L[i][j] = suma / L[j][j]
    return True, menor_pivote


def _separaciones(caja):
    """Cotas inferiores de |sin((φ_a − φ_b)/2)| sobre la caja, para todos los pares a < b."""
    m = len(caja)
    cotas = []
    for a in range(m):
        for b in range(a + 1, m):
            cotas.append(_cota_inferior(abs(iv.sin((caja[a] - caja[b]) / 2))))
    return cotas


def certificar(xq, Pq, phi_centro, radio="1e-40", dps=100, verbose=True):
    """Corre todos los pasos del certificado. Devuelve un dict; ok = True solo si todos pasan.

    La clave "caja" tiene la caja de fases (intervalos) donde Krawczyk garantiza la solución.
    """
    iv.dps = dps
    mp.mp.dps = dps
    m = len(xq)
    x = [_a_intervalo(v) for v in xq]
    P = []
    for fila in Pq:
        P.append([_a_intervalo(v) for v in fila])
    r = mp.mpf(radio)
    centro_mp = [mp.mpf(p) for p in phi_centro]
    centro = [iv.mpf(p) for p in centro_mp]
    caja = [iv.mpf(centro_mp[0])]
    for a in range(1, m):
        caja.append(iv.mpf([centro_mp[a] - r, centro_mp[a] + r]))

    resultado = {}
    resultado["krawczyk"] = _krawczyk(x, P, centro, caja)

    c = splitting(x, caja, P, iv)
    resultado["min c (cota inf)"] = min(_cota_inferior(ca) for ca in c)
    resultado["c>0"] = resultado["min c (cota inf)"] > 0

    L = matriz_clusters(x, caja, P, iv)
    L_sin_rotacion = [fila[1:] for fila in L[1:]]
    exito, menor_pivote = _cholesky_intervalos(L_sin_rotacion)
    resultado["cholesky"] = exito
    resultado["min pivote (cota inf)"] = menor_pivote

    separaciones = _separaciones(caja)
    resultado["min |sin(Δφ/2)| (cota inf)"] = min(separaciones)
    resultado["distintas"] = resultado["min |sin(Δφ/2)| (cota inf)"] > 0
    resultado["no sync"] = max(separaciones) > 0

    resultado["mu"] = min(grados(xq, Pq))
    resultado["caja"] = caja
    resultado["ok"] = (resultado["krawczyk"] and resultado["c>0"]
                       and resultado["cholesky"] and resultado["no sync"])
    if verbose:
        imprimir_certificado(resultado)
    return resultado


def imprimir_certificado(resultado):
    """Imprime cada paso del certificado y el μ exacto."""
    for clave, valor in resultado.items():
        if clave in ("mu", "caja"):
            continue
        texto = valor if isinstance(valor, bool) else mp.nstr(valor, 6)
        print(f"  {clave:28s} {texto}")
    mu = resultado["mu"]
    bits = mu.denominator.bit_length()
    print(f"  {'mu (exacto)':28s} {float(mu):.16f}  (denominador {bits} bits)")


# biblioteca
def sdp_interno(x, phi, escala=1.0, margen=1e-6, margen_c=1e-3, prohibidos=None, solver="CLARABEL"):
    """Mayor μ sobre las densidades P con x y φ fijos. Devuelve (μ, P) o (None, None) si es infactible."""
    m = len(x)
    diferencias = phi[:, None] - phi[None, :]
    senos = np.sin(diferencias)
    cosenos = np.cos(diferencias)
    pesos_L = np.outer(x, x) * cosenos
    np.fill_diagonal(pesos_L, 0)

    P = cp.Variable((m, m), symmetric=True)
    t = cp.Variable()
    K = cp.multiply(P, pesos_L)
    L = cp.diag(cp.sum(K, axis=1)) - K
    Q = base_perpendicular(m)
    restricciones = [
        P >= 0,
        P <= 1,
        cp.diag(P) == 1,
        P @ x >= t,
        cp.multiply(P, senos) @ x == 0,
        cp.multiply(P, cosenos) @ x >= margen_c,
        (Q.T @ L @ Q) / escala >> margen * np.eye(m - 1),
    ]
    if prohibidos is not None:
        restricciones.append(cp.multiply(P, prohibidos.astype(float)) == 0)
    problema = cp.Problem(cp.Maximize(t), restricciones)
    try:
        problema.solve(solver=solver)
    except cp.error.SolverError:
        return None, None
    if problema.status not in ("optimal", "optimal_inaccurate"):
        return None, None
    return float(t.value), np.array(P.value)


@dataclass
class Base:
    """Estado de sectores sobre el que se parten clusters.

    fases: fase de cada sector. tamanos: tamaño total de cada sector (para los arranques).
    prohibidos: pares de sectores sin links. radios: amplitud inicial de los desplazamientos por sector.
    """

    nombre: str
    fases: np.ndarray
    tamanos: np.ndarray
    prohibidos: np.ndarray
    radios: np.ndarray


def base_c4(q=0.6505691669447988, w=1.3982781340204612):
    """C4 con los tamaños de S7 (2a, 2a, 2a, 1 − 6a) y radios (1, w*, 1, 1); opuestos sin links."""
    a = 1 / (7 + q)
    T = plantilla_c4()
    return Base(
        nombre="C4",
        fases=T.phi,
        tamanos=np.array([2 * a, 2 * a, 2 * a, 1 - 6 * a]),
        prohibidos=(T.P == 0),
        radios=np.array([1.0, w, 1.0, 1.0]),
    )


def base_canale():
    """Canale en el borde d = 1/2 (8 sectores iguales); sin links donde Canale no los tiene."""
    T = plantilla_canale(d=0.5)
    return Base(
        nombre="Canale",
        fases=T.phi,
        tamanos=T.x,
        prohibidos=(T.P == 0),
        radios=np.ones(8),
    )


class FamiliaParticiones:
    """Plantillas que parten cada sector j de una base en piezas[j] piezas, separadas por h·s."""

    def __init__(self, base, piezas, h):
        """Arma la familia: a qué sector pertenece cada pieza y qué pares quedan prohibidos."""
        assert len(piezas) == len(base.fases), "un número de piezas por sector"
        self.base = base
        self.piezas = list(piezas)
        self.h = h
        self.sector = np.repeat(np.arange(len(piezas)), self.piezas)
        self.m = len(self.sector)
        self.partida = np.array([self.piezas[j] > 1 for j in self.sector])
        self.prohibidos = base.prohibidos[np.ix_(self.sector, self.sector)]

    def cantidad_parametros(self):
        """Largo de z: un logit por pieza más un desplazamiento por pieza partida."""
        return self.m + int(self.partida.sum())

    def decodificar(self, z):
        """z = (logits de tamaños, desplazamientos de las piezas partidas) → (x, φ)."""
        logits = z[: self.m]
        desplazamientos_libres = z[self.m:]
        x = np.exp(logits - logits.max())
        x = x / x.sum()
        desplazamientos = np.zeros(self.m)
        desplazamientos[self.partida] = desplazamientos_libres
        phi = self.base.fases[self.sector] + self.h * desplazamientos
        return x, phi

    def desplazamientos(self, phi):
        """Inversa parcial de decodificar: s = (φ − fase del sector)/h."""
        return (phi - self.base.fases[self.sector]) / self.h

    def mu(self, z, margen=1e-6):
        """μ del SDP interno para los parámetros z (−1 si es infactible o hay piezas vacías)."""
        x, phi = self.decodificar(z)
        if np.min(x) < 1e-3:
            return -1.0
        mu, _ = sdp_interno(x, phi, escala=self.h**2, margen=margen, prohibidos=self.prohibidos)
        return -1.0 if mu is None else mu


def arranque_estructurado(familia, rng=None, ruido=0.0):
    """Tamaños de la base repartidos en partes iguales y desplazamientos equiespaciados en [−r_j, r_j].

    Con la base C4 y el patrón (2,2,2,1) es el ansatz de S7. Con ruido > 0 sirve para reinicios.
    """
    rng = np.random.default_rng(rng)
    tamanos = np.repeat(familia.base.tamanos / np.array(familia.piezas), familia.piezas)
    desplazamientos = []
    for j, k in enumerate(familia.piezas):
        if k > 1:
            r = familia.base.radios[j]
            desplazamientos.extend(np.linspace(-r, r, k))
    z = np.concatenate([np.log(tamanos), np.array(desplazamientos)])
    return z + ruido * rng.standard_normal(len(z))


def arranque_aleatorio(familia, rng):
    """Tamaños casi iguales por sector y desplazamientos al azar en [−1.5, 1.5]."""
    sectores = len(familia.piezas)
    tamanos = np.repeat(1 / sectores / np.array(familia.piezas), familia.piezas)
    logits = np.log(tamanos) + 0.1 * rng.standard_normal(familia.m)
    desplazamientos = []
    for k in familia.piezas:
        if k > 1:
            desplazamientos.extend(np.sort(rng.uniform(-1.5, 1.5, k)))
    return np.concatenate([logits, np.array(desplazamientos)])


def optimizar_familia(familia, z0, margen=1e-6, max_iteraciones=400):
    """Nelder–Mead sobre z maximizando μ desde z0. Devuelve (mejor μ, mejor z)."""
    mejor = (familia.mu(z0, margen), z0)
    opciones = {"maxiter": max_iteraciones, "xatol": 1e-4, "fatol": 1e-7, "adaptive": True}
    resultado = minimize(lambda z: -familia.mu(z, margen), z0, method="Nelder-Mead", options=opciones)
    if -resultado.fun > mejor[0]:
        mejor = (-resultado.fun, resultado.x)
    return mejor


# biblioteca
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
        usar_estructurado = arranque == "estructurado" or (arranque == "mixto" and reinicio % 2 == 0)
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
        "arranque": arranque,
        "mejor_mu": max(mus),
        "todos_mu": mus,
        "x": x.tolist(),
        "s": familia.desplazamientos(phi).tolist(),
        "segundos": time.time() - inicio,
    }


# biblioteca
def grafos_conexos(k):
    """Genera (graph6, matriz de adyacencia) de todos los grafos conexos con k vértices (atlas de networkx)."""
    for G in nx.graph_atlas_g():
        if G.number_of_nodes() == k and nx.is_connected(G):
            codigo = nx.to_graph6_bytes(G, header=False).decode().strip()
            yield codigo, nx.to_numpy_array(G)


def mu_sin_estabilidad(M):
    """LP: max_x min_a (M x)_a con x en el símplex. Devuelve (valor, x óptimo)."""
    k = len(M)
    objetivo = np.zeros(k + 1)
    objetivo[-1] = -1
    desigualdades = np.hstack([-M, np.ones((k, 1))])
    suma_uno = [list(np.ones(k)) + [0]]
    resultado = linprog(objetivo, A_ub=desigualdades, b_ub=np.zeros(k), A_eq=suma_uno, b_eq=[1],
                        bounds=[(0, 1)] * (k + 1), method="highs")
    return -resultado.fun, resultado.x[:k]


class Restricciones:
    """Desempaqueta v = (logits de x, φ_1..φ_{k−1}, t) y evalúa las restricciones de una plantilla 0/1."""

    def __init__(self, M, margen, margen_c):
        """Guarda M = A + I, la base de 1^⊥ y los márgenes."""
        self.M = M
        self.k = len(M)
        self.Q = base_perpendicular(self.k)
        self.margen = margen
        self.margen_c = margen_c

    def desempaquetar(self, v):
        """v → (x en el símplex, φ con φ_0 = 0)."""
        k = self.k
        x = np.exp(v[:k] - v[:k].max())
        x = x / x.sum()
        phi = np.concatenate([[0.0], v[k: 2 * k - 1]])
        return x, phi

    def grado_menos_t(self, v):
        """d_a − t (tiene que ser ≥ 0)."""
        x, _ = self.desempaquetar(v)
        return self.M @ x - v[-1]

    def equilibrio(self, v):
        """F_a para a ≥ 1 (tiene que ser 0)."""
        x, phi = self.desempaquetar(v)
        return np.array(fuerzas(x, phi, self.M))[1:]

    def splitting_menos_margen(self, v):
        """c_a − margen_c (tiene que ser ≥ 0)."""
        x, phi = self.desempaquetar(v)
        return np.array(splitting(x, phi, self.M)) - self.margen_c

    def autovalor_min(self, v):
        """Menor autovalor de L en 1^⊥ (sin la rotación global)."""
        x, phi = self.desempaquetar(v)
        L = np.array(matriz_clusters(x, phi, self.M))
        return np.linalg.eigvalsh(self.Q.T @ L @ self.Q).min()

    def para_slsqp(self):
        """Las restricciones en el formato de scipy.optimize.minimize."""
        return [
            {"type": "ineq", "fun": self.grado_menos_t},
            {"type": "eq", "fun": self.equilibrio},
            {"type": "ineq", "fun": self.splitting_menos_margen},
            {"type": "ineq", "fun": lambda v: np.array([self.autovalor_min(v) - self.margen])},
        ]


def optimizar_grafo(M, arranques, rng, margen=1e-6, margen_c=1e-3):
    """Mejor estado estricto no sync del blow-up completo de M = A + I, o None si no se encontró."""
    k = len(M)
    restricciones = Restricciones(M, margen, margen_c)
    _, x_lp = mu_sin_estabilidad(M)
    mejor = None
    for _ in range(arranques):
        x0 = 0.5 * x_lp + 0.5 / k + 0.02 * rng.random(k)
        v0 = np.concatenate([np.log(x0), rng.uniform(-np.pi, np.pi, k - 1), [0.5]])
        try:
            resultado = minimize(lambda v: -v[-1], v0, constraints=restricciones.para_slsqp(),
                                 method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
        except Exception:
            continue
        v = resultado.x
        x, phi = restricciones.desempaquetar(v)
        autovalor = restricciones.autovalor_min(v)
        agrupamiento = np.abs(np.exp(1j * phi).mean())
        valido = (np.abs(restricciones.equilibrio(v)).max() < 1e-8
                  and restricciones.splitting_menos_margen(v).min() > -margen_c / 2
                  and autovalor > margen / 2
                  and agrupamiento < 1 - 1e-6)
        if not valido:
            continue
        mu = (M @ x).min()
        if mejor is None or mu > mejor["mu"]:
            mejor = {"mu": float(mu), "x": x.tolist(), "phi": phi.tolist(), "autovalor_min": float(autovalor)}
    return mejor


def optimizar_trabajo(trabajo):
    """Trabajo en paralelo: (graph6, arranques, semilla) → (graph6, mejor estado o None)."""
    codigo, arranques, semilla_rng = trabajo
    G = nx.from_graph6_bytes(codigo.encode())
    M = nx.to_numpy_array(G) + np.eye(G.number_of_nodes())
    return codigo, optimizar_grafo(M, arranques, np.random.default_rng(semilla_rng))


# biblioteca
def semilla(x, phi):
    """μ de borde (L ⪰ 0, sin brecha) y cantidad de modos de cluster neutros en el óptimo."""
    mu, P = sdp_interno(x, phi, margen=0.0, margen_c=1e-6)
    if mu is None:
        return None, None
    T = Plantilla(x, phi, (P + P.T) / 2)
    neutros = int((T.autovalores_clusters() < 1e-7).sum())
    return mu, neutros


def una_red(l):
    """ℓ clusters iguales equiespaciados."""
    return np.full(l, 1 / l), 2 * np.pi * np.arange(l) / l


def dos_redes(l, r, beta):
    """Dos redes de ℓ puntos rotadas β entre sí, con pesos totales r y 1 − r."""
    red = 2 * np.pi * np.arange(l) / l
    x = np.concatenate([np.full(l, r / l), np.full(l, (1 - r) / l)])
    phi = np.concatenate([red, red + beta])
    return x, phi


def barrer_una_red(ls):
    """Semilla de cada red de ℓ clusters."""
    resultados = []
    for l in ls:
        mu, neutros = semilla(*una_red(l))
        resultados.append({"l": l, "mu": mu, "neutros": neutros})
    return resultados


def barrer_dos_redes(ls, rs, cantidad_betas):
    """Mejores semillas de dos redes rotadas, sobre una grilla de β y pesos r."""
    resultados = []
    for l in ls:
        betas = np.linspace(0.02, 2 * np.pi / l - 0.02, cantidad_betas)
        for r in rs:
            for beta in betas:
                mu, neutros = semilla(*dos_redes(l, r, beta))
                if mu is not None:
                    resultados.append({"l": l, "r": r, "beta_grados": float(np.degrees(beta)),
                                       "mu": mu, "neutros": neutros})
    resultados.sort(key=lambda fila: -fila["mu"])
    return resultados


# biblioteca
SECTORES = [0, 0, 1, 1, 2, 2, 3]
K_S7 = [3] * 6 + [5]                     # tamaños (3, 5): x = (3, …, 3, 5)/23


def fases_s7(h, s):
    """φ = fase del sector (múltiplo de π/2) + h·s; W queda en 3π/2."""
    desplazamientos = np.concatenate([s, [0.0]])
    return np.array(SECTORES) * np.pi / 2 + h * desplazamientos


def prohibidos_s7():
    """Pares sin links: sectores opuestos (A–C y B–W)."""
    prohibidos = np.zeros((7, 7), dtype=bool)
    for a in range(7):
        for b in range(7):
            if abs(SECTORES[a] - SECTORES[b]) == 2:
                prohibidos[a, b] = True
    return prohibidos


def tamanos_racionales(k_mitad, k_w):
    """x = (k_mitad, …, k_mitad, k_w)/(6 k_mitad + k_w) como array."""
    k = np.array([k_mitad] * 6 + [k_w], dtype=float)
    return k / k.sum()


def desplazamientos_iniciales():
    """Ansatz de S7: s = (−1, 1, −w*, w*, −1, 1) para las seis mitades."""
    _, w, _ = esquina(30)
    w = float(w)
    return np.array([-1.0, 1.0, -w, w, -1.0, 1.0])


def mejor_P(x, h, s, margen):
    """SDP interno con L/h² ⪰ margen: (μ, P) o (None, None)."""
    return sdp_interno(x, fases_s7(h, s), escala=h**2, margen=margen, prohibidos=prohibidos_s7())


def optimizar_s7(x, h, margen, s0=None, max_iteraciones=600):
    """Nelder–Mead en los desplazamientos s con x fijo. Devuelve (μ, s, P)."""
    if s0 is None:
        s0 = desplazamientos_iniciales()

    def menos_mu(s):
        mu, _ = mejor_P(x, h, s, margen)
        return 1.0 if mu is None else -mu

    opciones = {"maxiter": max_iteraciones, "xatol": 1e-5, "fatol": 1e-9, "adaptive": True}
    resultado = minimize(menos_mu, s0, method="Nelder-Mead", options=opciones)
    s = resultado.x if resultado.fun < menos_mu(s0) else np.asarray(s0)
    mu, P = mejor_P(x, h, s, margen)
    return mu, s, P


def autovalores_normalizados(x, phi, P):
    """Autovalores de X^{-1/2} L X^{-1/2} sin el 0 de la rotación (la misma normalización que analizar con mp)."""
    x = np.asarray(x, dtype=float)
    L = np.array(matriz_clusters(x, phi, P), dtype=float)
    raices = np.sqrt(x)
    autovalores = np.linalg.eigvalsh(L / np.outer(raices, raices))
    indice_rotacion = np.argmin(np.abs(autovalores))
    return np.sort(np.delete(autovalores, indice_rotacion))


def espectro_s7(x, phi, P):
    """Autovalores normalizados de L, min c y número de condición del Jacobiano del equilibrio."""
    J = np.array(jacobiano_fuerzas(x, phi, P))
    return {
        "autovalores": autovalores_normalizados(x, phi, P).tolist(),
        "min c": float(min(splitting(x, phi, P))),
        "condicion": float(np.linalg.cond(J)),
    }


# biblioteca
TAMANOS = [(3, 5), (11, 18), (20, 33)]


def un_punto(trabajo):
    """Optimiza S7 para (h, margen, tamaños) y mide su espectro y el redondeo ingenuo."""
    h, margen, (k_mitad, k_w), con_ingenuo = trabajo
    inicio = time.time()
    x = tamanos_racionales(k_mitad, k_w)
    mu, s, P = optimizar_s7(x, h, margen)
    fila = {"h": h, "margen": margen, "k": [k_mitad, k_w], "mu": mu}
    if mu is None:
        return fila
    phi = fases_s7(h, s)
    fila.update(espectro_s7(x, phi, P))
    fila["s"] = s.tolist()
    fila["P"] = np.asarray(P).tolist()
    if con_ingenuo:
        k = [k_mitad] * 6 + [k_w]
        x_exacto, phi_exacto, P_exacto = pulir(tamanos_exactos(k), phi, P)
        n_min, _ = n_minimo_ingenuo(x_exacto, P_exacto, phi_exacto, h, k, exponentes=np.arange(4, 16.01, 1))
        fila["n_min_ingenuo"] = n_min
    fila["segundos"] = time.time() - inicio
    return fila


# biblioteca
def reporte_plantilla(x, phi, P):
    """μ, max|F|, min c, autovalores de X^{-1/2} L X^{-1/2} (sin la rotación) y valores singulares de J."""
    x = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)
    P = np.asarray(P, dtype=float)
    raiz = 1 / np.sqrt(x)
    M = raiz[:, None] * np.array(matriz_clusters(x, phi, P)) * raiz[None, :]
    autovalores = np.linalg.eigvalsh(M)
    rotacion = int(np.argmin(np.abs(autovalores)))
    return {
        "mu": float((P @ x).min()),
        "max|F|": float(np.abs(fuerzas(x, phi, P)).max()),
        "min c": float(min(splitting(x, phi, P))),
        "autovalores L": np.delete(autovalores, rotacion),
        "singulares J": np.linalg.svd(np.array(jacobiano_fuerzas(x, phi, P)), compute_uv=False),
    }


def fila_curva(curva, h, margen, k=(3, 5)):
    """Fila de la curva con esos parámetros."""
    for fila in curva:
        if fila["h"] == h and fila["margen"] == margen and tuple(fila["k"]) == tuple(k):
            return fila
    raise KeyError((h, margen, k))


# biblioteca
def tamanos_exactos(k):
    """x_a = k_a/K como Fraction."""
    K = sum(int(ka) for ka in k)
    return [Fraction(int(ka), K) for ka in k]


def tamanos_enteros(k, g):
    """n_a = k_a·g."""
    return [int(ka) * g for ka in k]


def paso_grado(k, a, b):
    """d_ab tiene que ser múltiplo de k_b/mcd(k_a, k_b) para que n_a d_ab = n_b d_ba."""
    return int(k[b]) // gcd(int(k[a]), int(k[b]))


def grados_ingenuos(x, P, k, g):
    """Redondea P_ab·n_b al múltiplo permitido más cercano (para a < b) y completa por simetría."""
    m = len(k)
    n_a = tamanos_enteros(k, g)
    d = [[0] * m for _ in range(m)]
    for a in range(m):
        d[a][a] = n_a[a] - 1
        for b in range(a + 1, m):
            paso = paso_grado(k, a, b)
            j = int(mp.nint(P[a][b] * n_a[b] / paso))
            j = max(0, min(j, n_a[b] // paso))
            d[a][b] = j * paso
            d[b][a] = d[a][b] * n_a[a] // n_a[b]
    return d


def plantilla_entera(k, g, d):
    """(x, P) exactos como Fraction: x_a = n_a/n, P_ab = d_ab/n_b (con P_aa = (n_a − 1)/n_a)."""
    n_a = tamanos_enteros(k, g)
    n = sum(n_a)
    x = [Fraction(na, n) for na in n_a]
    P = []
    for a in range(len(k)):
        fila = []
        for b in range(len(k)):
            fila.append(Fraction(d[a][b], n_a[b]))
        P.append(fila)
    return x, P


def mu_entero(k, g, d):
    """μ = (δ + 1)/n exacto, con δ el grado mínimo."""
    n_a = tamanos_enteros(k, g)
    n = sum(n_a)
    grados = []
    for fila in d:
        grados.append(sum(fila))
    return Fraction(min(grados) + 1, n)


# biblioteca
def _a_mp(valor):
    """Fraction o float → mpf (exacto para Fraction)."""
    if isinstance(valor, Fraction):
        return mp.mpf(valor.numerator) / valor.denominator
    return mp.mpf(float(valor))


def pares_libres(P, tolerancia=1e-7):
    """Pares a < b con densidad estrictamente entre 0 y 1 (los demás son vacíos o completos)."""
    m = len(P)
    pares = []
    for a in range(m):
        for b in range(a + 1, m):
            if tolerancia < float(P[a][b]) < 1 - tolerancia:
                pares.append((a, b))
    return pares


def _ajustar_extremo(valor, tolerancia=1e-7):
    """Lleva a 0 o 1 exactos las densidades que el SDP dejó a menos de tolerancia de un extremo."""
    valor = float(valor)
    if valor < tolerancia:
        return mp.mpf(0)
    if valor > 1 - tolerancia:
        return mp.mpf(1)
    return mp.mpf(valor)


def pulir(x, phi, P, dps=60):
    """Corrige P con el menor cambio (mínima norma) para que F(φ) = 0 exacto en alta precisión.

    El SDP deja F ≈ 1e-9, y como el Jacobiano del equilibrio es casi singular eso movería las fases
    mucho más que el redondeo. F es lineal en P, así que basta un problema de mínimos cuadrados.
    Solo se tocan los pares a < b con 0 < P_ab < 1 (los completos y vacíos quedan fijos).
    """
    m = len(x)
    with mp.workdps(dps):
        pares = pares_libres(P)
        x_mp = [_a_mp(v) for v in x]
        phi_mp = [mp.mpf(float(v)) for v in phi]
        P_mp = []
        for fila in P:
            P_mp.append([_ajustar_extremo(v) for v in fila])
        # La ecuación del último cluster sobra: se deduce de Σ x_a F_a ≡ 0.
        A = mp.matrix(m - 1, len(pares))
        for j, (a, b) in enumerate(pares):
            seno = mp.sin(phi_mp[a] - phi_mp[b])
            if a < m - 1:
                A[a, j] = x_mp[b] * seno
            if b < m - 1:
                A[b, j] = -x_mp[a] * seno
        F = mp.matrix(m - 1, 1)
        for a in range(m - 1):
            for b in range(m):
                if b != a:
                    F[a] += P_mp[a][b] * x_mp[b] * mp.sin(phi_mp[a] - phi_mp[b])
        delta = A.T * mp.lu_solve(A * A.T, -F)
        for j, (a, b) in enumerate(pares):
            P_mp[a][b] += delta[j]
            P_mp[b][a] = P_mp[a][b]
        return x_mp, phi_mp, P_mp


def evaluar_enteros(k, g, d, phi0, h, dps=40, phi_referencia=None):
    """Resuelve el equilibrio exacto de los datos enteros y mira si es estricto (en alta precisión).

    Devuelve un dict con ok, el menor autovalor de L, min c y cuánto se movieron las fases. Se exige
    además que las fases no se alejen más de h/4 de phi_referencia (por defecto φ0), para seguir en
    la forma de S7.
    """
    if phi_referencia is None:
        phi_referencia = phi0
    x, P = plantilla_entera(k, g, d)
    try:
        phi = fases_newton(x, P, list(phi0), dps=dps)
    except ZeroDivisionError:
        return {"ok": False, "menor autovalor": -np.inf, "min c": np.nan, "movimiento fases": np.inf, "phi": None}
    with mp.workdps(dps):
        x_mp = [mp.mpf(v.numerator) / v.denominator for v in x]
        P_mp = []
        for fila in P:
            P_mp.append([mp.mpf(v.numerator) / v.denominator for v in fila])
        reporte = analizar(x_mp, phi, P_mp, lib=mp, dps=dps)
        movimiento = max(abs(a - b) for a, b in zip(phi, phi_referencia))
    menor = float(min(reporte["autovalores L"]))
    min_c = float(reporte["min c"])
    max_F = float(reporte["max|F|"])
    ok = menor > 0 and min_c > 0 and max_F < 1e-20 and float(movimiento) < h / 4
    return {"ok": ok, "menor autovalor": menor, "min c": min_c, "movimiento fases": float(movimiento),
            "phi": [float(p) for p in phi]}


def n_minimo_ingenuo(x, P, phi, h, k, exponentes=None):
    """Menor n de la grilla a partir del cual el redondeo ingenuo es estricto en todos los n mayores."""
    if exponentes is None:
        exponentes = np.arange(3, 16.01, 0.5)
    K = int(sum(k))
    resultados = []
    for e in exponentes:
        g = max(1, int(round(10**e / K)))
        d = grados_ingenuos(x, P, k, g)
        evaluacion = evaluar_enteros(k, g, d, phi, h)
        resultados.append((K * g, evaluacion["ok"]))
    n_min = None
    for n, ok in reversed(resultados):
        if not ok:
            break
        n_min = n
    return n_min, resultados


# biblioteca
def gram_schmidt(B):
    """Ortogonalización sin normalizar: (B*, μ) con B_i = B*_i + Σ_{j<i} μ_ij B*_j."""
    n = B.shape[0]
    ortogonales = np.zeros_like(B)
    mu = np.zeros((n, n))
    for i in range(n):
        vector = B[i].copy()
        for j in range(i):
            mu[i, j] = B[i] @ ortogonales[j] / (ortogonales[j] @ ortogonales[j])
            vector -= mu[i, j] * ortogonales[j]
        ortogonales[i] = vector
    return ortogonales, mu


def lll(B, delta=0.99):
    """Base LLL-reducida del reticulado generado por las filas de B."""
    B = np.array(B, dtype=float)
    n = B.shape[0]
    ortogonales, mu = gram_schmidt(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q != 0:
                B[k] -= q * B[j]
                ortogonales, mu = gram_schmidt(B)
        lovasz = (delta - mu[k, k - 1] ** 2) * (ortogonales[k - 1] @ ortogonales[k - 1])
        if ortogonales[k] @ ortogonales[k] >= lovasz:
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            ortogonales, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    return B


def babai(B, objetivo):
    """Coeficientes enteros c tales que c·B está cerca de objetivo (plano más cercano de Babai).

    Conviene pasar una base LLL-reducida: así el error queda acotado por la base.
    """
    ortogonales, _ = gram_schmidt(B)
    resto = np.array(objetivo, dtype=float)
    coeficientes = np.zeros(B.shape[0])
    for i in range(B.shape[0] - 1, -1, -1):
        c = round(resto @ ortogonales[i] / (ortogonales[i] @ ortogonales[i]))
        resto -= c * B[i]
        coeficientes[i] = c
    return coeficientes


def enteros_cercanos(formas, objetivo, radio, peso=1e3):
    """u ∈ Z^m con |u − objetivo| del orden de radio y formas·(u − objetivo) ≈ 0.

    Encaje de Kannan: el reticulado tiene filas (e_i/radio, peso·formas[:, i]). Las formas se escalan
    todas por el mismo factor (máximo 1), así que el que llama decide cuánto pesa cada una.
    """
    formas = np.asarray(formas, dtype=float)
    formas = formas / max(np.abs(formas).max(), 1e-300)
    m = formas.shape[1]
    factor = peso * radio
    B = np.hstack([np.eye(m) / radio, factor * formas.T])
    objetivo = np.asarray(objetivo, dtype=float)
    vector_objetivo = np.concatenate([objetivo / radio, factor * formas @ objetivo])
    reducida = lll(B)
    coeficientes = babai(reducida, vector_objetivo)
    return np.round((coeficientes @ reducida)[:m] * radio)


# biblioteca
def base_blanda(x, P, phi, cantidad=2):
    """Vectores singulares de J en la plantilla: (U_blando, U_rígido, V_blando, V_rígido) en float."""
    J = np.array(jacobiano_fuerzas([float(v) for v in x], [float(v) for v in phi], np.array(P, dtype=float)))
    U, singulares, Vt = np.linalg.svd(J)
    corte = J.shape[0] - cantidad
    return {"U_blando": U[:, corte:], "U_rigido": U[:, :corte],
            "V_blando": Vt[corte:].T, "V_rigido": Vt[:corte].T, "singulares": singulares[corte:]}


def _fuerzas_vector(x, phi, P):
    """F_1..F_{m−1} como mp.matrix (F_0 se deduce de Σ x_a F_a ≡ 0)."""
    return mp.matrix(fuerzas(x, phi, P, mp)[1:])


def _mover(phi_referencia, V, z):
    """φ = φ_ref + V z en las coordenadas 1..m−1 (φ_0 queda fijo)."""
    phi = list(phi_referencia)
    for a in range(1, len(phi)):
        for j in range(V.cols):
            phi[a] += V[a - 1, j] * z[j]
    return phi


def residuo_blando(x, P, phi_referencia, base, dps=60, iteraciones=30):
    """Resuelve las ecuaciones rígidas U_rᵀ F(φ_ref + V_r z) = 0 y devuelve (g, φ) con g = U_bᵀ F."""
    with mp.workdps(dps):
        U_r = mp.matrix(base["U_rigido"].tolist())
        U_b = mp.matrix(base["U_blando"].tolist())
        V_r = mp.matrix(base["V_rigido"].tolist())
        z = mp.matrix(V_r.cols, 1)
        tolerancia = mp.mpf(10) ** (-dps + 10)
        for _ in range(iteraciones):
            phi = _mover(phi_referencia, V_r, z)
            rigidas = U_r.T * _fuerzas_vector(x, phi, P)
            if mp.norm(rigidas) < tolerancia:
                break
            J = mp.matrix(jacobiano_fuerzas(x, phi, P, mp))
            z = z - mp.lu_solve(U_r.T * J * V_r, rigidas)
        phi = _mover(phi_referencia, V_r, z)
        g = U_b.T * _fuerzas_vector(x, phi, P)
        return [g[i] for i in range(g.rows)], phi


def derivada_residuo(x, P, phi_referencia, base, pares, dps=60):
    """(g, dg/dP_ab) por diferencias finitas en alta precisión: vector y matriz (blandos × pares)."""
    with mp.workdps(dps):
        g, phi = residuo_blando(x, P, phi_referencia, base, dps)
        delta = mp.mpf(10) ** (-(dps // 2 - 5))
        columnas = []
        for a, b in pares:
            P_movida = [fila[:] for fila in P]
            P_movida[a][b] += delta
            P_movida[b][a] += delta
            g_movida, _ = residuo_blando(x, P_movida, phi_referencia, base, dps)
            columna = []
            for v_movido, v in zip(g_movida, g):
                columna.append(float((v_movido - v) / delta))
            columnas.append(columna)
        return np.array([float(v) for v in g]), np.array(columnas).T, phi


def densidades_enteras(d, k, g, dps=60):
    """P_ab = d_ab/n_b en mpmath."""
    n_a = tamanos_enteros(k, g)
    P = []
    with mp.workdps(dps):
        for a in range(len(k)):
            fila = []
            for b in range(len(k)):
                fila.append(mp.mpf(d[a][b]) / n_a[b])
            P.append(fila)
    return P


def corregir_grados(d, k, g, pares, residuo, derivada, radio, tolerancias):
    """Un paso de Newton entero: grados cuyo cambio anula (en lo posible) el residuo blando.

    Cada componente del residuo se mide en unidades de su tolerancia (la distancia al pliegue, ~σ_k²).
    """
    n_a = tamanos_enteros(k, g)
    pasos = []
    actuales = []
    escalas = []
    for a, b in pares:
        paso = paso_grado(k, a, b)
        pasos.append(paso)
        actuales.append(d[a][b] // paso)
        escalas.append(paso / n_a[b])
    pesos = 1 / np.asarray(tolerancias)
    formas = pesos[:, None] * derivada * np.array(escalas)[None, :]
    objetivo = np.linalg.lstsq(formas, -pesos * residuo, rcond=None)[0]
    u = enteros_cercanos(formas, objetivo, radio)
    nuevos = [fila[:] for fila in d]
    for i, (a, b) in enumerate(pares):
        j = max(0, min(actuales[i] + int(u[i]), n_a[b] // pasos[i]))
        nuevos[a][b] = j * pasos[i]
        nuevos[b][a] = nuevos[a][b] * n_a[a] // n_a[b]
    return nuevos


def _mejor(candidato, actual):
    """¿La evaluación candidato es mejor que actual? (primero estricta, después mayor autovalor)."""
    if actual is None:
        return True
    return (candidato["ok"], candidato["menor autovalor"]) > (actual["ok"], actual["menor autovalor"])


def redondeo_reticulado(x, P, phi, h, k, g, radios=(3, 10, 30, 100), iteraciones=4, cantidad=3):
    """Grados enteros que mantienen la estrictez: Newton entero sobre el residuo blando.

    Arranca del redondeo ingenuo; en cada paso calcula el residuo blando de los datos enteros y su
    derivada, y elige con enteros_cercanos la corrección que lo anula. Evalúa en exacto partiendo de
    las fases de la reducción (donde Newton sí converge) y se queda con la mejor evaluación.
    Devuelve (d, evaluación).
    """
    pares = pares_libres(P)
    base = base_blanda(x, P, phi, cantidad)
    tolerancias = base["singulares"] ** 2
    x_entero = [_a_mp(v) for v in tamanos_exactos(k)]
    inicial = grados_ingenuos(x, P, k, g)
    mejor_d = inicial
    mejor_evaluacion = evaluar_enteros(k, g, inicial, phi, h)
    for radio in radios:
        d = inicial
        for _ in range(iteraciones):
            residuo, derivada, _ = derivada_residuo(x_entero, densidades_enteras(d, k, g), phi, base, pares)
            d = corregir_grados(d, k, g, pares, residuo, derivada, radio, tolerancias)
            _, phi_reducida = residuo_blando(x_entero, densidades_enteras(d, k, g), phi, base)
            evaluacion = evaluar_enteros(k, g, d, phi_reducida, h, phi_referencia=phi)
            if _mejor(evaluacion, mejor_evaluacion):
                mejor_d = d
                mejor_evaluacion = evaluacion
    return mejor_d, mejor_evaluacion


# biblioteca
def plantilla_de_la_curva(curva, h, margen, tamanos):
    """Fila de data/s7_curva.json para (h, ε, tamaños)."""
    for fila in curva:
        if fila["h"] == h and fila["margen"] == margen and tuple(fila["k"]) == tamanos:
            return fila
    raise KeyError(f"no está en s7_curva.json: h = {h}, ε = {margen}, k = {tamanos}")


def un_n(trabajo):
    """Redondeo ingenuo y por reticulado para un candidato y un n."""
    fila, n = trabajo
    inicio = time.time()
    h = fila["h"]
    k_mitad, k_w = fila["k"]
    k = [k_mitad] * 6 + [k_w]
    g = int(round(n / sum(k)))
    phi = fases_s7(h, np.array(fila["s"]))
    x, phi, P = pulir(tamanos_exactos(k), phi, np.array(fila["P"]))
    ingenuo = evaluar_enteros(k, g, grados_ingenuos(x, P, k, g), phi, h)
    d, reticulado = redondeo_reticulado(x, P, phi, h, k, g)
    mu = mu_entero(k, g, d)
    return {
        "h": h, "margen": fila["margen"], "k": fila["k"], "mu_plantilla": fila["mu"],
        "n": g * sum(k), "g": g,
        "ingenuo_ok": ingenuo["ok"],
        "reticulado_ok": reticulado["ok"],
        "menor_autovalor": reticulado["menor autovalor"],
        "mu": float(mu), "mu_exacto": f"{mu.numerator}/{mu.denominator}",
        "grados": d, "fases": reticulado["phi"],
        "segundos": time.time() - inicio,
    }


# biblioteca
def tabla_transporte(d_ab, k_a, k_b, g):
    """Tabla t (k_a × k_b) con filas d_ab, columnas d_ba = k_a d_ab/k_b y entradas en {base, base + 1} ⊂ [0, g].

    Los "+1" de la fila i van en las columnas i·e, …, i·e + e − 1 (mod k_b), con e = d_ab − k_b·base:
    recorridas en orden, las filas cubren las columnas de forma cíclica y cada una recibe los mismos.
    """
    total = k_a * d_ab
    assert total % k_b == 0, "tiene que valer k_a d_ab = k_b d_ba"
    assert 0 <= d_ab <= k_b * g
    base = total // (k_a * k_b)
    extra_fila = d_ab - k_b * base
    t = [[base] * k_b for _ in range(k_a)]
    for i in range(k_a):
        for desplazamiento in range(extra_fila):
            t[i][(i * extra_fila + desplazamiento) % k_b] += 1
    return t


def tablas_bloques(k, g, d):
    """{(a, b): tabla t} para a < b."""
    tablas = {}
    for a in range(len(k)):
        for b in range(a + 1, len(k)):
            tablas[(a, b)] = tabla_transporte(d[a][b], k[a], k[b], g)
    return tablas


def desplazamientos_cluster(k, g):
    """Índice del primer vértice de cada cluster (vértice (a, i, v) ↦ inicio_a + i·g + v)."""
    inicios = [0]
    for ka in k[:-1]:
        inicios.append(inicios[-1] + ka * g)
    return inicios


def adyacencia_densa(k, g, d):
    """Matriz de adyacencia del blow-up (solo para n chico: tests y validación)."""
    n_a = tamanos_enteros(k, g)
    inicios = desplazamientos_cluster(k, g)
    W = np.zeros((sum(n_a), sum(n_a)))
    for a in range(len(k)):
        bloque = slice(inicios[a], inicios[a] + n_a[a])
        W[bloque, bloque] = 1 - np.eye(n_a[a])
    v = np.arange(g)
    for (a, b), t in tablas_bloques(k, g, d).items():
        for i in range(k[a]):
            for j in range(k[b]):
                origen = inicios[a] + i * g + v
                for s in range(t[i][j]):
                    destino = inicios[b] + j * g + (v + s) % g
                    W[origen, destino] = 1
                    W[destino, origen] = 1
    return W


def hessiano_denso(W, theta):
    """H = diag(Σ_v W_uv cos(θ_u − θ_v)) − [W_uv cos(θ_u − θ_v)]."""
    A = W * np.cos(theta[:, None] - theta[None, :])
    return np.diag(A.sum(axis=1)) - A


def descripcion_compacta(k, g, d, fases):
    """Todo lo necesario para reconstruir el grafo y su estado (lo que se publica)."""
    tablas = {}
    for (a, b), t in tablas_bloques(k, g, d).items():
        tablas[f"{a},{b}"] = t
    return {
        "k": list(k), "g": g, "n": sum(tamanos_enteros(k, g)), "grados": d,
        "tablas": tablas, "conjuntos": "intervalos {0, …, t−1}", "fases": list(fases),
    }


# biblioteca
LETRAS = ["A", "B", "C", "W"]


TONOS = [0.58, 0.07, 0.40, 0.80]     # azul, naranja, verde, violeta (uno por sector)


GRIS = "#8c959f"


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


# biblioteca
def transformada_intervalo(t, g, m):
    """N̂(m) = Σ_{s=0}^{t−1} e^{2πi m s/g}."""
    s = np.arange(t)
    return np.exp(2j * np.pi * m * s / g).sum()


def hessiano_fourier(k, g, d, fases, m):
    """Bloque H_m (K × K, hermítico) del Hessiano del blow-up."""
    K = sum(k)
    fases = np.asarray(fases, dtype=float)
    cos = np.cos(fases[:, None] - fases[None, :])
    inicio_bloque = np.cumsum([0] + list(k[:-1]))
    H = np.zeros((K, K), dtype=complex)
    for a in range(len(k)):
        diagonal = sum(d[a][b] * cos[a, b] for b in range(len(k)))
        for i in range(k[a]):
            fila = inicio_bloque[a] + i
            H[fila, fila] = diagonal - ((g if m == 0 else 0) - 1)
            for j in range(k[a]):
                if j != i:
                    H[fila, inicio_bloque[a] + j] = -(g if m == 0 else 0)
    for (a, b), t in tablas_bloques(k, g, d).items():
        for i in range(k[a]):
            for j in range(k[b]):
                valor = -cos[a, b] * transformada_intervalo(t[i][j], g, m)
                H[inicio_bloque[a] + i, inicio_bloque[b] + j] = valor
                H[inicio_bloque[b] + j, inicio_bloque[a] + i] = np.conj(valor)
    return H


# biblioteca
UMBRAL = Fraction(11, 16)


def acoplamientos(k, g, d):
    """Cotas de Schur s_ab de los acoplamientos entre clusters, en los dos tipos de modos.

    - m ≠ 0: |N̂_ij(m)| ≤ min(t_ij, g − t_ij), y sumando una fila ≤ min(d_ab, n_b − d_ab).
    - m = 0 con suma cero dentro de cada cluster: solo actúa E = t − base (la parte constante
      se anula), con filas d_ab − k_b·base y columnas d_ba − k_a·base.
    Devuelve (s_fourier, s_cero), matrices m × m simétricas con s_aa = 0 (enteros exactos bajo raíz).
    """
    m = len(k)
    n_a = tamanos_enteros(k, g)
    s_fourier = [[0] * m for _ in range(m)]
    s_cero = [[0] * m for _ in range(m)]
    for a in range(m):
        for b in range(a + 1, m):
            fila = min(d[a][b], n_a[b] - d[a][b])
            columna = min(d[b][a], n_a[a] - d[b][a])
            s_fourier[a][b] = s_fourier[b][a] = fila * columna
            base = (k[a] * d[a][b]) // (k[a] * k[b])
            fila = d[a][b] - k[b] * base
            columna = d[b][a] - k[a] * base
            s_cero[a][b] = s_cero[b][a] = fila * columna
    return s_fourier, s_cero


def margenes_gershgorin(k, g, d, fases, lib=np):
    """Margen por cluster: Σ_b d_ab cos_ab + 1 − Σ_{b≠a} |cos_ab| √s_ab, para cada tipo de modo.

    Si todos son > 0, la forma cuadrática del Hessiano es positiva en todos los modos que no son
    constantes por cluster (H_m con m ≠ 0, y la parte de H_0 ortogonal a los modos de cluster).
    Con lib = mpmath.iv y fases como intervalos (la caja de Krawczyk) la cota es rigurosa.
    """
    m = len(k)
    s_fourier, s_cero = acoplamientos(k, g, d)
    margenes = {"fourier": [], "cero": []}
    for a in range(m):
        cos = [lib.cos(fases[a] - fases[b]) for b in range(m)]
        diagonal = 1
        for b in range(m):
            diagonal = diagonal + d[a][b] * cos[b]
        for nombre, s in (("fourier", s_fourier), ("cero", s_cero)):
            total = diagonal
            for b in range(m):
                if b != a:
                    total = total - abs(cos[b]) * lib.sqrt(s[a][b])
            margenes[nombre].append(total)
    return margenes


def certificar_blowup(k, g, d, fases, dps=100, radio="1e-40"):
    """Certificado riguroso del blow-up de (k, g, d), con fases iniciales `fases`. Devuelve un dict.

    ok = True si: Krawczyk + c > 0 + Cholesky de L + estado no sync en el cociente (`certificar`),
    márgenes de Gershgorin > 0 sobre la caja de Krawczyk (resto de los modos) y μ > 11/16 exacto.
    """
    x, P = plantilla_entera(k, g, d)
    try:
        phi = fases_newton(x, P, fases, dps=dps + 20)
    except ZeroDivisionError:
        return {"ok": False, "motivo": "Newton no converge"}
    cociente = certificar(x, P, phi, radio=radio, dps=dps, verbose=False)
    margenes = margenes_gershgorin(k, g, d, cociente["caja"], lib=iv)
    menor_fourier = min(mp.mpf(v.a) for v in margenes["fourier"])
    menor_cero = min(mp.mpf(v.a) for v in margenes["cero"])
    mu = mu_entero(k, g, d)
    n = sum(k) * g
    ok = cociente["ok"] and menor_fourier > 0 and menor_cero > 0 and mu > UMBRAL
    return {
        "ok": bool(ok),
        "krawczyk": bool(cociente["krawczyk"]),
        "c>0": bool(cociente["c>0"]),
        "cholesky": bool(cociente["cholesky"]),
        "distintas": bool(cociente["distintas"]),
        "no sync": bool(cociente["no sync"]),
        "min pivote L (cota inf)": float(cociente["min pivote (cota inf)"]),
        "margen fourier / n (cota inf)": float(menor_fourier / n),
        "margen cero / n (cota inf)": float(menor_cero / n),
        "mu": float(mu), "mu_exacto": f"{mu.numerator}/{mu.denominator}",
        "mu > 11/16": bool(mu > UMBRAL),
        "radio caja": radio,
        "fases": [mp.nstr(p, 30) for p in phi],
    }


# biblioteca
def control_ingenuo(r, curva):
    """Grados del redondeo ingenuo para el mismo (h, ε, k, n) que la fila r de s7_enteros.json."""
    fila = [f for f in curva if f["h"] == r["h"] and f["margen"] == r["margen"] and f["k"] == r["k"]][0]
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


# biblioteca
class Arreglos:
    """Las pocas operaciones que hacen falta, con numpy (dispositivo None) o con torch."""

    def __init__(self, dispositivo=None):
        self.torch = None
        if dispositivo is not None:
            import torch
            self.torch = torch
            self.dispositivo = torch.device(dispositivo)

    def desde(self, x):
        """numpy → arreglo del backend (float64)."""
        x = np.asarray(x, dtype=np.float64)
        if self.torch is None:
            return x
        return self.torch.from_numpy(x).to(self.dispositivo)

    def ceros(self, n):
        """Vector de ceros de largo n."""
        return self.desde(np.zeros(n))

    def aleatorio(self, n, semilla):
        """Vector gaussiano de largo n (generado en CPU para que sea reproducible)."""
        return self.desde(np.random.default_rng(semilla).standard_normal(n))

    def a_numpy(self, x):
        """Arreglo del backend → numpy."""
        if self.torch is None:
            return np.asarray(x)
        return x.cpu().numpy()

    def escalar(self, x):
        """Escalar del backend → float."""
        return float(x)

    def suma_prefija_doble(self, Y):
        """S de forma (k, 2g + 1): S[:, l] = suma de las primeras l entradas de [Y, Y] (fila por fila)."""
        if self.torch is None:
            Z = np.concatenate([Y, Y], axis=1)
            return np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(Z, axis=1)], axis=1)
        Z = self.torch.cat([Y, Y], dim=1)
        ceros = self.torch.zeros((Y.shape[0], 1), dtype=Y.dtype, device=Y.device)
        return self.torch.cat([ceros, self.torch.cumsum(Z, dim=1)], dim=1)

    def media_filas(self, Y):
        """Media de cada fila, como columna (k, 1)."""
        if self.torch is None:
            return Y.mean(axis=1, keepdims=True)
        return Y.mean(dim=1, keepdim=True)

    def punto(self, x, y):
        """Producto escalar ⟨x, y⟩."""
        return self.escalar((x * y).sum())

    def norma(self, x):
        """Norma euclídea."""
        return self.punto(x, x) ** 0.5

    def coseno_seno(self, x):
        """(cos x, sin x) elemento a elemento."""
        if self.torch is None:
            return np.cos(x), np.sin(x)
        return self.torch.cos(x), self.torch.sin(x)

    def tomar(self, x, indices):
        """x[indices] como numpy (indices: arreglo numpy de enteros)."""
        if self.torch is None:
            return np.asarray(x[indices])
        return x[self.torch.from_numpy(indices).to(self.dispositivo)].cpu().numpy()


class GrafoImplicito:
    """Blow-up de la descripción compacta, con W·y y H·x implícitos.

    Vértice (a, i, v) ↦ índice inicio_a + i·g + v (el mismo orden que `blowup.adyacencia_densa`).
    """

    def __init__(self, descripcion, arreglos):
        self.arreglos = arreglos
        self.k = list(descripcion["k"])
        self.g = int(descripcion["g"])
        self.n = sum(self.k) * self.g
        self.inicios = [0]
        for ka in self.k[:-1]:
            self.inicios.append(self.inicios[-1] + ka * self.g)
        self.mascaras = {}
        for clave, tabla in descripcion["tablas"].items():
            a, b = (int(c) for c in clave.split(","))
            self.mascaras[(a, b)] = self._mascaras_tabla(np.array(tabla))
        fases_cluster = [float(p) for p in descripcion["fases"]]
        theta = np.repeat(fases_cluster, [ka * self.g for ka in self.k])
        self.coseno = arreglos.desde(np.cos(theta))
        self.seno = arreglos.desde(np.sin(theta))
        self.diagonal = (self.coseno * self.producto_adyacencia(self.coseno)
                         + self.seno * self.producto_adyacencia(self.seno))

    def _mascaras_tabla(self, tabla):
        """[(t, M_t)]: para cada valor t de la tabla, la matriz 0/1 de los pares (i, j) con t_ij = t."""
        assert tabla.min() >= 0 and tabla.max() <= self.g
        mascaras = []
        for t in np.unique(tabla):
            if t > 0:
                mascaras.append((int(t), self.arreglos.desde((tabla == t).astype(float))))
        return mascaras

    def por_cluster(self, y):
        """Vistas (k_a, g) de y, una por cluster (escribir en ellas escribe en y)."""
        partes = []
        for a, ka in enumerate(self.k):
            partes.append(y[self.inicios[a]: self.inicios[a] + ka * self.g].reshape(ka, self.g))
        return partes

    def producto_adyacencia(self, y):
        """W·y. Las ventanas cíclicas salen de sumas prefijas de y centrado por bloque (más preciso)."""
        g = self.g
        salida = self.arreglos.ceros(self.n)
        partes = self.por_cluster(y)
        partes_salida = self.por_cluster(salida)
        medias = []
        prefijas = []
        for a, Y in enumerate(partes):
            partes_salida[a] += Y.sum() - Y
            media = self.arreglos.media_filas(Y)
            medias.append(media)
            prefijas.append(self.arreglos.suma_prefija_doble(Y - media))
        for (a, b), mascaras in self.mascaras.items():
            for t, M in mascaras:
                # (a, i, v) recibe Σ_{s<t} y_(b, j, v+s);  (b, j, w) recibe Σ_{s<t} y_(a, i, w−s)
                adelante = prefijas[b][:, t: t + g] - prefijas[b][:, :g] + t * medias[b]
                atras = prefijas[a][:, g + 1: 2 * g + 1] - prefijas[a][:, g + 1 - t: 2 * g + 1 - t] + t * medias[a]
                partes_salida[a] += M @ adelante
                partes_salida[b] += M.T @ atras
        return salida

    def producto_hessiano(self, x):
        """H·x con H = diag(Σ_v W_uv cos(θ_u − θ_v)) − [W_uv cos(θ_u − θ_v)]."""
        cruzado = (self.coseno * self.producto_adyacencia(self.coseno * x)
                   + self.seno * self.producto_adyacencia(self.seno * x))
        return self.diagonal * x - cruzado

    def velocidad(self):
        """θ̇_u = Σ_v W_uv sin(θ_v − θ_u) en las fases del grafo."""
        return self._velocidad(self.coseno, self.seno)

    def velocidad_en(self, theta):
        """θ̇_u = Σ_v W_uv sin(θ_v − θ_u) en fases θ cualesquiera (para simular la dinámica)."""
        coseno, seno = self.arreglos.coseno_seno(theta)
        return self._velocidad(coseno, seno)

    def _velocidad(self, coseno, seno):
        """cos θ ∘ W sin θ − sin θ ∘ W cos θ, que es Σ_v W_uv sin(θ_v − θ_u)."""
        return coseno * self.producto_adyacencia(seno) - seno * self.producto_adyacencia(coseno)

    def indicadora(self, b):
        """Vector 1 en los vértices del cluster b."""
        y = self.arreglos.ceros(self.n)
        y[self.inicios[b]: self.inicios[b] + self.k[b] * self.g] = 1
        return y

    def quitar_medias_cluster(self, x):
        """Proyección al complemento de los modos de cluster (resta la media de cada cluster)."""
        x = x.clone() if self.arreglos.torch is not None else x.copy()
        for X in self.por_cluster(x):
            X -= X.mean()
        return x

    def quitar_media(self, x):
        """Proyección a 1^⊥ (quita la rotación global)."""
        return x - x.mean()


def grados_medidos(grafo):
    """Matriz (m × m) con [min, max] de los vecinos que tiene un vértice de a en b, contados con W·1_b."""
    m = len(grafo.k)
    minimos = np.zeros((m, m), dtype=np.int64)
    maximos = np.zeros((m, m), dtype=np.int64)
    for b in range(m):
        cuenta = grafo.arreglos.a_numpy(grafo.producto_adyacencia(grafo.indicadora(b)))
        for a in range(m):
            tramo = cuenta[grafo.inicios[a]: grafo.inicios[a] + grafo.k[a] * grafo.g]
            minimos[a, b] = int(np.rint(tramo.min()))
            maximos[a, b] = int(np.rint(tramo.max()))
    return minimos, maximos


def cociente_implicito(grafo):
    """Q_ab = 1_aᵀ H 1_b / √(n_a n_b), calculada con el producto implícito."""
    m = len(grafo.k)
    Q = np.zeros((m, m))
    tamanos = [ka * grafo.g for ka in grafo.k]
    for b in range(m):
        columna = grafo.arreglos.a_numpy(grafo.producto_hessiano(grafo.indicadora(b)))
        for a in range(m):
            tramo = columna[grafo.inicios[a]: grafo.inicios[a] + tamanos[a]]
            Q[a, b] = tramo.sum() / np.sqrt(tamanos[a] * tamanos[b])
    return Q


def cociente_formula(descripcion):
    """La misma Q con la fórmula de los grados: Q_ab = −√(n_a/n_b) d_ab cos_ab, Q_aa = Σ_{c≠a} d_ac cos_ac."""
    k = descripcion["k"]
    g = descripcion["g"]
    d = np.array(descripcion["grados"], dtype=float)
    fases = np.array([float(p) for p in descripcion["fases"]])
    coseno = np.cos(fases[:, None] - fases[None, :])
    tamanos = np.array(k, dtype=float) * g
    Q = -np.sqrt(tamanos[:, None] / tamanos[None, :]) * d * coseno
    for a in range(len(k)):
        Q[a, a] = sum(d[a, c] * coseno[a, c] for c in range(len(k)) if c != a)
    return Q


def _extremos_ritz(alfas, betas):
    """(menor valor de Ritz, último componente de su vector, mayor valor de Ritz) de la tridiagonal."""
    if len(alfas) == 1:
        return alfas[0], 1.0, alfas[0]
    menor, vector = eigh_tridiagonal(alfas, betas[:-1], select="i", select_range=(0, 0))
    mayor = eigh_tridiagonal(alfas, betas[:-1], eigvals_only=True, select="i",
                             select_range=(len(alfas) - 1, len(alfas) - 1))
    return float(menor[0]), float(vector[-1, 0]), float(mayor[0])


def lanczos(aplicar, proyectar, inicial, arreglos, iteraciones, cada=25, tolerancia=1e-6, informar=None):
    """Lanczos sin reortogonalizar para el menor autovalor de un operador simétrico.

    El residuo ‖H y − θ y‖ del vector de Ritz es β_k·|s_k| (último componente), sin calcular y.
    Corta cuando ese residuo es ≤ tolerancia·|θ|, o si β se anula (el Krylov ya es invariante).
    `informar(fila)` recibe el progreso cada `cada` pasos.
    """
    q = proyectar(inicial)
    q = q / arreglos.norma(q)
    q_anterior = None
    beta = 0.0
    alfas = []
    betas = []
    historial = []
    escala = 0.0
    for paso in range(1, iteraciones + 1):
        w = proyectar(aplicar(q))
        alfa = arreglos.punto(q, w)
        w = w - alfa * q
        if q_anterior is not None:
            w = w - beta * q_anterior
        # reproyectar: si β es chico, w/β amplifica el redondeo en las direcciones proyectadas
        # (donde el operador proyectado vale 0) y aparece un valor de Ritz espurio ≈ 0
        w = proyectar(w)
        beta = arreglos.norma(w)
        alfas.append(alfa)
        betas.append(beta)
        escala = max(escala, abs(alfa))
        final = paso == iteraciones or beta <= 1e-12 * escala
        if paso % cada == 0 or final:
            menor, ultimo, mayor = _extremos_ritz(np.array(alfas), np.array(betas))
            fila = {"paso": paso, "menor": menor, "residuo": beta * abs(ultimo), "mayor": mayor}
            historial.append(fila)
            if informar is not None:
                informar(fila)
            if fila["residuo"] <= tolerancia * abs(menor) or final:
                break
        q_anterior = q
        q = w / beta
    return {"menor": historial[-1]["menor"], "residuo": historial[-1]["residuo"],
            "mayor": historial[-1]["mayor"], "pasos": historial[-1]["paso"],
            "convergio": historial[-1]["residuo"] <= tolerancia * abs(historial[-1]["menor"]),
            "historial": historial}


def elegir_grafo(filas, n=None):
    """Fila certificada con ese n (o la de n más chico si n es None).

    `filas` es el contenido de data/grafos_s7.json (lista; se toman solo los redondeos por reticulado)
    o de data/grafo_simetrico_s7.json (un solo grafo).
    """
    if isinstance(filas, dict):
        filas = [filas]
    filas = [f for f in filas if f.get("redondeo", "reticulado") == "reticulado" and f["certificado"]["ok"]]
    if n is not None:
        filas = [f for f in filas if f["grafo"]["n"] == n]
    if not filas:
        raise ValueError(f"no hay un grafo certificado con n = {n}")
    return min(filas, key=lambda f: (f["grafo"]["n"], -f["h"]))


# biblioteca
def informar_lanczos(etiqueta, inicio):
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
                        informar=informar_lanczos(etiqueta, inicio))
    resultado["segundos"] = time.time() - inicio
    return resultado


# biblioteca
def inicios_bloques(k):
    """Índice del primer bloque de cada cluster dentro de los K bloques."""
    return np.cumsum([0] + list(k[:-1]))


def transformada_intervalos(tabla, g, frecuencias):
    """N̂_ij(m) = Σ_{s<t_ij} e^{2πi m s/g} con forma (M, k_a, k_b), por la fórmula de la suma geométrica.

    N̂ = e^{iπm(t−1)/g} sin(πmt/g) / sin(πm/g), y t si m = 0. Los productos m·t se reducen mod 2g en
    enteros, así los ángulos quedan en [0, 2π) sin perder precisión aunque m·t sea grande.
    """
    t = np.asarray(tabla, dtype=np.int64)[None, :, :]
    m = np.asarray(frecuencias, dtype=np.int64)[:, None, None]
    fase = np.pi * ((m * (t - 1)) % (2 * g)) / g
    numerador = np.sin(np.pi * ((m * t) % (2 * g)) / g)
    denominador = np.sin(np.pi * m / g)
    cociente = np.divide(numerador, denominador, out=np.zeros(numerador.shape), where=denominador != 0)
    return np.where(m == 0, t.astype(float), np.exp(1j * fase) * cociente)


def bloques_fourier(descripcion, frecuencias):
    """Bloques H_m (M × K × K, hermíticos) del Hessiano, para las frecuencias dadas."""
    k = list(descripcion["k"])
    g = int(descripcion["g"])
    d = np.array(descripcion["grados"], dtype=float)
    fases = np.array([float(p) for p in descripcion["fases"]])
    coseno = np.cos(fases[:, None] - fases[None, :])
    frecuencias = np.asarray(frecuencias, dtype=np.int64)
    inicios = inicios_bloques(k)
    K = sum(k)
    H = np.zeros((len(frecuencias), K, K), dtype=complex)
    clique = np.where(frecuencias == 0, g, 0).astype(float)
    for a, ka in enumerate(k):
        indices = np.arange(inicios[a], inicios[a] + ka)
        H[:, indices[:, None], indices[None, :]] = -clique[:, None, None] * np.ones((ka, ka))
        diagonal = float((d[a] * coseno[a]).sum())
        H[:, indices, indices] = diagonal - (clique[:, None] - 1)
    for clave, tabla in descripcion["tablas"].items():
        a, b = (int(c) for c in clave.split(","))
        N = transformada_intervalos(tabla, g, frecuencias)
        filas = slice(inicios[a], inicios[a] + k[a])
        columnas = slice(inicios[b], inicios[b] + k[b])
        H[:, filas, columnas] = -coseno[a, b] * N
        H[:, columnas, filas] = -coseno[a, b] * np.conj(N).transpose(0, 2, 1)
    return H


def producto_fourier(descripcion, x, lote=4096):
    """H·x por bloques: FFT en cada bloque Z_g, H_m en cada frecuencia y FFT inversa."""
    K = sum(descripcion["k"])
    g = int(descripcion["g"])
    X = np.fft.fft(np.asarray(x, dtype=float).reshape(K, g), axis=1)
    Y = np.empty_like(X)
    for inicio in range(0, g, lote):
        m = np.arange(inicio, min(inicio + lote, g))
        Y[:, m] = np.einsum("mij,jm->im", bloques_fourier(descripcion, m), X[:, m])
    return np.fft.ifft(Y, axis=1).real.reshape(-1)


def base_cluster(k):
    """Matriz K × len(k): columna a = indicadora de los bloques del cluster a, normalizada."""
    C = np.zeros((sum(k), len(k)))
    inicios = inicios_bloques(k)
    for a, ka in enumerate(k):
        C[inicios[a]: inicios[a] + ka, a] = 1 / np.sqrt(ka)
    return C


def espectro_fourier(descripcion, lote=4096, informar=None):
    """Todos los autovalores de H, agrupados.

    - "cluster": los de los modos constantes por cluster (H_0 restringida a la base de clusters);
    - "cero": los del resto de H_0 (diferencias entre bloques de un mismo cluster);
    - "minimos": el menor autovalor de H_m para cada m de "frecuencias" (1, …, ⌊g/2⌋);
    - "acople": max |Cᵀ H_0 C⊥|, que es 0 si los modos de cluster son invariantes (grados exactos);
    - "maximo": el mayor autovalor de H.
    `informar(hechas, total)` recibe el progreso.
    """
    k = list(descripcion["k"])
    g = int(descripcion["g"])
    H0 = bloques_fourier(descripcion, [0])[0].real
    C = base_cluster(k)
    completa, _ = np.linalg.qr(C, mode="complete")
    resto = completa[:, len(k):]
    cluster = np.linalg.eigvalsh(C.T @ H0 @ C)
    cero = np.linalg.eigvalsh(resto.T @ H0 @ resto)
    frecuencias = np.arange(1, g // 2 + 1)
    minimos = np.empty(len(frecuencias))
    maximo = max(cluster.max(), cero.max())
    for inicio in range(0, len(frecuencias), lote):
        tramo = slice(inicio, min(inicio + lote, len(frecuencias)))
        valores = np.linalg.eigvalsh(bloques_fourier(descripcion, frecuencias[tramo]))
        minimos[tramo] = valores[:, 0]
        maximo = max(maximo, float(valores[:, -1].max()))
        if informar is not None:
            informar(tramo.stop, len(frecuencias))
    return {
        "cluster": cluster, "cero": cero, "frecuencias": frecuencias, "minimos": minimos,
        "acople": float(np.abs(C.T @ H0 @ resto).max()), "maximo": float(maximo),
    }


def curva_minimos(frecuencias, minimos, g, puntos=2000):
    """(m/g, menor autovalor) reducida a ~`puntos` tramos, tomando el mínimo de cada tramo."""
    tramos = np.array_split(np.arange(len(minimos)), min(puntos, len(minimos)))
    abscisas = []
    valores = []
    for tramo in tramos:
        mejor = tramo[np.argmin(minimos[tramo])]
        abscisas.append(float(frecuencias[mejor] / g))
        valores.append(float(minimos[mejor]))
    return abscisas, valores


def chequear_producto(descripcion, grafo, semillas, lote):
    """Máximo error relativo ‖H x (Fourier) − H x (implícito)‖ / ‖H x‖ sobre vectores gaussianos."""
    errores = []
    for semilla in semillas:
        x = np.random.default_rng(semilla).standard_normal(grafo.n)
        implicito = grafo.arreglos.a_numpy(grafo.producto_hessiano(grafo.arreglos.desde(x)))
        fourier = producto_fourier(descripcion, x, lote=lote)
        errores.append(float(np.linalg.norm(fourier - implicito) / np.linalg.norm(implicito)))
    return max(errores)


def informar_espectro(inicio, lote):
    """Función que imprime el progreso de la diagonalización (cada 50 lotes)."""
    def imprimir(hechas, total):
        if hechas == total or (hechas // lote) % 50 == 0:
            print(f"  {hechas}/{total} frecuencias ({time.time() - inicio:.0f} s)", flush=True)
    return imprimir


# biblioteca
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


def envolver(angulo):
    """Ángulo llevado a [−π, π)."""
    return (angulo + np.pi) % (2 * np.pi) - np.pi


def integrar_clusters(descripcion, y0, tiempo, puntos=120):
    """Integra la dinámica por clusters desde y0 (LSODA): tiempos, distancia a las fases certificadas y r."""
    fases = np.array([float(p) for p in descripcion["fases"]])
    x = np.array(descripcion["k"], dtype=float) / sum(descripcion["k"])
    f, jac = campo_clusters(descripcion)
    tiempos = np.concatenate([[0], np.logspace(-1, np.log10(tiempo), puntos)])
    sol = solve_ivp(f, (0, tiempo), y0, method="LSODA", jac=jac, rtol=1e-10, atol=1e-13, t_eval=tiempos)
    distancias = []
    radios = []
    for y in sol.y.T:
        diferencia = envolver(y - fases)
        distancias.append(float(np.abs(envolver(diferencia - np.sum(x * diferencia))).max()))
        radios.append(float(abs(np.sum(x * np.exp(1j * y)))))
    return sol.t, distancias, radios


def simular_clusters(descripcion, amplitudes, tiempo, semilla):
    """Para cada amplitud δ: perturba las fases de los clusters (al azar y en el modo más lento) e integra."""
    fases = np.array([float(p) for p in descripcion["fases"]])
    x = np.array(descripcion["k"], dtype=float) / sum(descripcion["k"])
    _, jac = campo_clusters(descripcion)
    autovalores, vectores = np.linalg.eig(jac(0, fases))
    orden = np.argsort(-autovalores.real)
    direcciones = {"modo más lento": vectores[:, orden[1]].real,
                   "al azar": np.random.default_rng(semilla).standard_normal(len(fases))}
    corridas = []
    for nombre, direccion in direcciones.items():
        direccion = direccion - np.sum(x * direccion)
        direccion = direccion / np.abs(direccion).max()
        for delta in amplitudes:
            inicio = time.time()
            tiempos, distancias, radios = integrar_clusters(descripcion, fases + delta * direccion, tiempo)
            r_final = radios[-1]
            print(f"  {nombre:15s} δ = {delta:.0e}: distancia final {distancias[-1]:.1e},  r final = {r_final:.4f}"
                  f"  ({time.time() - inicio:.1f} s)", flush=True)
            corridas.append({"direccion": nombre, "delta": delta, "t": tiempos.tolist(), "distancia": distancias,
                             "r_final": r_final})
    return corridas, -np.sort(autovalores.real)[::-1]


# biblioteca
def fases_vertices(descripcion):
    """θ del estado certificado, vértice por vértice."""
    fases = [float(p) for p in descripcion["fases"]]
    return np.repeat(fases, np.array(descripcion["k"]) * descripcion["g"])


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

    def correr(self, theta, tiempo, dt, fotos, t0=0.0, medir_cada=1):
        """Integra de t0 a `tiempo`; guarda r y la distancia cada `medir_cada` pasos y las fases de la muestra en `fotos`.

        Medir obliga a traer números a la CPU (en GPU, una espera por medición): con pasos chicos y tiempos
        largos conviene medir cada tanto.
        """
        pasos = int(round((tiempo - t0) / dt))
        tiempos_foto = set()
        for t in fotos:
            tiempos_foto.add(int(round((t - t0) / dt)))
        aviso = max(1, pasos // 10)
        serie = {"t": [], "r": [], "distancia": []}
        instantaneas = []
        inicio = time.time()
        for paso in range(pasos + 1):
            if paso % medir_cada == 0 or paso % aviso == 0 or paso == pasos:
                r, distancia = self.medir(theta)
                serie["t"].append(t0 + paso * dt)
                serie["r"].append(r)
                serie["distancia"].append(distancia)
            if paso in tiempos_foto:
                instantaneas.append((t0 + paso * dt, self.arreglos.tomar(theta, self.muestra)))
            if paso % aviso == 0:
                print(f"  t = {t0 + paso * dt:9.1f}  r = {r:.4f}  distancia al estado = {distancia:.2e}"
                      f"  ({time.time() - inicio:.0f} s)", flush=True)
            if paso < pasos:
                theta = self.paso_rk4(theta, dt)
        return serie, instantaneas, theta

    def medias_cluster(self, theta):
        """Fase media (circular) de cada cluster, como numpy."""
        coseno, seno = self.arreglos.coseno_seno(theta)
        medias = []
        for C, S in zip(self.grafo.por_cluster(coseno), self.grafo.por_cluster(seno)):
            medias.append(np.arctan2(float(S.mean()), float(C.mean())))
        return np.array(medias)

    def inicial_perturbado(self, ruido, semilla):
        """Estado certificado + ruido gaussiano con media 0 en cada cluster (solo modos dentro de los clusters)."""
        perturbacion = self.arreglos.aleatorio(self.grafo.n, semilla) * ruido
        return self.estrella + self.grafo.quitar_medias_cluster(perturbacion)

    def inicial_azar(self, semilla):
        """Fases uniformes en [0, 2π)."""
        return self.arreglos.desde(np.random.default_rng(semilla).uniform(0, 2 * np.pi, self.grafo.n))


def dibujar_simulacion(serie, continuacion, instantaneas, cluster_muestra, descripcion, titulo, archivo):
    """r(t) y distancia al estado certificado (grafo completo y continuación por clusters), y fotos de las fases."""
    nombres, colores = nombres_y_colores([float(p) for p in descripcion["fases"]])
    fig = plt.figure(figsize=(13, 6.6))
    grilla = fig.add_gridspec(2, len(instantaneas), height_ratios=[1, 1.15])
    eje_r = fig.add_subplot(grilla[0, : len(instantaneas) // 2])
    eje_d = fig.add_subplot(grilla[0, len(instantaneas) // 2:])
    distancias = serie["distancia"] + continuacion["distancia"]
    for eje, clave, color in ((eje_r, "r", "#2a78d6"), (eje_d, "distancia", "#eb6834")):
        eje.plot(serie["t"][1:], serie[clave][1:], color=color, lw=2, label="grafo completo (RK4)")
        eje.plot(continuacion["t"], continuacion[clave], color=color, lw=2, ls="--",
                 label=f"reducción a clusters desde t = {continuacion['desde']:g}")
        eje.axvline(continuacion["desde"], color=GRIS, lw=1, ls=":")
        eje.set(xscale="log", xlabel="tiempo (normalizado)")
        eje.grid(alpha=0.25)
        eje.legend(fontsize=7, loc="best")
    eje_r.set(ylabel="r = |⟨e^{iθ}⟩|", ylim=(0, 1.03), title="parámetro de orden (1 = sincronizado)")
    eje_d.set(ylabel="max |θ − θ*| (rad)", title="distancia al estado certificado θ* (sin la rotación)")
    if min(distancias) < 0.1 * max(distancias):
        eje_d.set_yscale("log")
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


# biblioteca
REFLEJO = [5, 4, 3, 2, 1, 0, 6]          # Am↔Cp, Ap↔Cm, Bm↔Bp, W fijo
H = 0.03
MARGEN = 1e-3


def s_simetrico(p):
    """Desplazamientos (s_Am, s_Ap, s_Bm) → los seis, con s_Cp = −s_Am, s_Cm = −s_Ap, s_Bp = −s_Bm."""
    am, ap, bm = p
    return np.array([am, ap, bm, -bm, -ap, -am])


# biblioteca
CRUZADOS = [(0, 3), (1, 2), (2, 5), (3, 4)]                                 # Am–Bp, Ap–Bm, Bm–Cp, Bp–Cm
VECINOS_EN_1 = [(0, 2), (1, 3), (2, 4), (3, 5), (0, 6), (1, 6), (4, 6), (5, 6)]


def sdp_s7(x, phi, h, margen, restricciones=()):
    """Mayor μ sobre las densidades P con x y φ fijos. Devuelve (μ, P) o (None, None) si es infactible."""
    m = len(x)
    diferencias = phi[:, None] - phi[None, :]
    senos = np.sin(diferencias)
    cosenos = np.cos(diferencias)
    pesos_L = np.outer(x, x) * cosenos
    np.fill_diagonal(pesos_L, 0)

    P = cp.Variable((m, m), symmetric=True)
    t = cp.Variable()
    K = cp.multiply(P, pesos_L)
    L = cp.diag(cp.sum(K, axis=1)) - K
    Q = base_perpendicular(m)
    lista = [
        P >= 0,
        P <= 1,
        cp.diag(P) == 1,
        P @ x >= t,
        cp.multiply(P, senos) @ x == 0,
        cp.multiply(P, cosenos) @ x >= 1e-3,
        (Q.T @ L @ Q) / h**2 >> margen * np.eye(m - 1),
        cp.multiply(P, prohibidos_s7().astype(float)) == 0,
    ]
    if "reflejo" in restricciones:
        R = np.eye(m)[REFLEJO]
        lista.append(P == R @ P @ R.T)
    if "q_unico" in restricciones:
        a0, b0 = CRUZADOS[0]
        for a, b in CRUZADOS[1:]:
            lista.append(P[a, b] == P[a0, b0])
    if "vecinos_en_1" in restricciones:
        for a, b in VECINOS_EN_1:
            lista.append(P[a, b] == 1)
    problema = cp.Problem(cp.Maximize(t), lista)
    try:
        problema.solve(solver="CLARABEL")
    except cp.error.SolverError:
        return None, None
    if problema.status not in ("optimal", "optimal_inaccurate"):
        return None, None
    return float(t.value), np.array(P.value)


def optimizar_s7_restringido(x, h, margen, restricciones=(), reinicios=1):
    """Nelder–Mead en los desplazamientos (3 con reflejo, 6 sin), reiniciado desde su resultado. Devuelve (μ, s, P)."""
    _, w, _ = esquina(30)
    w = float(w)
    simetrico = "reflejo" in restricciones
    p0 = np.array([-1.0, 1.0, -w]) if simetrico else np.array([-1.0, 1.0, -w, w, -1.0, 1.0])

    def desplazamientos(p):
        return s_simetrico(p) if simetrico else p

    def objetivo(p):
        mu, _ = sdp_s7(x, fases_s7(h, desplazamientos(p)), h, margen, restricciones)
        return 1.0 if mu is None else -mu

    opciones = {"maxiter": 600, "xatol": 1e-5, "fatol": 1e-9}
    resultado = minimize(objetivo, p0, method="Nelder-Mead", options=opciones)
    for _ in range(reinicios):
        resultado = minimize(objetivo, resultado.x, method="Nelder-Mead", options=opciones)
    s = desplazamientos(resultado.x)
    mu, P = sdp_s7(x, fases_s7(h, s), h, margen, restricciones)
    return mu, s, P


# biblioteca
def mostrar_matriz(P, colores_celda, titulo):
    """Matriz de densidades 7 × 7 con nombres de clusters; las celdas con color son las intermedias."""
    textos = []
    for fila in P:
        textos.append(["1" if v > 1 - 1e-4 else "0" if v < 1e-4 else f"{v:.4f}" for v in fila])
    fig, eje = plt.subplots(figsize=(1, 0.22 * (len(P) + 1)))
    eje.axis("off")
    tabla = eje.table(cellText=textos, rowLabels=NOMBRES, colLabels=NOMBRES, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.auto_set_column_width(list(range(-1, len(P))))
    tabla.scale(1, 1.3)
    for (i, j), celda in tabla.get_celld().items():
        celda.set_edgecolor(BORDE)
        celda.set_linewidth(0.6)
        celda.PAD = 0.15
        celda.get_text().set_color(TINTA)
        if i == 0 or j == -1:
            celda.set_facecolor(FONDO_ENCABEZADO)
            celda.get_text().set_fontweight("bold")
        elif colores_celda[i - 1][j] is not None:
            celda.set_facecolor(colores_celda[i - 1][j])
            celda.get_text().set_fontweight("bold")
    fig.canvas.draw()
    caja = tabla.get_window_extent(fig.canvas.get_renderer())
    x0, y1 = eje.transAxes.inverted().transform((caja.x0, caja.y1))
    eje.annotate(titulo, xy=(x0, y1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                 fontsize=10, color=TINTA)
    plt.show()


# biblioteca
def jacobiano_completo(x, phi, P):
    """∂F_a/∂φ_b para las 7 fases (incluye la rotación global, que tiene valor singular 0)."""
    J = np.zeros((7, 7))
    for a in range(7):
        for b in range(7):
            if a != b:
                J[a, b] = -P[a, b] * x[b] * np.cos(phi[a] - phi[b])
        J[a, a] = -J[a].sum()
    return J


def matriz_reflejo():
    """δφ ↦ R δφ con (R δφ)_{R(a)} = −δφ_a."""
    R = np.zeros((7, 7))
    for a in range(7):
        R[REFLEJO[a], a] = -1
    return R


# biblioteca
def simetrizar(P, tolerancia=1e-4):
    """Promedio exacto con el reflejo, y los extremos que el SDP dejó a menos de tolerancia, a 0 o 1."""
    R = np.eye(7)[REFLEJO]
    P = (P + R @ P @ R.T) / 2
    P = (P + P.T) / 2
    P[P < tolerancia] = 0
    P[P > 1 - tolerancia] = 1
    return P


def parejas_libres(P):
    """Pares intermedios agrupados por el reflejo: [[par, espejo], ...], con a < b en cada par."""
    parejas = []
    vistos = set()
    for a in range(7):
        for b in range(a + 1, 7):
            if 0 < P[a][b] < 1 and (a, b) not in vistos:
                espejo = tuple(sorted((REFLEJO[a], REFLEJO[b])))
                parejas.append([(a, b), espejo])
                vistos |= {(a, b), espejo}
    return parejas


def fases_reducidas(psi):
    """Estado simétrico a partir de ψ = (φ_Am, φ_Ap, φ_Bm): φ_R(a) = π − φ_a y W en 3π/2."""
    return [psi[0], psi[1], psi[2], mp.pi - psi[2], mp.pi - psi[1], mp.pi - psi[0], 3 * mp.pi / 2]


def ecuaciones_reducidas(x, psi, P):
    """(F_Am, F_Ap, F_Bm): en un estado simétrico las demás son sus opuestas y F_W = 0."""
    F = fuerzas(x, fases_reducidas(psi), P, mp)
    return mp.matrix([F[0], F[1], F[2]])


def jacobiano_reducido(x, psi, P):
    """∂(F_Am, F_Ap, F_Bm)/∂ψ por la regla de la cadena: dφ/dψ vale +1 en Am, Ap, Bm y −1 en sus espejos."""
    phi = fases_reducidas(psi)
    dphi = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [-1, 0, 0], [0, 0, 0]]
    J = mp.matrix(3, 3)
    for i, a in enumerate([0, 1, 2]):
        fila = [mp.mpf(0)] * 7                       # ∂F_a/∂φ_b
        for b in range(7):
            if b != a:
                termino = P[a][b] * x[b] * mp.cos(phi[a] - phi[b])
                fila[a] += termino
                fila[b] -= termino
        for j in range(3):
            J[i, j] = sum(fila[b] * dphi[b][j] for b in range(7))
    return J


def pulir_simetrico(x, psi, P, parejas):
    """Cambio de mínima norma en P, una variable por pareja espejo, con F(φ(ψ)) = 0 exacto."""
    P = [[mp.mpf(float(v)) for v in fila] for fila in P]
    F = ecuaciones_reducidas(x, psi, P)
    phi = fases_reducidas(psi)
    A = mp.matrix(3, len(parejas))
    for c, pareja in enumerate(parejas):
        E = [[mp.mpf(0)] * 7 for _ in range(7)]       # F es lineal en P: la columna c es F(E_c)
        for a, b in pareja:
            E[a][b] = E[b][a] = mp.mpf(1)
        columna = fuerzas(x, phi, E, mp)
        for i in range(3):
            A[i, c] = columna[i]
    delta = A.T * mp.lu_solve(A * A.T, -F)
    for c, pareja in enumerate(parejas):
        for a, b in pareja:
            P[a][b] += delta[c]
            P[b][a] = P[a][b]
    return P


# biblioteca
def grados_simetricos(P, parejas, j, g):
    """d a partir de un entero por pareja (d_ab = j_c · paso, igual en el par y en su espejo)."""
    n_a = tamanos_enteros(K_S7, g)
    d = [[0] * 7 for _ in range(7)]
    for a in range(7):
        d[a][a] = n_a[a] - 1
        for b in range(7):
            if b != a and P[a][b] == 1:
                d[a][b] = n_a[b]
    for c, pareja in enumerate(parejas):
        for a, b in pareja:
            d[a][b] = j[c] * paso_grado(K_S7, a, b)
            d[b][a] = d[a][b] * n_a[a] // n_a[b]
    return d


def enteros_ingenuos(P, parejas, g):
    """El múltiplo permitido más cercano a P_ab·n_b, para el primer par de cada pareja."""
    n_a = tamanos_enteros(K_S7, g)
    j = []
    for pareja in parejas:
        a, b = pareja[0]
        j.append(int(mp.nint(P[a][b] * n_a[b] / paso_grado(K_S7, a, b))))
    return j


def densidades(d, g):
    """P_ab = d_ab/n_b en mpmath."""
    n_a = tamanos_enteros(K_S7, g)
    return [[mp.mpf(d[a][b]) / n_a[b] for b in range(7)] for a in range(7)]


def base_blanda_reducida(x, psi, P):
    """SVD del Jacobiano reducido: (U_rígido, U_blando, V_rígido) y los valores singulares."""
    J = np.array(jacobiano_reducido(x, psi, P).tolist(), dtype=float)
    U, singulares, Vt = np.linalg.svd(J)
    return (mp.matrix(U[:, :2].tolist()), mp.matrix(U[:, 2:].tolist()), mp.matrix(Vt[:2].T.tolist())), singulares


def residuo_blando_reducido(x, P, psi_referencia, base, iteraciones=30):
    """Newton en las 2 direcciones rígidas; devuelve el residuo de la blanda (un número) y ψ."""
    U_r, U_b, V_r = base
    z = mp.matrix(2, 1)
    for _ in range(iteraciones):
        psi = [psi_referencia[i] + V_r[i, 0] * z[0] + V_r[i, 1] * z[1] for i in range(3)]
        rigidas = U_r.T * ecuaciones_reducidas(x, psi, P)
        if mp.norm(rigidas) < mp.mpf(10) ** (-mp.mp.dps + 10):
            break
        z = z - mp.lu_solve(U_r.T * jacobiano_reducido(x, psi, P) * V_r, rigidas)
    psi = [psi_referencia[i] + V_r[i, 0] * z[0] + V_r[i, 1] * z[1] for i in range(3)]
    return (U_b.T * ecuaciones_reducidas(x, psi, P))[0], psi


def forma_normal(x, psi, P):
    """(σ, c) de la ecuación blanda σ y + (c/2) y² a lo largo de la dirección blanda (diferencias finitas)."""
    U, _, Vt = np.linalg.svd(np.array(jacobiano_reducido(x, psi, P).tolist(), dtype=float))
    u_b = mp.matrix(U[:, 2].tolist())
    v_b = Vt[2]

    def blanda(y):
        return (u_b.T * ecuaciones_reducidas(x, [psi[i] + y * v_b[i] for i in range(3)], P))[0]

    e = mp.mpf("1e-8")
    sigma = (blanda(e) - blanda(-e)) / (2 * e)
    c = (blanda(e) - 2 * blanda(0) + blanda(-e)) / e**2
    return abs(float(sigma)), float(c)


# biblioteca
def newton_entero(x, P, parejas, psi_referencia, base, g, radios=(20, 3)):
    """Newton entero exhaustivo desde el redondeo ingenuo: en cada paso, el u con |u_c| ≤ radio que minimiza |g + G·u|.

    Devuelve los enteros, los grados, ψ y |g| en cada paso.
    """
    j = enteros_ingenuos(P, parejas, g)
    historia = []
    for radio in radios:
        g_actual, _ = residuo_blando_reducido(x, densidades(grados_simetricos(P, parejas, j, g), g), psi_referencia, base)
        G = []
        for c in range(len(parejas)):
            j_mas = list(j)
            j_mas[c] += 1
            g_mas, _ = residuo_blando_reducido(x, densidades(grados_simetricos(P, parejas, j_mas, g), g),
                                               psi_referencia, base)
            G.append(float(g_mas - g_actual))
        historia.append(abs(float(g_actual)))
        candidatos = np.array(list(itertools.product(range(-radio, radio + 1), repeat=len(parejas))))
        u = candidatos[np.argmin(np.abs(float(g_actual) + candidatos @ np.array(G)))]
        j = [int(jc + uc) for jc, uc in zip(j, u)]
    d = grados_simetricos(P, parejas, j, g)
    g_final, psi = residuo_blando_reducido(x, densidades(d, g), psi_referencia, base)
    historia.append(abs(float(g_final)))
    return j, d, psi, historia


def barrido_simetrico(x, P_sdp, s, h, ns):
    """Fase 1 simétrica completa para la plantilla (x, s, P_sdp): pulido, base blanda y, para cada n, ingenuo y Newton."""
    parejas = parejas_libres(simetrizar(P_sdp))
    x_exacto = [mp.mpf(v.numerator) / v.denominator for v in tamanos_exactos(K_S7)]
    psi_plantilla = [mp.mpf(h) * mp.mpf(float(s[0])), mp.mpf(h) * mp.mpf(float(s[1])),
                     mp.pi / 2 + mp.mpf(h) * mp.mpf(float(s[2]))]
    P_pulida = pulir_simetrico(x_exacto, psi_plantilla, simetrizar(P_sdp), parejas)
    base, _ = base_blanda_reducida(x_exacto, psi_plantilla, P_pulida)
    phi_referencia = [float(v) for v in fases_reducidas(psi_plantilla)]
    barrido = []
    for n_objetivo in ns:
        g = int(round(n_objetivo / sum(K_S7)))
        j_ingenuo = enteros_ingenuos(P_pulida, parejas, g)
        d_ingenuo = grados_simetricos(P_pulida, parejas, j_ingenuo, g)
        ingenuo = evaluar_enteros(K_S7, g, d_ingenuo, phi_referencia, h)
        j, d, psi, historia = newton_entero(x_exacto, P_pulida, parejas, psi_plantilla, base, g)
        phi = [float(v) for v in fases_reducidas(psi)]
        newton = evaluar_enteros(K_S7, g, d, phi, h, phi_referencia=phi_referencia)
        barrido.append({"g": g, "n": sum(K_S7) * g, "j": j, "j_ingenuo": j_ingenuo, "d": d, "historia": historia,
                        "ingenuo": ingenuo, "newton": newton})
    return barrido, parejas
