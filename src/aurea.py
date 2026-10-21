"""Familia "girasol": osciladores en el círculo de fases separados por el ángulo áureo.

Las pruebas de los lemas y proposiciones citados están en notes/familia_aurea.md.
  Variante 1: circulantes C_{F_k}(S) con distancias de Fibonacci y el estado áureo.
  Variante 2: grafo geométrico sobre los puntos x_v = v/φ mod 1 (cualquier n).
  Variante 3: plantillas con clusters en las fases áureas (ver red_aurea en busqueda.py).
"""
from itertools import combinations
from math import gcd

import numpy as np
from scipy.optimize import minimize

from .grafos import distancias_circulares, grafo_geometrico_circular
from .kuramoto import autovalores_twisted_conjunto, clasificar, energia, hessiano, velocidad, winding_number

PHI = (1 + 5**0.5) / 2


def fibonacci(k):
    """F_k con F_1 = F_2 = 1."""
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


def es_fibonacci(n):
    """¿n es un número de Fibonacci?"""
    k = 1
    while fibonacci(k) < n:
        k += 1
    return fibonacci(k) == n


def puntos_aureos(n):
    """Posiciones x_v = v/φ mod 1 en [0, 1), v = 0..n−1 (las semillas del girasol vistas en ángulo)."""
    return (np.arange(n) / PHI) % 1


def estado_aureo(n):
    """Fase = posición: θ_v = 2π x_v."""
    return 2 * np.pi * puntos_aureos(n)


def orden_angular(x):
    """Vértices ordenados por su posición en el círculo (para medir el winding number)."""
    return np.argsort(x)


# ---------------------------------------------------------------- variante 1 (teoría)

def multiplicar_conjunto(S, q, n):
    """q·S mod n, simetrizado (Lema 2: el twisted state q en C_n(S) es el q = 1 en C_n(qS))."""
    assert gcd(q, n) == 1, "q tiene que ser coprimo con n"
    resultado = set()
    for s in S:
        resultado.add((q * s) % n)
        resultado.add((-q * s) % n)
    resultado.discard(0)
    return sorted(resultado)


def distancias_fibonacci(k):
    """Distancias de Fibonacci distintas en Z_{F_k}, como representantes en 1..F_k/2."""
    n = fibonacci(k)
    distancias = set()
    for j in range(2, k):
        s = fibonacci(j) % n
        distancias.add(min(s, n - s))
    distancias.discard(0)
    return sorted(distancias)


def simetrizar(distancias, n):
    """{s} → {±s} mod n."""
    S = set()
    for s in distancias:
        S.add(s % n)
        S.add((-s) % n)
    S.discard(0)
    return sorted(S)


def mejor_circulante_fibonacci(k, tol=1e-12):
    """Mayor μ entre los C_{F_k}(S), S ⊂ distancias de Fibonacci, donde el estado áureo es trampa.

    Recorre todos los subconjuntos (sirve para k ≤ ~15). El estado áureo es el twisted state
    q = F_{k−1}. Devuelve dict con μ, el subconjunto óptimo y cuántos subconjuntos dan trampa.
    """
    n = fibonacci(k)
    q = fibonacci(k - 1)
    distancias = distancias_fibonacci(k)
    mejor = {"k": k, "n": n, "mu": 0.0, "distancias": [], "trampas": 0}
    for tamano in range(1, len(distancias) + 1):
        for subconjunto in combinations(distancias, tamano):
            S = simetrizar(subconjunto, n)
            autovalores = autovalores_twisted_conjunto(n, S, q)
            if autovalores[1:].min() <= tol:
                continue
            mejor["trampas"] += 1
            mu = (len(S) + 1) / n
            if mu > mejor["mu"]:
                mejor["mu"] = mu
                mejor["distancias"] = list(subconjunto)
    return mejor


# ---------------------------------------------------------------- variante 2 (grafo geométrico)

def relajar(theta0, W, pasos_newton=20, tol=1e-12):
    """Baja la energía desde theta0 hasta un mínimo local: L-BFGS y después Newton para pulir.

    θ_0 queda fijo (gauge) para sacar el autovalor 0 de la rotación global. Newton solo se usa si el
    Hessiano reducido es definido positivo, para no saltar a una silla.
    """
    theta0 = np.asarray(theta0, dtype=float)

    def completar(z):
        return np.concatenate([[theta0[0]], z])

    resultado = minimize(
        lambda z: energia(completar(z), W),
        theta0[1:],
        jac=lambda z: -velocidad(completar(z), W)[1:],
        method="L-BFGS-B",
        options={"gtol": 1e-10, "maxiter": 20000},
    )
    z = resultado.x
    for _ in range(pasos_newton):
        gradiente = -velocidad(completar(z), W)[1:]
        if np.linalg.norm(gradiente) < tol:
            break
        H = hessiano(completar(z), W)[1:, 1:]
        if np.linalg.eigvalsh(H).min() <= 0:
            break
        z = z - np.linalg.solve(H, gradiente)
    return completar(z)


def trampa_geometrica(n, r):
    """Relaja el estado áureo en el grafo geométrico de radio r y clasifica el punto final.

    Es trampa si el final es un mínimo estricto con winding number ≠ 0 (medido en orden angular).
    """
    x = puntos_aureos(n)
    W = grafo_geometrico_circular(x, r)
    theta = relajar(estado_aureo(n), W)
    tipo, autovalores = clasificar(theta, W)
    orden = orden_angular(x)
    q = winding_number(theta[orden])
    return {
        "n": n,
        "r": float(r),
        "grado_min": int(W.sum(axis=1).min()),
        "mu": float((W.sum(axis=1).min() + 1) / n),
        "tipo": tipo,
        "winding": q,
        "velocidad": float(np.linalg.norm(velocidad(theta, W))),
        "autovalor_min": float(np.sort(autovalores)[1]),
        "trampa": tipo == "mínimo estricto" and q != 0,
    }


def radios_candidatos(n, r_min, r_max):
    """Radios donde el grafo geométrico cambia: puntos medios entre distancias distintas consecutivas."""
    x = puntos_aureos(n)
    distancias = np.unique(np.round(distancias_circulares(x)[np.triu_indices(n, 1)], 12))
    medios = (distancias[:-1] + distancias[1:]) / 2
    return medios[(medios >= r_min) & (medios <= r_max)]


def mejor_radio(n, r_min=0.1, r_max=0.45):
    """Mayor μ con trampa del grafo geométrico áureo de n vértices (None si no hay).

    Como μ no baja al crecer r, recorre los radios de mayor a menor y se queda con la primera trampa.
    """
    for r in radios_candidatos(n, r_min, r_max)[::-1]:
        resultado = trampa_geometrica(n, r)
        if resultado["trampa"]:
            return resultado
    return None
