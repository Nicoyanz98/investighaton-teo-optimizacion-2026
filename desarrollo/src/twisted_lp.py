"""Twisted states como plantilla: m clusters iguales en 2πqa/m con densidades circulantes, resuelto como LP.

Por simetría el equilibrio es automático y los autovalores de L son lineales en las densidades p_j
(j = distancia entre clusters), así que maximizar el grado es un LP.
"""
import numpy as np
from scipy.optimize import linprog


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
