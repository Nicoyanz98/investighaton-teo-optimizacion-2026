"""Reticulados chicos (dimensión ≲ 30) en float: reducción LLL y vector más cercano de Babai.

Se usan para el redondeo inteligente de src/enteros.py: elegir enteros cerca de un objetivo real de
modo que unas pocas formas lineales queden casi en cero. Los vectores del reticulado son las filas de B.
"""
import numpy as np


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
