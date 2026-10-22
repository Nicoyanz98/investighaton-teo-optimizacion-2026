"""Blow-up circulante explícito de datos enteros (k, g, d) y cotas para el Hessiano del grafo.

Construcción (generaliza `plantillas.realizar` a clusters de tamaños distintos):
  - El cluster a tiene k_a bloques, cada uno una copia de Z_g. Vértice (a, i, v), v ∈ Z_g.
  - Dentro de un cluster: clique (cada bloque es Z_g ∖ {0} y bloques distintos se unen completos).
  - Entre clusters a < b: (a, i, v) ~ (b, j, v + s) con s ∈ N_ij = {0, …, t_ij − 1} (intervalo, como
    el k-spinning de Canale). La tabla t (k_a × k_b) tiene filas que suman d_ab y columnas que suman
    d_ba (problema de transporte), así que cada vértice de a tiene exactamente d_ab vecinos en b.

El Hessiano H = diag(Σ_v W_uv cos) − [W_uv cos(θ_u − θ_v)] conmuta con el corrimiento v → v + 1 en
todos los bloques, así que se parte en g matrices K × K (K = Σ k_a), una por frecuencia m:
  H_m[(a,i),(b,j)] = δ·Σ_c d_ac cos_ac − cos_ab · N̂_ij(m),   N̂(m) = Σ_{s∈N} e^{2πi m s/g}.
"""
from fractions import Fraction

import mpmath as mp
import numpy as np
from mpmath import iv

from .certificado import certificar, fases_newton
from .enteros import mu_entero, plantilla_entera, tamanos_enteros

UMBRAL = Fraction(11, 16)


# ---------------------------------------------------------------- construcción

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


def fases_vertices(k, g, fases):
    """θ del grafo: cada vértice toma la fase de su cluster."""
    return np.repeat(np.asarray(fases, dtype=float), tamanos_enteros(k, g))


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


# ---------------------------------------------------------------- Hessiano por bloques de Fourier

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


# ---------------------------------------------------------------- cotas de los modos que no son de cluster

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


# ---------------------------------------------------------------- certificado del grafo completo

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
