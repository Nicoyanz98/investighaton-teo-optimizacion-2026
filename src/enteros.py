"""Plantillas S7 con datos enteros: el paso de una plantilla a un grafo de n vértices concreto.

Estructura (orden de clusters de src/s7.py: Am, Ap, Bm, Bp, Cm, Cp, W):
  tamaños   n_a = k_a·g, con k = (k_A, …, k_A, k_W): seis mitades iguales y W
  grados    d_ab = vecinos que tiene un vértice de a en b (entero), con n_a d_ab = n_b d_ba
            y d_aa = n_a − 1 (cada cluster es un clique)
Con eso la plantilla del grafo es x_a = n_a/n y P_ab = d_ab/n_b, y μ = (δ + 1)/n = min_a d_a.

Los tamaños se fijan racionales desde el principio (x_a = k_a/K) y la optimización solo mueve los
desplazamientos s y las densidades P, así que lo único que hay que redondear son los grados.
"""
from fractions import Fraction
from math import gcd

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

from .busqueda import FamiliaParticiones, base_c4, sdp_interno
from .certificado import fases_newton
from .plantillas import analizar, fuerzas, jacobiano_fuerzas, matriz_clusters, splitting
from .reticulado import enteros_cercanos
from .s7 import esquina

PIEZAS_S7 = (2, 2, 2, 1)


# ---------------------------------------------------------------- plantilla con tamaños racionales

def tamanos_racionales(k_mitad, k_w):
    """x = (k_mitad, …, k_mitad, k_w)/(6 k_mitad + k_w) como array."""
    k = np.array([k_mitad] * 6 + [k_w], dtype=float)
    return k / k.sum()


def desplazamientos_iniciales():
    """Ansatz de S7: s = (−1, 1, −w*, w*, −1, 1) para las seis mitades."""
    _, w, _ = esquina(30)
    w = float(w)
    return np.array([-1.0, 1.0, -w, w, -1.0, 1.0])


def fases_s7(h, s):
    """φ = fase del sector + h·s (W queda en 3π/2)."""
    familia = FamiliaParticiones(base_c4(), PIEZAS_S7, h)
    desplazamientos = np.concatenate([s, [0.0]])
    return familia.base.fases[familia.sector] + h * desplazamientos


def prohibidos_s7():
    """Pares de clusters sin links (sectores opuestos de C4)."""
    return FamiliaParticiones(base_c4(), PIEZAS_S7, 0.01).prohibidos


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


def _a_mp(valor):
    """Fraction o float → mpf (exacto para Fraction)."""
    if isinstance(valor, Fraction):
        return mp.mpf(valor.numerator) / valor.denominator
    return mp.mpf(float(valor))


def tamanos_exactos(k):
    """x_a = k_a/K como Fraction."""
    K = sum(int(ka) for ka in k)
    return [Fraction(int(ka), K) for ka in k]


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


# ---------------------------------------------------------------- datos enteros

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


# ---------------------------------------------------------------- redondeo por reticulado
#
# Cerca de S7 el Jacobiano J del equilibrio tiene valores singulares muy chicos: el mínimo estricto
# está al lado de un pliegue (silla-nodo). Si el error de redondeo empuja el residuo de las ecuaciones
# blandas más allá de ~λ²/c, el equilibrio deja de existir. Reducción de Lyapunov–Schmidt: con
# J = U S Vᵀ, las direcciones rígidas (valores singulares grandes) se resuelven exactamente, y queda el
# residuo blando g = U_blandoᵀ F, bien definido aunque no haya equilibrio y casi lineal en los grados.

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
