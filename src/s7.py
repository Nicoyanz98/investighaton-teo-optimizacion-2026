"""Reconstrucción independiente del estado S7 de O'Keeffe (arXiv 2610.21271, Tabla II y Apéndice A).

Clusters en el orden (Am, Ap, Bm, Bp, Cm, Cp, W):
  tamaños   x = (a, a, a, a, a, a, 1 − 6a),  a = 1/(7 + q)
  fases     φ_c = j_c π/2 + h s_c,  j = (0,0,1,1,2,2,3),  s = (−1, 1, −w, w, −1, 1, 0)
  densidades P(h) = p + h p′ + Δ(h)
con q = q* − t, w = w*, u = u13(q, w), κ = 19/10. Δ = O(h²) vive en el camino
Am–Bm–Cm–W–Cp–Bp–Ap y se elige para que el equilibrio sea exacto (F es afín en P).
"""
import mpmath as mp

from .plantillas import fuerzas

NOMBRES = ["Am", "Ap", "Bm", "Bp", "Cm", "Cp", "W"]
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
