"""Reconstrucción independiente del estado S7 de O'Keeffe (arXiv 2610.21271, Tabla II y Apéndice A).

Clusters en el orden (Am, Ap, Bm, Bp, Cm, Cp, W):
  tamaños  x = (a, a, a, a, a, a, 1 − 6a),  a = 1/(7 + q)
  fases    φ_c = j_c π/2 + h s_c,  j = (0,0,1,1,2,2,3),  s = (−1, 1, −w, w, −1, 1, 0)
  densidades P(h) = p + h p′ + Δ(h)
con q = q* − t, w = w*, u = u13(q, w), κ = 19/10. Δ = O(h²) vive en el camino
Am–Bm–Cm–W–Cp–Bp–Ap y se elige para que el equilibrio sea exacto (F es afín en P).
"""
import mpmath as mp

NAMES = ["Am", "Ap", "Bm", "Bp", "Cm", "Cp", "W"]
IDX = {n: i for i, n in enumerate(NAMES)}
J = [0, 0, 1, 1, 2, 2, 3]
PATH = [("Am", "Bm"), ("Bm", "Cm"), ("Cm", "W"), ("W", "Cp"), ("Cp", "Bp"), ("Bp", "Ap")]


def f_e(q, w):
    f = -2 * q**2 * w**2 - 4 * q**2 * w - 3 * q**2 - 2 * q + 4 * w + 1
    e = -q**2 * w**2 + q**2 - 2 * q * w**2 + 6 * q - w**2 + 1
    return f, e


def u13(q, w):
    return (2 * q**2 * w**2 + 2 * q**2 * w - q**2 - 6 * q + 4 * w**2 - 2 * w + 3) / 2


def poly3(q):
    return 9*q**8 + 72*q**7 + 288*q**6 + 448*q**5 + 470*q**4 + 88*q**3 - 208*q**2 - 128*q - 15


def corner(dps=50):
    """(q*, w*, μ0): punto donde los dos dipolos del modelo límite son exactamente neutros."""
    with mp.workdps(dps):
        q, w = mp.findroot(lambda q, w: f_e(q, w), (mp.mpf("0.65"), mp.mpf("1.4")))
        return q, w, (4 + 2 * q) / (7 + q)


def limit_pattern(q):
    """Patrón límite de la Tabla II (1 completo, q adelgazado, 0 vacío)."""
    rows = [
        [1, 1, 1, q, 0, 0, 1],
        [1, 1, q, 1, 0, 0, 1],
        [1, q, 1, 1, 1, q, 0],
        [q, 1, 1, 1, q, 1, 0],
        [0, 0, 1, q, 1, 1, 1],
        [0, 0, q, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1],
    ]
    return mp.matrix([[mp.mpf(v) for v in r] for r in rows])


def first_order(q, w, a, xW, kappa=mp.mpf(19) / 10):
    u = u13(q, w)
    A1 = w * (w + 1) - u / 2
    A2 = u / 2 - w * (w - 1)
    pp = mp.matrix(7, 7)

    def put(n1, n2, val):
        pp[IDX[n1], IDX[n2]] = val
        pp[IDX[n2], IDX[n1]] = val

    put("Am", "Bm", A1 - kappa); put("Bp", "Cp", A1 - kappa)
    put("Ap", "Bm", A2 - kappa); put("Bp", "Cm", A2 - kappa)
    for n1, n2 in [("Ap", "Bp"), ("Bm", "Cm"), ("Am", "Bp"), ("Bm", "Cp")]:
        put(n1, n2, -kappa)
    put("Am", "W", a / xW * (A1 + 2 - 2 * kappa)); put("Cp", "W", a / xW * (A1 + 2 - 2 * kappa))
    put("Ap", "W", a / xW * (A2 - 2 - 2 * kappa)); put("Cm", "W", a / xW * (A2 - 2 - 2 * kappa))
    return pp


def forces(x, phi, P):
    m = len(x)
    return [mp.fsum(P[a, b] * x[b] * mp.sin(phi[a] - phi[b]) for b in range(m)) for a in range(m)]


def build(t, h, dps=60, solve_delta=True):
    """Plantilla S7 para (t, h). Devuelve dict con x, phi, P (mpmath) y parámetros."""
    with mp.workdps(dps):
        t, h = mp.mpf(t), mp.mpf(h)
        qs, ws, _ = corner(dps)
        q, w = qs - t, ws
        a = 1 / (7 + q)
        xW = 1 - 6 * a
        x = [a] * 6 + [xW]
        s = [-1, 1, -w, w, -1, 1, 0]
        phi = [J[c] * mp.pi / 2 + h * s[c] for c in range(7)]
        P = limit_pattern(q) + h * first_order(q, w, a, xW)
        F_first = forces(x, phi, P)
        if solve_delta:
            # F es afín en las 6 entradas del camino: resolver J Δ = −F (descartando la ecuación de W,
            # que se deduce de Σ x_a F_a = 0)
            cols = []
            for n1, n2 in PATH:
                E = mp.matrix(7, 7)
                E[IDX[n1], IDX[n2]] = 1
                E[IDX[n2], IDX[n1]] = 1
                cols.append(forces(x, phi, E))
            Jm = mp.matrix([[cols[k][r] for k in range(6)] for r in range(6)])
            delta = mp.lu_solve(Jm, mp.matrix([-v for v in F_first[:6]]))
            for k, (n1, n2) in enumerate(PATH):
                P[IDX[n1], IDX[n2]] += delta[k]
                P[IDX[n2], IDX[n1]] += delta[k]
        else:
            delta = None
        return {"x": x, "phi": phi, "P": P, "q": q, "w": w, "t": t, "h": h,
                "F_first": F_first, "delta": delta}


def analyze(S, dps=60):
    """μ, equilibrio, c_a y autovalores de L (en ⊥ de la rotación, vía M = X^{-1/2} L X^{-1/2} + v vᵀ)."""
    with mp.workdps(dps):
        x, phi, P = S["x"], S["phi"], S["P"]
        m = len(x)
        d = [mp.fsum(P[a, b] * x[b] for b in range(m)) for a in range(m)]
        F = forces(x, phi, P)
        c = [mp.fsum(P[a, b] * x[b] * mp.cos(phi[a] - phi[b]) for b in range(m)) for a in range(m)]
        L = mp.matrix(m, m)
        for a in range(m):
            for b in range(m):
                if a != b:
                    L[a, b] = -x[a] * x[b] * P[a, b] * mp.cos(phi[a] - phi[b])
            L[a, a] = -mp.fsum(L[a, b] for b in range(m) if b != a)
        M = mp.matrix(m, m)
        v = [mp.sqrt(xa) for xa in x]
        for a in range(m):
            for b in range(m):
                M[a, b] = L[a, b] / (v[a] * v[b])
        eig = sorted(mp.eigsy(M)[0])  # el primero es el 0 de la rotación
        Pmin = min(P[a, b] for a in range(m) for b in range(m))
        Pmax = max(P[a, b] for a in range(m) for b in range(m))
        return {"mu": min(d), "max|F|": max(abs(f) for f in F), "min c": min(c),
                "eig": eig, "P range": (Pmin, Pmax)}
