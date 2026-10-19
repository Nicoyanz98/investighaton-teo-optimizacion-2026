"""Certificado riguroso de que una plantilla racional (x, P) tiene un equilibrio estricto.

Implementación independiente del esquema del Apéndice B de O'Keeffe (arXiv 2610.21271):
  1. Datos racionales exactos: x_a = X_a/D, P_ab = Q_ab/D (fractions.Fraction).
  2. Krawczyk (mpmath.iv, redondeo hacia afuera): existe una única solución de F_a = 0, a = 2..m,
     con φ_1 fijo, en una caja explícita. F_1 = 0 sigue de Σ x_a F_a ≡ 0.
  3. Sobre toda la caja: c_a > 0 y la matriz de L sin la fila/columna 1 (gauge y_1 = 0) admite
     Cholesky por intervalos con pivotes positivos ⇒ L(y) > 0 salvo rotaciones.
  4. Grados d_a y μ = min d_a en aritmética racional exacta.
  5. Las fases son distintas dos a dos sobre toda la caja (el estado no es sync).
"""
from fractions import Fraction

import mpmath as mp
from mpmath import iv


def rationalize(x, P, D, diag_max=True):
    """Redondea (x, P) a denominador común D (par). Σ x = 1 exacto ajustando el último cluster.

    Si diag_max, las diagonales quedan en (D − 1)/D (clique completo, como pide el Lema 2).
    """
    assert D % 2 == 0
    m = len(x)
    with mp.workdps(len(str(D)) + 30):  # precisión suficiente para redondear a 1/D
        X = [int(mp.nint(xa * D)) for xa in x[:-1]]
        X.append(D - sum(X))
        Q = [[0] * m for _ in range(m)]
        for a in range(m):
            for b in range(a, m):
                v = int(mp.floor(P[a, b] * D)) if a == b else int(mp.nint(P[a, b] * D))
                if a == b and diag_max:
                    v = D - 1
                v = max(0, min(D - (a == b), v))
                Q[a][b] = Q[b][a] = v
    return [Fraction(Xa, D) for Xa in X], [[Fraction(Q[a][b], D) for b in range(m)] for a in range(m)]


def _F(x, P, phi, a, lib):
    return sum(P[a][b] * x[b] * lib.sin(phi[a] - phi[b]) for b in range(len(x)) if b != a)


def _jac(x, P, phi, lib):
    # ∂F_a/∂φ_b para a, b = 1..m−1 (índices 0-based 1..m−1; φ_0 fijo)
    m = len(x)
    Jm = [[None] * (m - 1) for _ in range(m - 1)]
    for a in range(1, m):
        for b in range(1, m):
            if a == b:
                Jm[a - 1][b - 1] = sum(P[a][k] * x[k] * lib.cos(phi[a] - phi[k]) for k in range(m) if k != a)
            else:
                Jm[a - 1][b - 1] = -P[a][b] * x[b] * lib.cos(phi[a] - phi[b])
    return Jm


def newton_phases(xq, Pq, phi0, dps=120, iters=60):
    """Refina las fases (con φ_0 fijo) para los datos racionales, en mpmath de alta precisión."""
    with mp.workdps(dps):
        x = [mp.mpf(v.numerator) / v.denominator for v in xq]
        P = [[mp.mpf(v.numerator) / v.denominator for v in row] for row in Pq]
        phi = [mp.mpf(p) for p in phi0]
        m = len(x)
        for _ in range(iters):
            Fv = mp.matrix([_F(x, P, phi, a, mp) for a in range(1, m)])
            if mp.norm(Fv) < mp.mpf(10) ** (-dps + 10):
                break
            step = mp.lu_solve(mp.matrix(_jac(x, P, phi, mp)), Fv)
            for a in range(1, m):
                phi[a] -= step[a - 1]
        return phi


def _lo(v):
    # cota inferior de un intervalo como mpf
    return mp.mpf(v.a.a)


def _to_iv(fr):
    return iv.mpf(fr.numerator) / fr.denominator


def certify(xq, Pq, phi_center, radius="1e-40", dps=100, verbose=True):
    """Devuelve dict con el resultado del certificado. ok = True solo si todos los pasos pasan."""
    out = {}
    m = len(xq)
    iv.dps = dps
    mp.mp.dps = dps
    x = [_to_iv(v) for v in xq]
    P = [[_to_iv(v) for v in row] for row in Pq]
    r = mp.mpf(radius)
    c = [mp.mpf(p) for p in phi_center]
    box = [iv.mpf(c[0])] + [iv.mpf([c[a] - r, c[a] + r]) for a in range(1, m)]
    mid = [iv.mpf(c[a]) for a in range(m)]

    # --- Krawczyk: K = m − Y F(m) + (I − Y J(X)) (X − m) ⊂ int X
    Fm = [_F(x, P, mid, a, iv) for a in range(1, m)]
    Jc = mp.matrix([[mp.mpf(v.mid) for v in row] for row in _jac(x, P, mid, iv)])
    Y = Jc ** -1
    JX = _jac(x, P, box, iv)
    n = m - 1
    K_inside = True
    widths = []
    for i in range(n):
        acc = iv.mpf(c[i + 1])
        acc -= sum(iv.mpf(Y[i, k]) * Fm[k] for k in range(n))
        for j in range(n):
            Mij = (1 if i == j else 0) - sum(iv.mpf(Y[i, k]) * JX[k][j] for k in range(n))
            acc += Mij * (box[j + 1] - mid[j + 1])
        widths.append(acc.delta)
        if not (acc.a > box[i + 1].a and acc.b < box[i + 1].b):
            K_inside = False
    out["krawczyk"] = K_inside

    # --- c_a > 0 sobre la caja
    cs = [sum(P[a][b] * x[b] * iv.cos(box[a] - box[b]) for b in range(m) if b != a) + P[a][a] * x[a]
          for a in range(m)]
    out["min c (cota inf)"] = min(_lo(cv) for cv in cs)
    out["c>0"] = all(cv.a > 0 for cv in cs)

    # --- Cholesky por intervalos de L sin fila/columna 0
    L = [[None] * m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            if a != b:
                L[a][b] = -x[a] * x[b] * P[a][b] * iv.cos(box[a] - box[b])
        L[a][a] = -sum(L[a][b] for b in range(m) if b != a)
    A = [[L[i][j] for j in range(1, m)] for i in range(1, m)]
    pivots = []
    chol_ok = True
    Lc = [[iv.mpf(0)] * n for _ in range(n)]
    for j in range(n):
        s = A[j][j] - sum(Lc[j][k] ** 2 for k in range(j))
        pivots.append(s)
        if not s.a > 0:
            chol_ok = False
            break
        Lc[j][j] = iv.sqrt(s)
        for i in range(j + 1, n):
            Lc[i][j] = (A[i][j] - sum(Lc[i][k] * Lc[j][k] for k in range(j))) / Lc[j][j]
    out["cholesky"] = chol_ok
    out["min pivote (cota inf)"] = min(_lo(p) for p in pivots)

    # --- grados exactos
    d = [sum(Pq[a][b] * xq[b] for b in range(m)) for a in range(m)]
    out["mu"] = min(d)

    # --- fases distintas
    sep = min(_lo(abs(iv.sin((box[a] - box[b]) / 2))) for a in range(m) for b in range(a + 1, m))
    out["min |sin(Δφ/2)| (cota inf)"] = sep
    out["distinct"] = sep > 0

    out["ok"] = K_inside and out["c>0"] and chol_ok and out["distinct"]
    if verbose:
        for k, v in out.items():
            if k != "mu":
                print(f"  {k:28s} {mp.nstr(v, 6) if not isinstance(v, bool) else v}")
        print(f"  {'mu (exacto)':28s} {float(out['mu']):.16f}  (denominador {out['mu'].denominator.bit_length()} bits)")
    return out
