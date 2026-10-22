"""Certificado riguroso de que una plantilla racional (x, P) tiene un equilibrio estricto no sincronizado.

Implementación independiente del esquema del Apéndice B de O'Keeffe (arXiv 2610.21271):
  1. Datos racionales exactos: x_a = X_a/D, P_ab = Q_ab/D (fractions.Fraction).
  2. Krawczyk (mpmath.iv, redondeo hacia afuera): hay una única solución de F_a = 0, a = 1..m−1,
     con φ_0 fijo, dentro de una caja explícita. F_0 = 0 sigue de Σ x_a F_a ≡ 0.
  3. Sobre toda la caja: c_a > 0, y la matriz de L sin la fila/columna 0 (gauge y_0 = 0) admite
     Cholesky por intervalos con pivotes positivos ⇒ L(y) > 0 salvo rotaciones.
  4. Grados d_a y μ = min d_a en aritmética racional exacta.
  5. Las fases son distintas dos a dos sobre toda la caja (el estado no es sync).
"""
from fractions import Fraction

import mpmath as mp
from mpmath import iv

from .plantillas import fuerzas, grados, jacobiano_fuerzas, matriz_clusters, splitting


# ---------------------------------------------------------------- datos racionales y fases

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


# ---------------------------------------------------------------- pasos del certificado

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


def _menor_separacion(caja):
    """Cota inferior de min_{a<b} |sin((φ_a − φ_b)/2)| sobre la caja (> 0 ⇒ fases distintas)."""
    m = len(caja)
    menor = None
    for a in range(m):
        for b in range(a + 1, m):
            cota = _cota_inferior(abs(iv.sin((caja[a] - caja[b]) / 2)))
            menor = cota if menor is None else min(menor, cota)
    return menor


# ---------------------------------------------------------------- certificado completo

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

    resultado["min |sin(Δφ/2)| (cota inf)"] = _menor_separacion(caja)
    resultado["distintas"] = resultado["min |sin(Δφ/2)| (cota inf)"] > 0

    resultado["mu"] = min(grados(xq, Pq))
    resultado["caja"] = caja
    resultado["ok"] = (resultado["krawczyk"] and resultado["c>0"]
                       and resultado["cholesky"] and resultado["distintas"])
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
