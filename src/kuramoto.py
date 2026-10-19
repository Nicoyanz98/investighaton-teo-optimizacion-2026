"""Modelo de Kuramoto homogéneo sobre un grafo con pesos W: dinámica, energía, Hessiano y twisted states."""
import numpy as np

from .grafos import grafo_circulante


# ---------------------------------------------------------------- dinámica y energía

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


def wrap(x):
    """Lleva ángulos a (−π, π]."""
    return np.angle(np.exp(1j * x))


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


# ---------------------------------------------------------------- twisted states

def twisted_state(n, q):
    """θ_v = 2π q v / n."""
    return 2 * np.pi * q * np.arange(n) / n


def winding_number(theta):
    """q = (1/2π) Σ_v wrap(θ_{v+1} − θ_v), recorriendo el ciclo una vez."""
    saltos = wrap(np.roll(theta, -1) - theta)
    return int(np.rint(saltos.sum() / (2 * np.pi)))


def autovalores_twisted(n, k, q):
    """Fórmula cerrada en C_{n,k}: λ_m = Σ_j cos(j d)(2 − 2 cos(2π j m / n)), d = 2πq/n, m = 0..n−1."""
    d = 2 * np.pi * q / n
    m = np.arange(n)[:, None]
    j = np.arange(1, k + 1)[None, :]
    terminos = np.cos(j * d) * (2 - 2 * np.cos(2 * np.pi * j * m / n))
    return terminos.sum(axis=1)


def es_trampa(n, k, q):
    """¿El twisted state q es mínimo estricto en C_{n,k}? (todos los λ_m con m ≠ 0 positivos)."""
    return autovalores_twisted(n, k, q)[1:].min() > 1e-12


def k_max(n):
    """Mayor k con algún twisted state q ≠ 0 estable en C_{n,k}.

    Devuelve (k_max, lista de k estables, q estables en k_max).
    """
    qs = range(1, n // 2 + 1)
    ks_estables = []
    for k in range(1, (n - 1) // 2 + 1):
        if any(es_trampa(n, k, q) for q in qs):
            ks_estables.append(k)
    if not ks_estables:
        return 0, [], []
    ultimo = ks_estables[-1]
    qs_ultimo = []
    for q in qs:
        if es_trampa(n, ultimo, q):
            qs_ultimo.append(q)
    return ultimo, ks_estables, qs_ultimo


def k_max_clasificando(n):
    """Igual que k_max pero clasificando con el Hessiano numérico (sin fórmula cerrada)."""
    mejor = 0
    for k in range(1, (n - 1) // 2 + 1):
        W = grafo_circulante(n, k)
        for q in range(1, n // 2 + 1):
            if clasificar(twisted_state(n, q), W)[0] == "mínimo estricto":
                mejor = k
                break
    return mejor


def k_max_q1(n):
    """k_max solo para q = 1, acumulando la suma de λ_m al crecer k (sirve para n grande)."""
    m = np.arange(n)
    d = 2 * np.pi / n
    autovalores = np.zeros(n)
    mejor = 0
    for k in range(1, (n - 1) // 2 + 1):
        autovalores += np.cos(k * d) * (2 - 2 * np.cos(2 * np.pi * k * m / n))
        if autovalores[1:].min() <= 1e-12:
            break
        mejor = k
    return mejor


# ---------------------------------------------------------------- clasificación abreviada

ABREVIATURA = {
    "mínimo estricto": "mín",
    "máximo estricto": "máx",
    "máximo (degenerado)": "máx",
    "silla": "silla",
    "degenerado": "deg",
}


def tipo_predicho_ciclo(n, q):
    """Tipo del twisted state q en C_n según el signo de cos(2πq/n): mín, máx o deg."""
    c = np.cos(2 * np.pi * q / n)
    if c > 1e-12:
        return "mín"
    if c < -1e-12:
        return "máx"
    return "deg"


def autovalor_no_nulo_min(autovalores):
    """Menor autovalor sacando el más cercano a 0 (la rotación global)."""
    resto = np.delete(autovalores, np.argmin(np.abs(autovalores)))
    menor = resto.min()
    return menor if abs(menor) > 1e-12 else 0.0
