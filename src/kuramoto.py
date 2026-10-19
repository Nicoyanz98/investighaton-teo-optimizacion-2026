"""Funciones básicas de Kuramoto homogéneo.

Copiadas de notebooks/Calentamientos.ipynb (secciones 1–3) para reusarlas desde módulos y otros
notebooks. El notebook original no se modifica.
"""
import numpy as np


def complete_graph(n):
    # K_n: todos con todos, sin self-loops
    return np.ones((n, n)) - np.eye(n)


def phase_diff(theta):
    # D[v, u] = θ_u − θ_v
    return theta[None, :] - theta[:, None]


def velocity(theta, W):
    # θ̇_v = Σ_u w_uv sin(θ_u − θ_v)   ( = −∇E )
    return (W * np.sin(phase_diff(theta))).sum(axis=1)


def energy(theta, W):
    # E = ½ Σ_{u,v} w_uv (1 − cos(θ_u − θ_v))
    return 0.5 * (W * (1 - np.cos(phase_diff(theta)))).sum()


def order_parameter(theta):
    # r = (1/n) Σ e^{iθ_v};  |r| = cuán agrupadas,  ψ = arg r = hacia dónde
    return np.exp(1j * theta).mean()


def hessian(theta, W):
    # Laplaciano con pesos w_uv cos(θ_u − θ_v)
    C = W * np.cos(phase_diff(theta))
    return np.diag(C.sum(axis=1)) - C


def wrap(x):
    # lleva ángulos a (−π, π]
    return np.angle(np.exp(1j * x))


def simulate(theta0, W, h=None, tol=1e-10, max_steps=200_000, record=True):
    # Descenso de gradiente sobre E desde theta0. Devuelve el estado final y (opcional) la historia.
    if h is None:
        h = 1.0 / (2 * W.sum(axis=1).max())
    theta = np.array(theta0, dtype=float)
    history = [theta.copy()]
    converged = False
    for step in range(1, max_steps + 1):
        v = velocity(theta, W)
        if np.linalg.norm(v) < tol:
            converged = True
            break
        theta = theta + h * v
        if record:
            history.append(theta.copy())
    out = {"theta": theta, "steps": step, "converged": converged, "h": h}
    if record:
        H = np.array(history)
        out["t"] = h * np.arange(len(H))
        out["history"] = H
        out["E"] = np.array([energy(th, W) for th in H])
        out["r"] = np.array([order_parameter(th) for th in H])
    return out


def classify(theta, W, tol=1e-6):
    # Clasifica un equilibrio por los autovalores del Hessiano (el 0 de la rotación global siempre está).
    eig = np.linalg.eigvalsh(hessian(theta, W))
    scale = max(1.0, np.abs(eig).max())
    neg = int(np.sum(eig < -tol * scale))
    pos = int(np.sum(eig > tol * scale))
    zero = len(eig) - neg - pos
    if neg == 0 and zero == 1:
        label = "mínimo estricto"
    elif pos == 0 and zero == 1:
        label = "máximo estricto"
    elif neg > 0:
        label = "silla" if pos > 0 else "máximo (degenerado)"
    else:
        label = "degenerado"
    return label, eig


def cycle_graph(n):
    # C_n: v conectado con v−1 y v+1 (módulo n)
    W = np.zeros((n, n))
    v = np.arange(n)
    W[v, (v + 1) % n] = 1
    W[(v + 1) % n, v] = 1
    return W


def twisted_state(n, q):
    """θ_v = 2π q v / n"""
    return 2 * np.pi * q * np.arange(n) / n


def winding_number(theta):
    """q = (1/2π) Σ_v wrap(θ_{v+1} − θ_v)"""
    d = wrap(np.roll(theta, -1) - theta)
    return int(np.rint(d.sum() / (2 * np.pi)))


def circulant_graph(n, k):
    # C_{n,k}: v conectado con v±1, …, v±k (módulo n)
    W = np.zeros((n, n))
    v = np.arange(n)
    for j in range(1, k + 1):
        W[v, (v + j) % n] = 1
        W[(v + j) % n, v] = 1
    return W


def twisted_eigs(n, k, q):
    """λ_m = Σ_j cos(j d) (2 − 2 cos(2π j m / n)),  m = 0..n−1"""
    d = 2 * np.pi * q / n
    m = np.arange(n)[:, None]
    j = np.arange(1, k + 1)[None, :]
    return (np.cos(j * d) * (2 - 2 * np.cos(2 * np.pi * j * m / n))).sum(axis=1)
