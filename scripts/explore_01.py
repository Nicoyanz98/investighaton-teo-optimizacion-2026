"""Caso 0/1 (la idea original del "stackeo"): blow-ups completos de un grafo base H.

Cada vértice de H es un clique y cada arista de H un bipartito completo: P = A_H + I, sin
densidades intermedias. Para cada H conexo con k vértices (generado por nauty) se busca
    max μ = min_a d_a   sobre tamaños x y fases φ (φ_0 = 0)
    s.t. F = 0,  c_a ≥ eps,  λ_min(L en ⊥1) ≥ eps,  fases no todas iguales
con SLSQP desde varios arranques al azar. Prefiltro: el LP max_x min_a x(N[a]) tiene que superar
`floor` (ignorar estabilidad solo puede dar más).

Uso:  .venv/bin/python scripts/explore_01.py k [starts] [floor]
Salida: data/explore_01_k{k}.json (los mejores H, con x y φ).
"""
import json
import os
import subprocess
import sys
import time
import warnings
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy.optimize import linprog, minimize

ROOT = os.path.join(os.path.dirname(__file__), "..")
GENG = os.path.join(ROOT, "tools", "nauty", "geng")


def lp_mu(M):
    k = len(M)
    c = np.zeros(k + 1)
    c[-1] = -1
    res = linprog(c, A_ub=np.hstack([-M, np.ones((k, 1))]), b_ub=np.zeros(k),
                  A_eq=[list(np.ones(k)) + [0]], b_eq=[1], bounds=[(0, 1)] * (k + 1), method="highs")
    return -res.fun, res.x[:k]


def _perp(k):
    Q, _ = np.linalg.qr(np.eye(k) - 1.0 / k)
    return Q[:, : k - 1]


def optimize_graph(M, starts, rng, eps=1e-6, eps_c=1e-3):
    k = len(M)
    Q = _perp(k)

    def unpack(v):
        x = np.exp(v[:k] - v[:k].max())
        x /= x.sum()
        phi = np.concatenate([[0.0], v[k: 2 * k - 1]])
        return x, phi, v[-1]

    def parts(v):
        x, phi, t = unpack(v)
        D = phi[:, None] - phi[None, :]
        F = (M * np.sin(D)) @ x
        c = (M * np.cos(D)) @ x
        K = np.outer(x, x) * M * np.cos(D)
        np.fill_diagonal(K, 0)
        L = np.diag(K.sum(1)) - K
        lam = np.linalg.eigvalsh(Q.T @ L @ Q).min()
        return x, phi, t, F, c, lam

    cons = [
        {"type": "ineq", "fun": lambda v: M @ unpack(v)[0] - v[-1]},
        {"type": "eq", "fun": lambda v: parts(v)[3][1:]},
        {"type": "ineq", "fun": lambda v: parts(v)[4] - eps_c},
        {"type": "ineq", "fun": lambda v: np.array([parts(v)[5] - eps])},
    ]
    _, x_lp = lp_mu(M)
    best = None
    for _ in range(starts):
        x0 = 0.5 * x_lp + 0.5 / k + 0.02 * rng.random(k)
        v0 = np.concatenate([np.log(x0), rng.uniform(-np.pi, np.pi, k - 1), [0.5]])
        try:
            res = minimize(lambda v: -v[-1], v0, constraints=cons, method="SLSQP",
                           options={"maxiter": 300, "ftol": 1e-10})
        except Exception:
            continue
        x, phi, t, F, c, lam = parts(res.x)
        spread = np.abs(np.exp(1j * phi).mean())
        ok = (np.abs(F).max() < 1e-8 and c.min() > eps_c / 2 and lam > eps / 2 and spread < 1 - 1e-6)
        if ok:
            mu = (M @ x).min()
            if best is None or mu > best["mu"]:
                best = {"mu": float(mu), "x": x.tolist(), "phi": phi.tolist(), "lam": float(lam)}
    return best


def work(args):
    g6, starts, seed = args
    warnings.filterwarnings("ignore")
    G = nx.from_graph6_bytes(g6.encode())
    M = nx.to_numpy_array(G) + np.eye(G.number_of_nodes())
    best = optimize_graph(M, starts, np.random.default_rng(seed))
    return g6, best


if __name__ == "__main__":
    k = int(sys.argv[1])
    starts = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 11 / 16
    t0 = time.time()
    graphs = subprocess.run([GENG, "-cq", str(k)], capture_output=True, text=True).stdout.split()
    keep = []
    for g6 in graphs:
        G = nx.from_graph6_bytes(g6.encode())
        M = nx.to_numpy_array(G) + np.eye(k)
        if M.min() == 1:  # H completo: sincroniza
            continue
        if lp_mu(M)[0] > floor + 1e-9:
            keep.append(g6)
    print(f"k = {k}: {len(graphs)} grafos conexos, {len(keep)} pasan el prefiltro LP > {floor:.4f}", flush=True)
    found = []
    with Pool(6) as pool:
        for i, (g6, best) in enumerate(pool.imap_unordered(work, [(g, starts, j) for j, g in enumerate(keep)], chunksize=4)):
            if best is not None:
                found.append({"g6": g6, **best})
    found.sort(key=lambda r: -r["mu"])
    print(f"con algún estado estricto no sync: {len(found)}   [{time.time() - t0:.0f}s]")
    for r in found[:10]:
        print(f"   {r['g6']:12s} μ = {r['mu']:.5f}   λ = {r['lam']:.1e}")
    json.dump({"k": k, "n_graphs": len(graphs), "n_prefilter": len(keep), "found": found[:200]},
              open(os.path.join(ROOT, "data", f"explore_01_k{k}.json"), "w"), indent=1)
