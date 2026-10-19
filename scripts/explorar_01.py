"""Caso 0/1 (la idea original del "stackeo"): blow-ups completos de un grafo base H.

Cada vértice de H es un clique y cada arista de H un bipartito completo: P = A_H + I, sin densidades
intermedias. Para cada H conexo con k vértices (nauty) se busca
    max μ = min_a d_a   sobre tamaños x y fases φ (φ_0 = 0)
    s.t. F = 0,  c_a ≥ margen_c,  λ_min(L en ⊥1) ≥ margen,  fases no todas iguales
con SLSQP desde varios arranques al azar. Prefiltro: el LP max_x min_a x(N[a]) tiene que superar
--piso (ignorar la estabilidad solo puede dar más).

Ejemplo:  .venv/bin/python scripts/explorar_01.py --k 6 --arranques 20 --piso 0.6
Salida:   data/blowups_01_k<k>.json (los mejores H, con x y φ).
"""
import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linprog, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimentos import correr_en_paralelo, guardar_json  # noqa: E402
from src.grafos import grafos_nauty  # noqa: E402
from src.plantillas import base_perpendicular, fuerzas, matriz_clusters, splitting  # noqa: E402


def mu_sin_estabilidad(M):
    """LP: max_x min_a (M x)_a con x en el símplex. Devuelve (valor, x óptimo)."""
    k = len(M)
    objetivo = np.zeros(k + 1)
    objetivo[-1] = -1
    desigualdades = np.hstack([-M, np.ones((k, 1))])
    suma_uno = [list(np.ones(k)) + [0]]
    resultado = linprog(objetivo, A_ub=desigualdades, b_ub=np.zeros(k), A_eq=suma_uno, b_eq=[1],
                        bounds=[(0, 1)] * (k + 1), method="highs")
    return -resultado.fun, resultado.x[:k]


class Restricciones:
    """Desempaqueta v = (logits de x, φ_1..φ_{k−1}, t) y evalúa las restricciones de una plantilla 0/1."""

    def __init__(self, M, margen, margen_c):
        """Guarda M = A + I, la base de ⊥1 y los márgenes."""
        self.M = M
        self.k = len(M)
        self.Q = base_perpendicular(self.k)
        self.margen = margen
        self.margen_c = margen_c

    def desempaquetar(self, v):
        """v → (x en el símplex, φ con φ_0 = 0)."""
        k = self.k
        x = np.exp(v[:k] - v[:k].max())
        x = x / x.sum()
        phi = np.concatenate([[0.0], v[k: 2 * k - 1]])
        return x, phi

    def grado_menos_t(self, v):
        """d_a − t (tiene que ser ≥ 0)."""
        x, _ = self.desempaquetar(v)
        return self.M @ x - v[-1]

    def equilibrio(self, v):
        """F_a para a ≥ 1 (tiene que ser 0)."""
        x, phi = self.desempaquetar(v)
        return np.array(fuerzas(x, phi, self.M))[1:]

    def splitting_menos_margen(self, v):
        """c_a − margen_c (tiene que ser ≥ 0)."""
        x, phi = self.desempaquetar(v)
        return np.array(splitting(x, phi, self.M)) - self.margen_c

    def autovalor_min(self, v):
        """Menor autovalor de L en ⊥1."""
        x, phi = self.desempaquetar(v)
        L = np.array(matriz_clusters(x, phi, self.M))
        return np.linalg.eigvalsh(self.Q.T @ L @ self.Q).min()

    def para_slsqp(self):
        """Las restricciones en el formato de scipy.optimize.minimize."""
        return [
            {"type": "ineq", "fun": self.grado_menos_t},
            {"type": "eq", "fun": self.equilibrio},
            {"type": "ineq", "fun": self.splitting_menos_margen},
            {"type": "ineq", "fun": lambda v: np.array([self.autovalor_min(v) - self.margen])},
        ]


def optimizar_grafo(M, arranques, rng, margen=1e-6, margen_c=1e-3):
    """Mejor estado estricto no sync del blow-up completo de M = A + I, o None si no se encontró."""
    k = len(M)
    restricciones = Restricciones(M, margen, margen_c)
    _, x_lp = mu_sin_estabilidad(M)
    mejor = None
    for _ in range(arranques):
        x0 = 0.5 * x_lp + 0.5 / k + 0.02 * rng.random(k)
        v0 = np.concatenate([np.log(x0), rng.uniform(-np.pi, np.pi, k - 1), [0.5]])
        try:
            resultado = minimize(lambda v: -v[-1], v0, constraints=restricciones.para_slsqp(),
                                 method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
        except Exception:
            continue
        v = resultado.x
        x, phi = restricciones.desempaquetar(v)
        autovalor = restricciones.autovalor_min(v)
        agrupamiento = np.abs(np.exp(1j * phi).mean())
        valido = (np.abs(restricciones.equilibrio(v)).max() < 1e-8
                  and restricciones.splitting_menos_margen(v).min() > -margen_c / 2
                  and autovalor > margen / 2
                  and agrupamiento < 1 - 1e-6)
        if not valido:
            continue
        mu = (M @ x).min()
        if mejor is None or mu > mejor["mu"]:
            mejor = {"mu": float(mu), "x": x.tolist(), "phi": phi.tolist(), "autovalor_min": float(autovalor)}
    return mejor


def optimizar_trabajo(trabajo):
    """Trabajo en paralelo: (graph6, arranques, semilla) → (graph6, mejor estado o None)."""
    codigo, arranques, semilla_rng = trabajo
    G = nx.from_graph6_bytes(codigo.encode())
    M = nx.to_numpy_array(G) + np.eye(G.number_of_nodes())
    return codigo, optimizar_grafo(M, arranques, np.random.default_rng(semilla_rng))


def main():
    """Prefiltra los grafos de nauty y optimiza los que pasan, en paralelo."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, required=True, help="vértices del grafo base")
    parser.add_argument("--arranques", type=int, default=20)
    parser.add_argument("--piso", type=float, default=11 / 16, help="prefiltro: μ del LP mayor que esto")
    parser.add_argument("--procesos", type=int, default=6)
    args = parser.parse_args()

    inicio = time.time()
    cantidad = 0
    candidatos = []
    for codigo, A in grafos_nauty(args.k):
        cantidad += 1
        M = A + np.eye(args.k)
        if M.min() == 1:  # H completo: sincroniza
            continue
        if mu_sin_estabilidad(M)[0] > args.piso + 1e-9:
            candidatos.append(codigo)
    print(f"k = {args.k}: {cantidad} grafos conexos, {len(candidatos)} pasan el prefiltro LP > {args.piso:.4f}", flush=True)

    trabajos = []
    for i, codigo in enumerate(candidatos):
        trabajos.append((codigo, args.arranques, i))
    encontrados = []
    for codigo, mejor in correr_en_paralelo(optimizar_trabajo, trabajos, args.procesos):
        if mejor is not None:
            encontrados.append({"g6": codigo, **mejor})
    encontrados.sort(key=lambda r: -r["mu"])

    print(f"con algún estado estricto no sync: {len(encontrados)}   [{time.time() - inicio:.0f}s]")
    for r in encontrados[:10]:
        print(f"   {r['g6']:12s} μ = {r['mu']:.5f}   λ = {r['autovalor_min']:.1e}")
    datos = {"k": args.k, "cantidad_grafos": cantidad, "cantidad_prefiltro": len(candidatos),
             "encontrados": encontrados[:200]}
    guardar_json(f"blowups_01_k{args.k}.json", datos)


if __name__ == "__main__":
    main()
