"""Construcción de grafos: familias básicas, blow-ups y enumeración con nauty."""
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np

GENG = Path(__file__).resolve().parents[1] / "tools" / "nauty" / "geng"


def grafo_completo(n):
    """K_n: todos con todos, sin self-loops."""
    return np.ones((n, n)) - np.eye(n)


def grafo_ciclo(n):
    """C_n: cada v conectado con v − 1 y v + 1 (módulo n)."""
    return grafo_circulante(n, 1)


def grafo_circulante(n, k):
    """C_{n,k}: cada v conectado con v ± 1, …, v ± k (módulo n)."""
    W = np.zeros((n, n))
    v = np.arange(n)
    for j in range(1, k + 1):
        W[v, (v + j) % n] = 1
        W[(v + j) % n, v] = 1
    return W


def blowup_lexicografico(W, m):
    """G[K_m]: cada vértice pasa a ser un clique de m copias unido a todas las copias de sus vecinos."""
    unos = np.ones((m, m))
    identidad = np.eye(m)
    n = len(W)
    return np.kron(unos, W) + np.kron(unos - identidad, np.eye(n))


def blowup_cartesiano(W, m):
    """G □ K_m: m copias de G, y las copias de un mismo vértice forman un clique."""
    unos = np.ones((m, m))
    identidad = np.eye(m)
    n = len(W)
    return np.kron(identidad, W) + np.kron(unos - identidad, np.eye(n))


def conectividad(W):
    """μ = (δ + 1)/n, con δ el grado mínimo."""
    return (W.sum(axis=1).min() + 1) / len(W)


def grafos_nauty(k):
    """Genera (graph6, matriz de adyacencia) de todos los grafos conexos con k vértices."""
    if not GENG.exists():
        raise FileNotFoundError("Falta nauty: correr `bash tools/install_nauty.sh`")
    salida = subprocess.run([str(GENG), "-cq", str(k)], capture_output=True, text=True, check=True)
    for codigo in salida.stdout.split():
        G = nx.from_graph6_bytes(codigo.encode())
        yield codigo, nx.to_numpy_array(G)
