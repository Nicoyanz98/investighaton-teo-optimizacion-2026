"""Tests del blow-up circulante (src/blowup.py): grados exactos, espectro por Fourier y cotas."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blowup import (  # noqa: E402
    adyacencia_densa, certificar_blowup, desplazamientos_cluster, fases_vertices, hessiano_denso,
    hessiano_fourier, margenes_gershgorin, tabla_transporte, tablas_bloques,
)
from src.enteros import paso_grado, tamanos_enteros  # noqa: E402
from src.experimentos import cargar_json  # noqa: E402

K_CHICO = [1, 2, 3]
G_CHICO = 5


def grados_al_azar(k, g, rng):
    """Grados d_ab válidos (n_a d_ab = n_b d_ba, d_aa = n_a − 1) elegidos al azar."""
    m = len(k)
    n_a = tamanos_enteros(k, g)
    d = [[0] * m for _ in range(m)]
    for a in range(m):
        d[a][a] = n_a[a] - 1
        for b in range(a + 1, m):
            paso = paso_grado(k, a, b)
            d[a][b] = int(rng.integers(0, n_a[b] // paso + 1)) * paso
            d[b][a] = d[a][b] * n_a[a] // n_a[b]
    return d


@pytest.mark.parametrize("k_a,k_b,g", [(3, 5, 7), (5, 3, 4), (2, 2, 3), (1, 4, 6)])
def test_tabla_transporte_tiene_filas_y_columnas_correctas(k_a, k_b, g):
    paso = k_b // np.gcd(k_a, k_b)
    for d_ab in range(0, k_b * g + 1, paso):
        t = np.array(tabla_transporte(d_ab, k_a, k_b, g))
        assert (t.sum(axis=1) == d_ab).all()
        assert (t.sum(axis=0) == k_a * d_ab // k_b).all()
        assert t.min() >= 0 and t.max() <= g


def test_blowup_denso_tiene_los_grados_pedidos():
    rng = np.random.default_rng(0)
    d = grados_al_azar(K_CHICO, G_CHICO, rng)
    W = adyacencia_densa(K_CHICO, G_CHICO, d)
    assert (W == W.T).all() and (np.diag(W) == 0).all()
    n_a = tamanos_enteros(K_CHICO, G_CHICO)
    inicios = desplazamientos_cluster(K_CHICO, G_CHICO)
    for a in range(len(K_CHICO)):
        filas = W[inicios[a]: inicios[a] + n_a[a]]
        for b in range(len(K_CHICO)):
            vecinos = filas[:, inicios[b]: inicios[b] + n_a[b]].sum(axis=1)
            assert (vecinos == d[a][b]).all()


def test_espectro_denso_es_la_union_de_los_bloques_de_fourier():
    rng = np.random.default_rng(1)
    d = grados_al_azar(K_CHICO, G_CHICO, rng)
    fases = rng.uniform(0, 2 * np.pi, len(K_CHICO))
    H = hessiano_denso(adyacencia_densa(K_CHICO, G_CHICO, d), fases_vertices(K_CHICO, G_CHICO, fases))
    denso = np.sort(np.linalg.eigvalsh(H))
    por_bloques = []
    for m in range(G_CHICO):
        por_bloques.extend(np.linalg.eigvalsh(hessiano_fourier(K_CHICO, G_CHICO, d, fases, m)))
    assert np.allclose(denso, np.sort(por_bloques), atol=1e-9)


def test_margenes_acotan_los_modos_que_no_son_de_cluster():
    rng = np.random.default_rng(2)
    k = [2, 3, 3]
    g = 6
    for _ in range(5):
        d = grados_al_azar(k, g, rng)
        fases = rng.uniform(-0.3, 0.3, len(k))
        margenes = margenes_gershgorin(k, g, d, fases)
        for m in range(1, g):
            menor = np.linalg.eigvalsh(hessiano_fourier(k, g, d, fases, m)).min()
            assert menor >= min(margenes["fourier"]) - 1e-9
        H0 = hessiano_fourier(k, g, d, fases, 0).real
        base_cluster = np.zeros((sum(k), len(k)))
        inicio = 0
        for a, ka in enumerate(k):
            base_cluster[inicio: inicio + ka, a] = 1 / np.sqrt(ka)
            inicio += ka
        Q, _ = np.linalg.qr(np.hstack([base_cluster, rng.standard_normal((sum(k), sum(k) - len(k)))]))
        complemento = Q[:, len(k):]
        menor = np.linalg.eigvalsh(complemento.T @ H0 @ complemento).min()
        assert menor >= min(margenes["cero"]) - 1e-9


def _grafo_mas_chico():
    """Grafo certificado más chico de data/grafos_s7.json (o skip si no hay)."""
    grafos = [f for f in cargar_json("grafos_s7.json") if f["redondeo"] == "reticulado" and f["certificado"]["ok"]]
    if not grafos:
        pytest.skip("no hay grafos certificados en data/grafos_s7.json")
    return min(grafos, key=lambda f: f["grafo"]["n"])["grafo"]


def test_tablas_guardadas_dan_los_grados():
    grafo = _grafo_mas_chico()
    k = grafo["k"]
    g = grafo["g"]
    for (a, b), t in tablas_bloques(k, g, grafo["grados"]).items():
        t = np.array(t)
        assert (t.sum(axis=1) == grafo["grados"][a][b]).all()
        assert (t.sum(axis=0) == grafo["grados"][b][a]).all()
        assert t.min() >= 0 and t.max() <= g
        assert grafo["tablas"][f"{a},{b}"] == t.tolist()


def test_grafo_guardado_se_recertifica():
    grafo = _grafo_mas_chico()
    resultado = certificar_blowup(grafo["k"], grafo["g"], grafo["grados"], grafo["fases"])
    assert resultado["ok"]
    assert resultado["mu"] > 11 / 16
