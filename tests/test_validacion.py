"""Tests de la validación implícita (src/validacion.py) contra el grafo denso de src/blowup.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blowup import adyacencia_densa, descripcion_compacta, fases_vertices, hessiano_denso  # noqa: E402
from src.enteros import paso_grado, tamanos_enteros  # noqa: E402
from src.validacion import (  # noqa: E402
    Arreglos, GrafoImplicito, cociente_formula, cociente_implicito, grados_medidos, lanczos,
)
from tests.test_blowup import _grafo_mas_chico, grados_al_azar  # noqa: E402

K = [2, 3, 1]
G = 7


def _caso(semilla):
    """Descripción compacta al azar, su grafo implícito (numpy) y su Hessiano denso."""
    rng = np.random.default_rng(semilla)
    d = grados_al_azar(K, G, rng)
    fases = rng.uniform(-1, 1, len(K))
    descripcion = descripcion_compacta(K, G, d, list(fases))
    grafo = GrafoImplicito(descripcion, Arreglos())
    H = hessiano_denso(adyacencia_densa(K, G, d), fases_vertices(K, G, fases))
    return descripcion, grafo, H, d


def test_productos_implicitos_coinciden_con_los_densos():
    for semilla in range(4):
        descripcion, grafo, H, d = _caso(semilla)
        W = adyacencia_densa(K, G, d)
        y = np.random.default_rng(10 + semilla).standard_normal(grafo.n)
        assert np.allclose(grafo.producto_adyacencia(y), W @ y, atol=1e-10)
        assert np.allclose(grafo.producto_hessiano(y), H @ y, atol=1e-10)


def test_grados_y_cociente():
    descripcion, grafo, H, d = _caso(5)
    minimos, maximos = grados_medidos(grafo)
    assert (minimos == np.array(d)).all() and (maximos == np.array(d)).all()
    assert np.allclose(cociente_implicito(grafo), cociente_formula(descripcion), atol=1e-9)


def test_lanczos_encuentra_el_menor_autovalor_del_complemento():
    descripcion, grafo, H, d = _caso(6)
    arreglos = grafo.arreglos
    resultado = lanczos(grafo.producto_hessiano, grafo.quitar_medias_cluster, arreglos.aleatorio(grafo.n, 0),
                        arreglos, iteraciones=400, cada=1, tolerancia=1e-9)
    base = np.zeros((grafo.n, len(K)))
    for a in range(len(K)):
        base[grafo.inicios[a]: grafo.inicios[a] + K[a] * G, a] = 1
    Q, _ = np.linalg.qr(np.hstack([base, np.eye(grafo.n)]))
    complemento = Q[:, len(K):]
    esperado = np.linalg.eigvalsh(complemento.T @ H @ complemento).min()
    assert abs(resultado["menor"] - esperado) < 1e-6 * max(1, abs(esperado))


def _complemento(base):
    """Base ortonormal del complemento de las columnas de `base`."""
    n, m = base.shape
    Q, _ = np.linalg.qr(np.hstack([base, np.eye(n)]))
    return Q[:, m:]


def test_lanczos_con_densidades_del_grafo_certificado():
    """Mismas densidades y fases que el grafo certificado pero g = 60: espectro muy degenerado."""
    grande = _grafo_mas_chico()
    k = grande["k"]
    g = 60
    n_a = tamanos_enteros(k, g)
    N_a = tamanos_enteros(k, grande["g"])
    d = [[0] * len(k) for _ in k]
    for a in range(len(k)):
        d[a][a] = n_a[a] - 1
        for b in range(a + 1, len(k)):
            paso = paso_grado(k, a, b)
            d[a][b] = int(round(grande["grados"][a][b] / N_a[b] * n_a[b] / paso)) * paso
            d[b][a] = d[a][b] * n_a[a] // n_a[b]
    fases = [float(p) for p in grande["fases"]]
    grafo = GrafoImplicito(descripcion_compacta(k, g, d, fases), Arreglos())
    H = hessiano_denso(adyacencia_densa(k, g, d), fases_vertices(k, g, fases))
    base = np.zeros((grafo.n, len(k)))
    for a in range(len(k)):
        base[grafo.inicios[a]: grafo.inicios[a] + k[a] * g, a] = 1
    casos = [(grafo.quitar_medias_cluster, _complemento(base)),
             (grafo.quitar_media, _complemento(np.ones((grafo.n, 1))))]
    for proyectar, C in casos:
        esperado = np.linalg.eigvalsh(C.T @ H @ C).min()
        resultado = lanczos(grafo.producto_hessiano, proyectar, grafo.arreglos.aleatorio(grafo.n, 0),
                            grafo.arreglos, iteraciones=2000, cada=5, tolerancia=1e-9)
        assert abs(resultado["menor"] - esperado) < 1e-6 * max(1, abs(esperado))
