"""Tests del espectro por bloques de Fourier (src/espectro.py) contra el Hessiano denso."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desarrollo.src.blowup import (  # noqa: E402
    adyacencia_densa, descripcion_compacta, fases_vertices, hessiano_denso, hessiano_fourier,
    transformada_intervalo,
)
from desarrollo.src.espectro import bloques_fourier, espectro_fourier, producto_fourier, transformada_intervalos  # noqa: E402
from desarrollo.src.validacion import cociente_formula  # noqa: E402
from desarrollo.tests.test_blowup import grados_al_azar  # noqa: E402

K = [2, 3, 1]
G = 7


def _caso(semilla):
    """Descripción compacta al azar, sus grados y su Hessiano denso."""
    rng = np.random.default_rng(semilla)
    d = grados_al_azar(K, G, rng)
    fases = rng.uniform(-1, 1, len(K))
    descripcion = descripcion_compacta(K, G, d, list(fases))
    H = hessiano_denso(adyacencia_densa(K, G, d), fases_vertices(K, G, fases))
    return descripcion, d, fases, H


def test_transformada_cerrada_coincide_con_la_suma():
    tabla = np.array([[0, 3, 7], [1, 5, 2]])
    frecuencias = np.arange(G)
    N = transformada_intervalos(tabla, G, frecuencias)
    for m in frecuencias:
        for i in range(2):
            for j in range(3):
                assert abs(N[m, i, j] - transformada_intervalo(tabla[i, j], G, m)) < 1e-12


def test_bloques_coinciden_con_hessiano_fourier():
    descripcion, d, fases, H = _caso(0)
    bloques = bloques_fourier(descripcion, np.arange(G))
    for m in range(G):
        assert np.allclose(bloques[m], hessiano_fourier(K, G, d, fases, m), atol=1e-12)


def test_producto_fourier_coincide_con_el_denso():
    for semilla in range(3):
        descripcion, d, fases, H = _caso(semilla)
        x = np.random.default_rng(10 + semilla).standard_normal(H.shape[0])
        assert np.allclose(producto_fourier(descripcion, x, lote=3), H @ x, atol=1e-10)


def test_espectro_es_el_del_hessiano_denso():
    for semilla in range(3):
        descripcion, d, fases, H = _caso(semilla)
        denso = np.sort(np.linalg.eigvalsh(H))
        espectro = espectro_fourier(descripcion, lote=2)
        assert espectro["acople"] < 1e-10
        assert np.allclose(np.sort(espectro["cluster"]), np.linalg.eigvalsh(cociente_formula(descripcion)),
                           atol=1e-9)
        todos = list(espectro["cluster"]) + list(espectro["cero"])
        bloques = bloques_fourier(descripcion, np.arange(1, G))
        todos.extend(np.linalg.eigvalsh(bloques).ravel())
        assert np.allclose(np.sort(todos), denso, atol=1e-9)
        minimos = np.linalg.eigvalsh(bloques)[: G // 2, 0]
        assert np.allclose(espectro["minimos"], minimos, atol=1e-12)
        assert abs(espectro["maximo"] - denso[-1]) < 1e-9
