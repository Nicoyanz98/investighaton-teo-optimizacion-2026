"""Tests de la familia girasol: verifican numéricamente los lemas y proposiciones de notes/familia_aurea.md."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aurea import (  # noqa: E402
    distancias_fibonacci,
    estado_aureo,
    fibonacci,
    mejor_circulante_fibonacci,
    multiplicar_conjunto,
    puntos_aureos,
    simetrizar,
    trampa_geometrica,
)
from src.busqueda import red_aurea  # noqa: E402
from src.grafos import grafo_circulante, grafo_circulante_conjunto, grafo_geometrico_circular  # noqa: E402
from src.kuramoto import (  # noqa: E402
    autovalores_twisted,
    autovalores_twisted_conjunto,
    hessiano,
    twisted_state,
    velocidad,
)


def test_lema_1_formula_contra_hessiano():
    n = 17
    S = simetrizar([2, 5, 7], n)
    W = grafo_circulante_conjunto(n, S)
    for q in [1, 3, 6]:
        theta = twisted_state(n, q)
        assert np.abs(velocidad(theta, W)).max() < 1e-12
        numericos = np.sort(np.linalg.eigvalsh(hessiano(theta, W)))
        formula = np.sort(autovalores_twisted_conjunto(n, S, q))
        assert numericos == pytest.approx(formula, abs=1e-10)


def test_lema_1_recupera_anillos():
    n, k = 20, 6
    S = simetrizar(range(1, k + 1), n)
    for q in [1, 2]:
        assert autovalores_twisted_conjunto(n, S, q) == pytest.approx(autovalores_twisted(n, k, q))


def test_lema_3_congruencia_fibonacci():
    for k in range(4, 41):
        n = fibonacci(k)
        for j in range(2, k):
            izquierda = (fibonacci(k - 1) * fibonacci(j)) % n
            derecha = ((-1) ** (j - 1) * fibonacci(k - j)) % n
            assert izquierda == derecha


def test_proposicion_a_mismo_espectro():
    k = 10
    n = fibonacci(k)
    q = fibonacci(k - 1)
    S = simetrizar(distancias_fibonacci(k)[1:4], n)
    S_prima = multiplicar_conjunto(S, q, n)
    aureo = np.sort(autovalores_twisted_conjunto(n, S, q))
    uno = np.sort(autovalores_twisted_conjunto(n, S_prima, 1))
    assert aureo == pytest.approx(uno, abs=1e-10)


def test_proposicion_b_geometrico_es_anillo():
    for k in [8, 10, 12]:
        n = fibonacci(k)
        q = fibonacci(k - 1)
        ranura = (np.arange(n) * q) % n
        for rho in [3, n // 4]:
            W = grafo_geometrico_circular(puntos_aureos(n), (rho + 0.5) / n)
            anillo = grafo_circulante(n, rho)
            assert np.array_equal(W, anillo[np.ix_(ranura, ranura)])
        desvio = np.angle(np.exp(1j * (estado_aureo(n) - 2 * np.pi * ranura / n)))
        assert np.abs(desvio).max() < 2 * np.pi * 0.45 / n


def test_variante_1_por_debajo_del_lp():
    resultado = mejor_circulante_fibonacci(8)
    assert resultado["mu"] == pytest.approx(9 / 21)
    assert resultado["mu"] < 0.6839


def test_trampa_geometrica_n100():
    resultado = trampa_geometrica(100, 0.3)
    assert resultado["trampa"]
    assert resultado["winding"] != 0
    assert resultado["velocidad"] < 1e-10


def test_red_aurea_pesos_suman_uno():
    for pesos in ["iguales", "arcos", "geometricos"]:
        x, phi = red_aurea(13, pesos)
        assert x.sum() == pytest.approx(1)
        assert len(phi) == 13
