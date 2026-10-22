"""Tests del paso plantilla → datos enteros (src/enteros.py) y del reticulado (src/reticulado.py)."""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.enteros import (  # noqa: E402
    evaluar_enteros, fases_s7, grados_ingenuos, pulir, subdividir, tamanos_enteros, tamanos_exactos,
)
from src.experimentos import cargar_json  # noqa: E402
from src.plantillas import fuerzas  # noqa: E402
from src.reticulado import enteros_cercanos, lll  # noqa: E402


def test_lll_conserva_el_reticulado():
    rng = np.random.default_rng(1)
    B = rng.integers(-20, 20, size=(6, 6)).astype(float)
    reducida = lll(B)
    assert abs(abs(np.linalg.det(reducida)) - abs(np.linalg.det(B))) < 1e-6 * abs(np.linalg.det(B))
    cambio = np.linalg.solve(B.T, reducida.T)
    assert np.allclose(cambio, np.round(cambio), atol=1e-8)


def test_enteros_cercanos_anula_formas():
    rng = np.random.default_rng(0)
    formas = rng.standard_normal((2, 13))
    objetivo = rng.uniform(-0.5, 0.5, 13)
    u = enteros_cercanos(formas, objetivo, 10)
    ingenuo = np.abs(formas @ (np.round(objetivo) - objetivo)).max()
    assert np.abs(formas @ (u - objetivo)).max() < 1e-3 * ingenuo


def _plantilla(fila):
    k = [fila["k"][0]] * 6 + [fila["k"][1]]
    phi = fases_s7(fila["h"], np.array(fila["s"]))
    return k, pulir(tamanos_exactos(k), phi, np.array(fila["P"]))


def test_pulir_da_equilibrio_exacto():
    fila = cargar_json("s7_curva.json")[0]
    _, (x, phi, P) = _plantilla(fila)
    with mp.workdps(60):
        assert max(abs(f) for f in fuerzas(x, phi, P, mp)) < mp.mpf(10) ** -40


def test_grados_ingenuos_son_consistentes():
    fila = cargar_json("s7_curva.json")[0]
    k, (x, phi, P) = _plantilla(fila)
    g = 1000
    d = grados_ingenuos(x, P, k, g)
    n_a = tamanos_enteros(k, g)
    for a in range(7):
        assert d[a][a] == n_a[a] - 1
        for b in range(7):
            assert n_a[a] * d[a][b] == n_a[b] * d[b][a]


def test_grado_entero_guardado_es_estricto():
    exitos = [r for r in cargar_json("s7_enteros.json") if r["reticulado_ok"]]
    if not exitos:
        pytest.skip("no hay soluciones en data/s7_enteros.json")
    r = min(exitos, key=lambda r: r["n"])
    k = [r["k"][0]] * 6 + [r["k"][1]]
    fila = [f for f in cargar_json("s7_curva.json") if f["h"] == r["h"] and f["margen"] == r["margen"]
            and f["k"] == r["k"]][0]
    phi_plantilla = fases_s7(r["h"], np.array(fila["s"]))
    evaluacion = evaluar_enteros(k, r["g"], r["grados"], r["fases"], r["h"], phi_referencia=phi_plantilla)
    assert evaluacion["ok"]
    assert r["mu"] > 11 / 16


def test_subdividir_mantiene_el_equilibrio():
    """Partir W en piezas (2, 3): mismas fases, densidades copiadas, piezas completas entre sí, F = 0."""
    fila = [f for f in cargar_json("s7_curva.json") if f["h"] == 0.03 and f["margen"] == 1e-3 and f["k"] == [3, 5]][0]
    k = [3] * 6 + [5]
    x, phi, P = pulir(tamanos_exactos(k), fases_s7(0.03, np.array(fila["s"])), np.array(fila["P"]))
    x2, phi2, P2, k2 = subdividir(x, phi, P, k, [[3]] * 6 + [[2, 3]])
    assert k2 == [3] * 6 + [2, 3]
    assert abs(sum(x2) - 1) < mp.mpf(10) ** -40
    assert phi2[6] == phi2[7] == phi[6]
    assert P2[6][7] == P2[7][6] == 1
    for b in range(6):
        assert P2[6][b] == P2[7][b] == P[6][b]
    with mp.workdps(60):
        assert max(abs(f) for f in fuerzas(x2, phi2, P2, mp)) < mp.mpf(10) ** -40
