"""Tests de regresión: fijan los números clave de la reproducción (C4, twisted, Canale, S7, certificado)."""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desarrollo.src import s7
from desarrollo.src.busqueda import FamiliaParticiones, base_c4, base_canale, dos_redes, semilla, una_red
from desarrollo.src.certificado import certificar, fases_newton, racionalizar
from desarrollo.src.grafos import blowup_cartesiano, blowup_lexicografico, conectividad, grafo_circulante
from desarrollo.src.kuramoto import clasificar, twisted_state
from desarrollo.src.plantillas import analizar, plantilla_c4, plantilla_canale, realizar
from desarrollo.src.twisted_lp import mejor_twisted


def test_blowup_lexicografico_preserva_mu_y_cartesiano_la_baja():
    W = grafo_circulante(20, 6)
    theta = twisted_state(20, 1)
    lexicografico = blowup_lexicografico(W, 3)
    assert conectividad(lexicografico) == pytest.approx(conectividad(W))
    assert conectividad(blowup_cartesiano(W, 3)) == pytest.approx(0.25)
    assert clasificar(np.tile(theta, 3), lexicografico)[0] == "mínimo estricto"


def test_c4_es_degenerado():
    T = plantilla_c4()
    assert T.mu() == pytest.approx(0.75)
    assert np.abs(T.autovalores_clusters()).max() < 1e-12


def test_canale_estable_solo_debajo_del_borde():
    assert plantilla_canale(d=0.49).es_estricta()
    assert not plantilla_canale(d=0.5).es_estricta()
    assert not plantilla_canale(d=0.51).es_estricta()
    assert plantilla_canale(d=0.5).mu() == pytest.approx(11 / 16)


def test_canale_k_spinning_da_minimo_estricto():
    W, theta = realizar(plantilla_canale(d=3 / 8), 8, rng=0, tipo="intervalo")
    assert clasificar(theta, W)[0] == "mínimo estricto"
    assert W.sum(axis=1).min() == 42


def test_twisted_lp_m19():
    assert mejor_twisted(19) == pytest.approx(0.683876, abs=1e-6)


def test_s7_esquina_y_primer_orden():
    mp.mp.dps = 40
    q, w, mu0 = s7.esquina()
    assert float(q) == pytest.approx(0.65056916694479880261, abs=1e-15)
    assert float(w) == pytest.approx(1.39827813402046117964, abs=1e-15)
    assert float(mu0) == pytest.approx(0.69290770636958636365, abs=1e-15)
    residuos = []
    for h in ["1e-3", "1e-4"]:
        S = s7.construir("1e-3", h, corregir=False)
        residuos.append(max(abs(v) for v in S["F_primer_orden"]))
    assert float(residuos[0] / residuos[1]) == pytest.approx(100, rel=0.05)


def test_s7_finito_es_estricto():
    S = s7.construir("1e-3", "1e-4")
    r = analizar(S["x"], S["phi"], S["P"], mp)
    assert r["max|F|"] < 1e-50
    assert min(r["autovalores L"]) > 0
    assert float(r["mu"]) == pytest.approx(0.6926394919, abs=1e-9)


def _certificar_s7(t, h, log_d):
    D = 2 * 10**log_d
    dps = 2 * log_d + 30
    S = s7.construir(t, h, dps=dps)
    xq, Pq = racionalizar(S["x"], S["P"], D)
    phi = fases_newton(xq, Pq, S["phi"], dps=dps + 20)
    return certificar(xq, Pq, phi, radio=mp.mpf(10) ** (-(log_d + 12)), dps=dps, verbose=False)


def test_certificado_proposicion_1():
    res = _certificar_s7("1e-5", "1e-6", 42)
    assert res["ok"]
    assert float(res["mu"]) == pytest.approx(0.6929050246493601, abs=1e-15)


def test_certificado_control_negativo():
    res = _certificar_s7("-0.01", "1e-4", 30)
    assert res["krawczyk"]
    assert not res["cholesky"]
    assert not res["ok"]


def test_familia_canale_sin_partir_da_canale():
    familia = FamiliaParticiones(base_canale(), [1] * 8, h=0.01)
    z = np.log(np.full(8, 1 / 8))
    assert familia.mu(z, margen=0.0) == pytest.approx(11 / 16, abs=1e-4)


def test_familia_c4_arranque_s7():
    from desarrollo.src.busqueda import arranque_estructurado

    familia = FamiliaParticiones(base_c4(), [2, 2, 2, 1], h=0.01)
    assert familia.mu(arranque_estructurado(familia)) == pytest.approx(0.6851, abs=5e-4)


def test_semillas_c4_y_canale():
    mu_c4, neutros_c4 = semilla(*una_red(4))
    assert mu_c4 == pytest.approx(0.75, abs=1e-5)
    assert neutros_c4 == 3
    mu_canale, _ = semilla(*dos_redes(4, 0.5, np.arctan(0.5)))
    assert mu_canale == pytest.approx(11 / 16, abs=1e-4)
