"""Barridos de la familia girasol (notes/familia_aurea.md).

  1. Circulantes de Fibonacci C_{F_k}(S): mayor μ con el estado áureo como trampa (fuerza bruta en S).
  2. Grafo geométrico áureo (x_v = v/φ mod 1, radio r): mayor μ con trampa para cada n, comparado
     con el mejor anillo C_{n,k} del mismo n.
  3. Plantillas con clusters en las fases áureas: μ del SDP con L ≻ 0, comparado con el LP de twisted.

Guarda data/aurea_fibonacci.json, data/aurea_geometrico.json y data/aurea_plantillas.json.

Ejemplo:  .venv/bin/python scripts/explorar_aurea.py --n-max 200 --m-max 40
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desarrollo.src.aurea import es_fibonacci, fibonacci, mejor_circulante_fibonacci, mejor_radio  # noqa: E402
from desarrollo.src.busqueda import red_aurea, sdp_interno  # noqa: E402
from desarrollo.src.experimentos import correr_en_paralelo, guardar_json, silenciar_avisos  # noqa: E402
from desarrollo.src.kuramoto import k_max  # noqa: E402
from desarrollo.src.twisted_lp import mejor_twisted  # noqa: E402

PESOS = ["iguales", "arcos", "geometricos"]


def variante_fibonacci(k_max_fibonacci):
    """Variante 1 para k = 5..k_max_fibonacci."""
    resultados = []
    for k in range(5, k_max_fibonacci + 1):
        fila = mejor_circulante_fibonacci(k)
        fila["twisted_lp"] = mejor_twisted(fibonacci(k))
        resultados.append(fila)
        print(f"  k = {k:2d}  n = {fila['n']:4d}  μ = {fila['mu']:.4f}  distancias = {fila['distancias']}"
              f"  (LP twisted del mismo n: {fila['twisted_lp']:.4f})", flush=True)
    return resultados


def geometrico_un_n(n):
    """Variante 2 para un n: mejor radio con trampa y el anillo de referencia."""
    mejor = mejor_radio(n)
    k = k_max(n)[0]
    return {
        "n": n,
        "fibonacci": es_fibonacci(n),
        "mu_aureo": mejor["mu"] if mejor else 0.0,
        "r": mejor["r"] if mejor else None,
        "velocidad": mejor["velocidad"] if mejor else None,
        "mu_anillo": (2 * k + 1) / n,
    }


def variante_geometrica(n_max, procesos):
    """Variante 2 para n = 8..n_max, en paralelo."""
    resultados = list(correr_en_paralelo(geometrico_un_n, list(range(8, n_max + 1)), procesos))
    resultados.sort(key=lambda fila: fila["n"])
    for fila in resultados:
        if fila["fibonacci"] or fila["n"] % 25 == 0:
            marca = "(Fib)" if fila["fibonacci"] else "     "
            print(f"  n = {fila['n']:3d} {marca}  μ_áureo = {fila['mu_aureo']:.4f}"
                  f"  μ_anillo = {fila['mu_anillo']:.4f}")
    return resultados


def variante_plantillas(m_max):
    """Variante 3 para m = 5..m_max y los tres tipos de pesos."""
    resultados = []
    for m in range(5, m_max + 1):
        fila = {"m": m, "fibonacci": es_fibonacci(m), "twisted_lp": mejor_twisted(m)}
        for pesos in PESOS:
            mu, _ = sdp_interno(*red_aurea(m, pesos), margen=1e-9, margen_c=1e-6)
            fila[pesos] = mu if mu is not None else 0.0
        resultados.append(fila)
        print(f"  m = {m:2d}  LP twisted = {fila['twisted_lp']:.4f}  " +
              "  ".join(f"{pesos} = {fila[pesos]:.4f}" for pesos in PESOS), flush=True)
    return resultados


def main():
    """Corre las tres variantes, imprime un resumen y guarda los JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k-max", type=int, default=14, help="último k de la variante 1 (n = F_k)")
    parser.add_argument("--n-max", type=int, default=200, help="último n de la variante 2")
    parser.add_argument("--m-max", type=int, default=40, help="último m de la variante 3")
    parser.add_argument("--procesos", type=int, default=6)
    args = parser.parse_args()
    silenciar_avisos()

    inicio = time.time()
    print("variante 1: circulantes de Fibonacci")
    fibonacci_resultados = variante_fibonacci(args.k_max)
    print("  guardado en", guardar_json("aurea_fibonacci.json", fibonacci_resultados))

    print("\nvariante 2: grafo geométrico áureo")
    geometrico = variante_geometrica(args.n_max, args.procesos)
    no_fibonacci = [fila for fila in geometrico if not fila["fibonacci"]]
    mejores = [fila["n"] for fila in no_fibonacci if fila["mu_aureo"] > fila["mu_anillo"] + 1e-12]
    print(f"  n no Fibonacci donde el áureo supera al anillo: {mejores}")
    print("  guardado en", guardar_json("aurea_geometrico.json", geometrico))

    print("\nvariante 3: plantillas áureas")
    plantillas = variante_plantillas(args.m_max)
    print("  guardado en", guardar_json("aurea_plantillas.json", plantillas))
    print(f"\n{time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()
