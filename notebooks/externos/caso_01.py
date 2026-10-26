"""Caso 0/1 (sección 4.3): blow-ups completos de todos los grafos conexos H con k vértices.

Completo: k = 5, 6, 7 con 20 arranques. Reducido: k = 5, 6 con 5 arranques.
Uso: python externos/caso_01.py [--reducido] [--procesos N]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biblioteca_s7 import *  # noqa: E402,F403


def un_k(k, arranques, piso, procesos):
    """Prefiltro LP y optimización en paralelo para los grafos de k vértices."""
    inicio = time.time()
    cantidad = 0
    candidatos = []
    for codigo, A in grafos_conexos(k):
        cantidad += 1
        M = A + np.eye(k)
        if M.min() == 1:  # H completo: sincroniza
            continue
        if mu_sin_estabilidad(M)[0] > piso + 1e-9:
            candidatos.append(codigo)
    print(f"k = {k}: {cantidad} grafos conexos, {len(candidatos)} pasan el prefiltro LP > {piso}", flush=True)
    trabajos = []
    for i, codigo in enumerate(candidatos):
        trabajos.append((codigo, arranques, i))
    encontrados = []
    for codigo, mejor in correr_en_paralelo(optimizar_trabajo, trabajos, procesos):
        if mejor is not None:
            encontrados.append({"g6": codigo, **mejor})
    encontrados.sort(key=lambda r: -r["mu"])
    print(f"   con algún estado estricto no sync: {len(encontrados)}   [{time.time() - inicio:.0f} s]", flush=True)
    return {"k": k, "arranques": arranques, "cantidad_grafos": cantidad, "cantidad_prefiltro": len(candidatos),
            "encontrados": encontrados[:200]}


def main():
    """Recorre los k pedidos y guarda resultados/blowups_01[_reducido].json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reducido", action="store_true")
    parser.add_argument("--piso", type=float, default=0.6, help="prefiltro: μ del LP mayor que esto")
    parser.add_argument("--procesos", type=int, default=8)
    args = parser.parse_args()
    ks = [5, 6] if args.reducido else [5, 6, 7]
    arranques = 5 if args.reducido else 20
    resultados = []
    for k in ks:
        resultados.append(un_k(k, arranques, args.piso, args.procesos))
    nombre = "blowups_01_reducido.json" if args.reducido else "blowups_01.json"
    print("guardado en", guardar_json(nombre, resultados))


if __name__ == "__main__":
    main()
