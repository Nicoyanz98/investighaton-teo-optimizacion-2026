"""Camino B: plantillas S7 con clusters partidos en sub-piezas de la misma fase, para bajar n.

Partir un cluster (por ejemplo W, k = 5, en piezas de k = 2 y 3) no cambia la plantilla: cada pieza
ve lo mismo que el cluster original y las piezas quedan unidas completas, así que el modo relativo
entre ellas es rígido. Pero duplica los grados libres hacia los vecinos: el redondeo por reticulado
(src/enteros.py) tiene más enteros para anular las mismas 3 ecuaciones blandas, y alcanza con n menor.

Para cada (h, partición, n) prueba varios g alrededor de n/K (el éxito del reticulado no es monótono
en n), corre el redondeo por reticulado y, si el equilibrio exacto es estricto, el certificado
riguroso del blow-up (`blowup.certificar_blowup`).

Salida: data/s7_subpiezas.json (una fila por intento; los certificados llevan la descripción compacta).

Ejemplos:
  .venv/bin/python scripts/s7_subpiezas.py --procesos 8                                     # local
  OMP_NUM_THREADS=1 python scripts/s7_subpiezas.py --procesos 12 --barrido 12               # server
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desarrollo.src.blowup import certificar_blowup, descripcion_compacta  # noqa: E402
from desarrollo.src.enteros import fases_s7, mu_entero, pulir, redondeo_reticulado, subdividir, tamanos_exactos  # noqa: E402
from desarrollo.src.experimentos import cargar_json, correr_en_paralelo, guardar_json  # noqa: E402

PARTICIONES = {
    "nada": [[3]] * 6 + [[5]],
    "W23": [[3]] * 6 + [[2, 3]],
    "W122": [[3]] * 6 + [[1, 2, 2]],
    "todo": [[1, 2]] * 6 + [[2, 3]],
}


def plantilla_partida(h, particion):
    """Plantilla S7 (h, ε = 1e-3, tamaños (3, 5)) de data/s7_curva.json, pulida y partida."""
    fila = [f for f in cargar_json("s7_curva.json") if f["h"] == h and f["margen"] == 1e-3 and f["k"] == [3, 5]][0]
    k = [3] * 6 + [5]
    x, phi, P = pulir(tamanos_exactos(k), fases_s7(h, np.array(fila["s"])), np.array(fila["P"]))
    return subdividir(x, phi, P, k, PARTICIONES[particion])


def un_intento(trabajo):
    """Reticulado + certificado para un (h, partición, g)."""
    h, particion, n_objetivo, g = trabajo
    inicio = time.time()
    x, phi, P, k = plantilla_partida(h, particion)
    d, evaluacion = redondeo_reticulado(x, P, phi, h, k, g)
    fila = {"h": h, "particion": particion, "n_objetivo": n_objetivo, "n": sum(k) * g, "g": g, "k": k,
            "reticulado_ok": bool(evaluacion["ok"]), "menor_autovalor": evaluacion["menor autovalor"],
            "mu": float(mu_entero(k, g, d)), "certificado": None}
    if evaluacion["ok"]:
        certificado = certificar_blowup(k, g, d, evaluacion["phi"])
        fila["certificado"] = certificado
        if certificado["ok"]:
            fila["grafo"] = descripcion_compacta(k, g, d, certificado["fases"])
    fila["segundos"] = time.time() - inicio
    return fila


def main():
    """Barre (h, partición, n, g) en paralelo y guarda todos los intentos."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hs", default="0.03,0.02")
    parser.add_argument("--particiones", default="W23,W122")
    parser.add_argument("--ns", default="3e5,5e5,7e5,1e6,1.5e6,2e6,3e6")
    parser.add_argument("--barrido", type=int, default=8, help="cuántos g consecutivos probar por n")
    parser.add_argument("--procesos", type=int, default=8)
    args = parser.parse_args()

    trabajos = []
    for h in [float(v) for v in args.hs.split(",")]:
        for particion in args.particiones.split(","):
            K = sum(sum(piezas) for piezas in PARTICIONES[particion])
            for n in [float(v) for v in args.ns.split(",")]:
                for desplazamiento in range(args.barrido):
                    trabajos.append((h, particion, n, int(round(n / K)) + desplazamiento))

    resultados = []
    for fila in correr_en_paralelo(un_intento, trabajos, args.procesos):
        certificado = fila["certificado"]
        estado = "—" if certificado is None else ("CERTIFICADO" if certificado["ok"] else "falla el certificado")
        print(f"h = {fila['h']}  {fila['particion']:5s} n = {fila['n']:9d}: reticulado {fila['reticulado_ok']!s:5s}"
              f" λ = {fila['menor_autovalor']: .2e}  μ = {fila['mu']:.6f}  {estado}", flush=True)
        resultados.append(fila)
    resultados.sort(key=lambda fila: (fila["h"], fila["particion"], fila["n"]))
    print("guardado en", guardar_json("s7_subpiezas.json", resultados))
    for h in sorted(set(fila["h"] for fila in resultados)):
        certificados = [f for f in resultados if f["h"] == h and f["certificado"] and f["certificado"]["ok"]]
        if certificados:
            mejor = min(certificados, key=lambda f: f["n"])
            print(f"h = {h}: menor n certificado = {mejor['n']} ({mejor['particion']}), μ = {mejor['certificado']['mu_exacto']}")


if __name__ == "__main__":
    main()
