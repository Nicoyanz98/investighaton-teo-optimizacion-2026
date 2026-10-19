"""Utilidades comunes de los scripts: paralelismo, guardado de resultados en data/ y avisos."""
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

CARPETA_DATOS = Path(__file__).resolve().parents[1] / "data"


def silenciar_avisos():
    """Oculta los avisos de precisión de cvxpy y scipy (los resultados se verifican aparte)."""
    warnings.filterwarnings("ignore")


def correr_en_paralelo(funcion, trabajos, procesos=6):
    """Aplica funcion a cada trabajo en varios procesos; va devolviendo resultados a medida que terminan."""
    with Pool(min(procesos, len(trabajos)), initializer=silenciar_avisos) as pool:
        for resultado in pool.imap_unordered(funcion, trabajos):
            yield resultado


def guardar_json(nombre, datos):
    """Guarda datos en data/<nombre> y devuelve la ruta."""
    CARPETA_DATOS.mkdir(exist_ok=True)
    ruta = CARPETA_DATOS / nombre
    with open(ruta, "w") as archivo:
        json.dump(datos, archivo, indent=1)
    return ruta


def cargar_json(nombre):
    """Lee data/<nombre>."""
    with open(CARPETA_DATOS / nombre) as archivo:
        return json.load(archivo)
