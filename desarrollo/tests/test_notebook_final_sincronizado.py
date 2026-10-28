"""exploracion_s7_final.ipynb es autocontenido: sus funciones, clases y constantes son copias de src/ y scripts/.

Este test avisa si una copia del notebook y su original dejaron de ser idénticas. Las pocas definiciones
que el notebook adapta a propósito están en ADAPTADAS, con el motivo.
"""
import ast
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORIGENES = [
    "src/kuramoto.py", "src/grafos.py", "src/plantillas.py", "src/twisted_lp.py", "src/s7.py",
    "src/certificado.py", "src/busqueda.py", "src/reticulado.py", "src/enteros.py", "src/blowup.py",
    "src/validacion.py", "src/espectro.py", "src/experimentos.py",
    "scripts/explorar_particiones.py", "scripts/explorar_01.py", "scripts/barrer_semillas.py",
    "scripts/curva_s7.py", "scripts/s7_entero.py", "scripts/certificar_grafo_s7.py",
    "scripts/espectro_fourier.py", "scripts/visualizar_grafo.py",
]
ADAPTADAS = {
    "fases_s7": "escrita directamente para S7, sin FamiliaParticiones",
    "prohibidos_s7": "escrita directamente para S7, sin FamiliaParticiones",
    "guardar_json": "guarda en resultados/ y no en data/",
    "cargar_json": "lee de resultados/ y no de data/",
    "explorar_patron": "acepta arranques puramente al azar y guarda el tipo de arranque",
    "control_ingenuo": "recibe la curva en lugar de leer data/s7_curva.json",
    "plantilla_de_la_curva": "el docstring nombra resultados/ y no data/",
    "elegir_grafo": "el docstring nombra resultados/ y no data/",
}


def definiciones(texto):
    """{nombre: código} de las funciones, clases y constantes de nivel superior (con decoradores)."""
    lineas = texto.splitlines()
    resultado = {}
    for nodo in ast.parse(texto).body:
        if isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
            nombre = nodo.name
        elif isinstance(nodo, ast.Assign) and isinstance(nodo.targets[0], ast.Name):
            nombre = nodo.targets[0].id
        else:
            continue
        inicio = nodo.lineno
        if getattr(nodo, "decorator_list", None):
            inicio = min(d.lineno for d in nodo.decorator_list)
        resultado[nombre] = "\n".join(lineas[inicio - 1: nodo.end_lineno])
    return resultado


def celdas_de_biblioteca():
    """Código de las celdas que empiezan con '# biblioteca'."""
    notebook = json.loads((RAIZ / "notebooks" / "exploracion_s7_final.ipynb").read_text())
    codigos = []
    for celda in notebook["cells"]:
        codigo = "".join(celda["source"])
        if celda["cell_type"] == "code" and codigo.startswith("# biblioteca"):
            codigos.append(codigo)
    return codigos


def test_copias_del_notebook_iguales_a_los_originales():
    originales = {}
    for ruta in ORIGENES:
        for nombre, fuente in definiciones((RAIZ / ruta).read_text()).items():
            originales.setdefault(nombre, []).append(fuente)
    copias = {}
    for codigo in celdas_de_biblioteca():
        for nombre, fuente in definiciones(codigo).items():
            if nombre in originales and nombre not in ADAPTADAS:
                copias[nombre] = fuente
    assert len(copias) >= 120, f"se esperaban ~130 definiciones copiadas, hay {len(copias)}"
    distintas = []
    for nombre, fuente in copias.items():
        if fuente not in originales[nombre]:
            distintas.append(nombre)
    assert not distintas, f"distintas de su original: {distintas}"


def test_adaptadas_siguen_en_el_notebook():
    nombres = set()
    for codigo in celdas_de_biblioteca():
        nombres |= set(definiciones(codigo))
    assert set(ADAPTADAS) <= nombres
