"""Calentamientos.ipynb es autocontenido (corre en Colab sin el repo): sus funciones son copias de src/.

Este test avisa si una copia del notebook y su original en src/ dejaron de ser idénticas.
"""
import ast
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MODULOS = ["grafos", "kuramoto", "graficos"]


def definiciones(texto):
    """{nombre: código} de las funciones y constantes de nivel superior de un texto de Python."""
    arbol = ast.parse(texto)
    resultado = {}
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef):
            resultado[nodo.name] = ast.get_source_segment(texto, nodo)
        elif isinstance(nodo, ast.Assign) and isinstance(nodo.targets[0], ast.Name):
            resultado[nodo.targets[0].id] = ast.get_source_segment(texto, nodo)
    return resultado


def test_funciones_del_notebook_iguales_a_src():
    originales = {}
    for modulo in MODULOS:
        originales.update(definiciones((RAIZ / "src" / f"{modulo}.py").read_text()))
    notebook = json.loads((RAIZ / "notebooks" / "Calentamientos.ipynb").read_text())
    copias = {}
    for celda in notebook["cells"]:
        if celda["cell_type"] == "code":
            codigo = "".join(celda["source"])
            for nombre, fuente in definiciones(codigo).items():
                if nombre in originales and fuente.startswith(("def ", nombre + " =")):
                    copias[nombre] = fuente
    assert len(copias) >= 20, f"se esperaban ~23 funciones copiadas, hay {len(copias)}"
    distintas = []
    for nombre, fuente in copias.items():
        if fuente != originales[nombre]:
            distintas.append(nombre)
    assert not distintas, f"distintas de src/: {distintas}"
