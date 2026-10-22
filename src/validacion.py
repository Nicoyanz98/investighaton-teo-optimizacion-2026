"""Validación independiente del blow-up: productos W·y y H·x implícitos (sin armar la matriz) y Lanczos.

Solo usa la regla de adyacencia de la descripción compacta (ver src/blowup.py):
  - (a, i, v) ~ (a, i', v') para todo par distinto dentro del mismo cluster (clique);
  - (a, i, v) ~ (b, j, v + s mod g) con s ∈ {0, …, t_ij − 1}, para a < b.
No usa la descomposición de Fourier ni las cotas de Gershgorin: es un chequeo ciego del certificado.
Las sumas sobre intervalos se hacen con sumas prefijas, así que un producto cuesta O(n·k) y la
memoria es O(n). Corre con numpy (CPU) o con torch (CPU o CUDA), siempre en float64.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal


# ---------------------------------------------------------------- arreglos numpy / torch

class Arreglos:
    """Las pocas operaciones que hacen falta, con numpy (dispositivo None) o con torch."""

    def __init__(self, dispositivo=None):
        self.torch = None
        if dispositivo is not None:
            import torch
            self.torch = torch
            self.dispositivo = torch.device(dispositivo)

    def desde(self, x):
        """numpy → arreglo del backend (float64)."""
        x = np.asarray(x, dtype=np.float64)
        if self.torch is None:
            return x
        return self.torch.from_numpy(x).to(self.dispositivo)

    def ceros(self, n):
        """Vector de ceros de largo n."""
        return self.desde(np.zeros(n))

    def aleatorio(self, n, semilla):
        """Vector gaussiano de largo n (generado en CPU para que sea reproducible)."""
        return self.desde(np.random.default_rng(semilla).standard_normal(n))

    def a_numpy(self, x):
        """Arreglo del backend → numpy."""
        if self.torch is None:
            return np.asarray(x)
        return x.cpu().numpy()

    def escalar(self, x):
        """Escalar del backend → float."""
        return float(x)

    def suma_prefija_doble(self, Y):
        """S de forma (k, 2g + 1): S[:, l] = suma de las primeras l entradas de [Y, Y] (fila por fila)."""
        if self.torch is None:
            Z = np.concatenate([Y, Y], axis=1)
            return np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(Z, axis=1)], axis=1)
        Z = self.torch.cat([Y, Y], dim=1)
        ceros = self.torch.zeros((Y.shape[0], 1), dtype=Y.dtype, device=Y.device)
        return self.torch.cat([ceros, self.torch.cumsum(Z, dim=1)], dim=1)

    def media_filas(self, Y):
        """Media de cada fila, como columna (k, 1)."""
        if self.torch is None:
            return Y.mean(axis=1, keepdims=True)
        return Y.mean(dim=1, keepdim=True)

    def punto(self, x, y):
        """Producto escalar ⟨x, y⟩."""
        return self.escalar((x * y).sum())

    def norma(self, x):
        """Norma euclídea."""
        return self.punto(x, x) ** 0.5

    def coseno_seno(self, x):
        """(cos x, sin x) elemento a elemento."""
        if self.torch is None:
            return np.cos(x), np.sin(x)
        return self.torch.cos(x), self.torch.sin(x)

    def tomar(self, x, indices):
        """x[indices] como numpy (indices: arreglo numpy de enteros)."""
        if self.torch is None:
            return np.asarray(x[indices])
        return x[self.torch.from_numpy(indices).to(self.dispositivo)].cpu().numpy()


# ---------------------------------------------------------------- operador del grafo

class GrafoImplicito:
    """Blow-up de la descripción compacta, con W·y y H·x implícitos.

    Vértice (a, i, v) ↦ índice inicio_a + i·g + v (el mismo orden que `blowup.adyacencia_densa`).
    """

    def __init__(self, descripcion, arreglos):
        self.arreglos = arreglos
        self.k = list(descripcion["k"])
        self.g = int(descripcion["g"])
        self.n = sum(self.k) * self.g
        self.inicios = [0]
        for ka in self.k[:-1]:
            self.inicios.append(self.inicios[-1] + ka * self.g)
        self.mascaras = {}
        for clave, tabla in descripcion["tablas"].items():
            a, b = (int(c) for c in clave.split(","))
            self.mascaras[(a, b)] = self._mascaras_tabla(np.array(tabla))
        fases_cluster = [float(p) for p in descripcion["fases"]]
        theta = np.repeat(fases_cluster, [ka * self.g for ka in self.k])
        self.coseno = arreglos.desde(np.cos(theta))
        self.seno = arreglos.desde(np.sin(theta))
        self.diagonal = (self.coseno * self.producto_adyacencia(self.coseno)
                         + self.seno * self.producto_adyacencia(self.seno))

    def _mascaras_tabla(self, tabla):
        """[(t, M_t)]: para cada valor t de la tabla, la matriz 0/1 de los pares (i, j) con t_ij = t."""
        assert tabla.min() >= 0 and tabla.max() <= self.g
        mascaras = []
        for t in np.unique(tabla):
            if t > 0:
                mascaras.append((int(t), self.arreglos.desde((tabla == t).astype(float))))
        return mascaras

    def por_cluster(self, y):
        """Vistas (k_a, g) de y, una por cluster (escribir en ellas escribe en y)."""
        partes = []
        for a, ka in enumerate(self.k):
            partes.append(y[self.inicios[a]: self.inicios[a] + ka * self.g].reshape(ka, self.g))
        return partes

    def producto_adyacencia(self, y):
        """W·y. Las ventanas cíclicas salen de sumas prefijas de y centrado por bloque (más preciso)."""
        g = self.g
        salida = self.arreglos.ceros(self.n)
        partes = self.por_cluster(y)
        partes_salida = self.por_cluster(salida)
        medias = []
        prefijas = []
        for a, Y in enumerate(partes):
            partes_salida[a] += Y.sum() - Y
            media = self.arreglos.media_filas(Y)
            medias.append(media)
            prefijas.append(self.arreglos.suma_prefija_doble(Y - media))
        for (a, b), mascaras in self.mascaras.items():
            for t, M in mascaras:
                # (a, i, v) recibe Σ_{s<t} y_(b, j, v+s);  (b, j, w) recibe Σ_{s<t} y_(a, i, w−s)
                adelante = prefijas[b][:, t: t + g] - prefijas[b][:, :g] + t * medias[b]
                atras = prefijas[a][:, g + 1: 2 * g + 1] - prefijas[a][:, g + 1 - t: 2 * g + 1 - t] + t * medias[a]
                partes_salida[a] += M @ adelante
                partes_salida[b] += M.T @ atras
        return salida

    def producto_hessiano(self, x):
        """H·x con H = diag(Σ_v W_uv cos(θ_u − θ_v)) − [W_uv cos(θ_u − θ_v)]."""
        cruzado = (self.coseno * self.producto_adyacencia(self.coseno * x)
                   + self.seno * self.producto_adyacencia(self.seno * x))
        return self.diagonal * x - cruzado

    def velocidad(self):
        """θ̇_u = Σ_v W_uv sin(θ_v − θ_u) en las fases del grafo."""
        return self._velocidad(self.coseno, self.seno)

    def velocidad_en(self, theta):
        """θ̇_u = Σ_v W_uv sin(θ_v − θ_u) en fases θ cualesquiera (para simular la dinámica)."""
        coseno, seno = self.arreglos.coseno_seno(theta)
        return self._velocidad(coseno, seno)

    def _velocidad(self, coseno, seno):
        """cos θ ∘ W sin θ − sin θ ∘ W cos θ, que es Σ_v W_uv sin(θ_v − θ_u)."""
        return coseno * self.producto_adyacencia(seno) - seno * self.producto_adyacencia(coseno)

    def indicadora(self, b):
        """Vector 1 en los vértices del cluster b."""
        y = self.arreglos.ceros(self.n)
        y[self.inicios[b]: self.inicios[b] + self.k[b] * self.g] = 1
        return y

    def quitar_medias_cluster(self, x):
        """Proyección al complemento de los modos de cluster (resta la media de cada cluster)."""
        x = x.clone() if self.arreglos.torch is not None else x.copy()
        for X in self.por_cluster(x):
            X -= X.mean()
        return x

    def quitar_media(self, x):
        """Proyección a 1^⊥ (quita la rotación global)."""
        return x - x.mean()


# ---------------------------------------------------------------- chequeos

def grados_medidos(grafo):
    """Matriz (m × m) con [min, max] de los vecinos que tiene un vértice de a en b, contados con W·1_b."""
    m = len(grafo.k)
    minimos = np.zeros((m, m), dtype=np.int64)
    maximos = np.zeros((m, m), dtype=np.int64)
    for b in range(m):
        cuenta = grafo.arreglos.a_numpy(grafo.producto_adyacencia(grafo.indicadora(b)))
        for a in range(m):
            tramo = cuenta[grafo.inicios[a]: grafo.inicios[a] + grafo.k[a] * grafo.g]
            minimos[a, b] = int(np.rint(tramo.min()))
            maximos[a, b] = int(np.rint(tramo.max()))
    return minimos, maximos


def cociente_implicito(grafo):
    """Q_ab = 1_aᵀ H 1_b / √(n_a n_b), calculada con el producto implícito."""
    m = len(grafo.k)
    Q = np.zeros((m, m))
    tamanos = [ka * grafo.g for ka in grafo.k]
    for b in range(m):
        columna = grafo.arreglos.a_numpy(grafo.producto_hessiano(grafo.indicadora(b)))
        for a in range(m):
            tramo = columna[grafo.inicios[a]: grafo.inicios[a] + tamanos[a]]
            Q[a, b] = tramo.sum() / np.sqrt(tamanos[a] * tamanos[b])
    return Q


def cociente_formula(descripcion):
    """La misma Q con la fórmula de los grados: Q_ab = −√(n_a/n_b) d_ab cos_ab, Q_aa = Σ_{c≠a} d_ac cos_ac."""
    k = descripcion["k"]
    g = descripcion["g"]
    d = np.array(descripcion["grados"], dtype=float)
    fases = np.array([float(p) for p in descripcion["fases"]])
    coseno = np.cos(fases[:, None] - fases[None, :])
    tamanos = np.array(k, dtype=float) * g
    Q = -np.sqrt(tamanos[:, None] / tamanos[None, :]) * d * coseno
    for a in range(len(k)):
        Q[a, a] = sum(d[a, c] * coseno[a, c] for c in range(len(k)) if c != a)
    return Q


# ---------------------------------------------------------------- Lanczos

def _extremos_ritz(alfas, betas):
    """(menor valor de Ritz, último componente de su vector, mayor valor de Ritz) de la tridiagonal."""
    if len(alfas) == 1:
        return alfas[0], 1.0, alfas[0]
    menor, vector = eigh_tridiagonal(alfas, betas[:-1], select="i", select_range=(0, 0))
    mayor = eigh_tridiagonal(alfas, betas[:-1], eigvals_only=True, select="i",
                             select_range=(len(alfas) - 1, len(alfas) - 1))
    return float(menor[0]), float(vector[-1, 0]), float(mayor[0])


def lanczos(aplicar, proyectar, inicial, arreglos, iteraciones, cada=25, tolerancia=1e-6, informar=None):
    """Lanczos sin reortogonalizar para el menor autovalor de un operador simétrico.

    El residuo ‖H y − θ y‖ del vector de Ritz es β_k·|s_k| (último componente), sin calcular y.
    Corta cuando ese residuo es ≤ tolerancia·|θ|, o si β se anula (el Krylov ya es invariante).
    `informar(fila)` recibe el progreso cada `cada` pasos.
    """
    q = proyectar(inicial)
    q = q / arreglos.norma(q)
    q_anterior = None
    beta = 0.0
    alfas = []
    betas = []
    historial = []
    escala = 0.0
    for paso in range(1, iteraciones + 1):
        w = proyectar(aplicar(q))
        alfa = arreglos.punto(q, w)
        w = w - alfa * q
        if q_anterior is not None:
            w = w - beta * q_anterior
        # reproyectar: si β es chico, w/β amplifica el redondeo en las direcciones proyectadas
        # (donde el operador proyectado vale 0) y aparece un valor de Ritz espurio ≈ 0
        w = proyectar(w)
        beta = arreglos.norma(w)
        alfas.append(alfa)
        betas.append(beta)
        escala = max(escala, abs(alfa))
        final = paso == iteraciones or beta <= 1e-12 * escala
        if paso % cada == 0 or final:
            menor, ultimo, mayor = _extremos_ritz(np.array(alfas), np.array(betas))
            fila = {"paso": paso, "menor": menor, "residuo": beta * abs(ultimo), "mayor": mayor}
            historial.append(fila)
            if informar is not None:
                informar(fila)
            if fila["residuo"] <= tolerancia * abs(menor) or final:
                break
        q_anterior = q
        q = w / beta
    return {"menor": historial[-1]["menor"], "residuo": historial[-1]["residuo"],
            "mayor": historial[-1]["mayor"], "pasos": historial[-1]["paso"],
            "convergio": historial[-1]["residuo"] <= tolerancia * abs(historial[-1]["menor"]),
            "historial": historial}
