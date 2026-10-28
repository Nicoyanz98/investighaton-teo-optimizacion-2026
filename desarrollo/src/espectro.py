"""Espectro completo del Hessiano del blow-up por bloques de Fourier, en float64.

H conmuta con el corrimiento v → v + 1 en todos los bloques Z_g, así que es unitariamente equivalente a
g matrices hermíticas H_m de K × K (K = Σ k_a), una por frecuencia m (ver src/blowup.py):
  H_m[(a,i),(a,i)] = Σ_b d_ab cos_ab − (g·[m = 0] − 1),   H_m[(a,i),(a,j)] = −g·[m = 0]  (i ≠ j),
  H_m[(a,i),(b,j)] = −cos_ab · N̂_ij(m),   N̂_ij(m) = Σ_{s<t_ij} e^{2πi m s/g}.
A diferencia de Lanczos, que solo da una cota superior del menor autovalor, acá se calculan todos los
autovalores. Como H_{g−m} = conj(H_m), alcanza con m = 0, …, ⌊g/2⌋.

Se lee solo la descripción compacta (k, g, grados, tablas, fases), igual que src/validacion.py.
"""
import numpy as np


def inicios_bloques(k):
    """Índice del primer bloque de cada cluster dentro de los K bloques."""
    return np.cumsum([0] + list(k[:-1]))


def transformada_intervalos(tabla, g, frecuencias):
    """N̂_ij(m) = Σ_{s<t_ij} e^{2πi m s/g} con forma (M, k_a, k_b), por la fórmula de la suma geométrica.

    N̂ = e^{iπm(t−1)/g} sin(πmt/g) / sin(πm/g), y t si m = 0. Los productos m·t se reducen mod 2g en
    enteros, así los ángulos quedan en [0, 2π) sin perder precisión aunque m·t sea grande.
    """
    t = np.asarray(tabla, dtype=np.int64)[None, :, :]
    m = np.asarray(frecuencias, dtype=np.int64)[:, None, None]
    fase = np.pi * ((m * (t - 1)) % (2 * g)) / g
    numerador = np.sin(np.pi * ((m * t) % (2 * g)) / g)
    denominador = np.sin(np.pi * m / g)
    cociente = np.divide(numerador, denominador, out=np.zeros(numerador.shape), where=denominador != 0)
    return np.where(m == 0, t.astype(float), np.exp(1j * fase) * cociente)


def bloques_fourier(descripcion, frecuencias):
    """Bloques H_m (M × K × K, hermíticos) del Hessiano, para las frecuencias dadas."""
    k = list(descripcion["k"])
    g = int(descripcion["g"])
    d = np.array(descripcion["grados"], dtype=float)
    fases = np.array([float(p) for p in descripcion["fases"]])
    coseno = np.cos(fases[:, None] - fases[None, :])
    frecuencias = np.asarray(frecuencias, dtype=np.int64)
    inicios = inicios_bloques(k)
    K = sum(k)
    H = np.zeros((len(frecuencias), K, K), dtype=complex)
    clique = np.where(frecuencias == 0, g, 0).astype(float)
    for a, ka in enumerate(k):
        indices = np.arange(inicios[a], inicios[a] + ka)
        H[:, indices[:, None], indices[None, :]] = -clique[:, None, None] * np.ones((ka, ka))
        diagonal = float((d[a] * coseno[a]).sum())
        H[:, indices, indices] = diagonal - (clique[:, None] - 1)
    for clave, tabla in descripcion["tablas"].items():
        a, b = (int(c) for c in clave.split(","))
        N = transformada_intervalos(tabla, g, frecuencias)
        filas = slice(inicios[a], inicios[a] + k[a])
        columnas = slice(inicios[b], inicios[b] + k[b])
        H[:, filas, columnas] = -coseno[a, b] * N
        H[:, columnas, filas] = -coseno[a, b] * np.conj(N).transpose(0, 2, 1)
    return H


def producto_fourier(descripcion, x, lote=4096):
    """H·x por bloques: FFT en cada bloque Z_g, H_m en cada frecuencia y FFT inversa."""
    K = sum(descripcion["k"])
    g = int(descripcion["g"])
    X = np.fft.fft(np.asarray(x, dtype=float).reshape(K, g), axis=1)
    Y = np.empty_like(X)
    for inicio in range(0, g, lote):
        m = np.arange(inicio, min(inicio + lote, g))
        Y[:, m] = np.einsum("mij,jm->im", bloques_fourier(descripcion, m), X[:, m])
    return np.fft.ifft(Y, axis=1).real.reshape(-1)


def base_cluster(k):
    """Matriz K × len(k): columna a = indicadora de los bloques del cluster a, normalizada."""
    C = np.zeros((sum(k), len(k)))
    inicios = inicios_bloques(k)
    for a, ka in enumerate(k):
        C[inicios[a]: inicios[a] + ka, a] = 1 / np.sqrt(ka)
    return C


def espectro_fourier(descripcion, lote=4096, informar=None):
    """Todos los autovalores de H, agrupados.

    - "cluster": los de los modos constantes por cluster (H_0 restringida a la base de clusters);
    - "cero": los del resto de H_0 (diferencias entre bloques de un mismo cluster);
    - "minimos": el menor autovalor de H_m para cada m de "frecuencias" (1, …, ⌊g/2⌋);
    - "acople": max |Cᵀ H_0 C⊥|, que es 0 si los modos de cluster son invariantes (grados exactos);
    - "maximo": el mayor autovalor de H.
    `informar(hechas, total)` recibe el progreso.
    """
    k = list(descripcion["k"])
    g = int(descripcion["g"])
    H0 = bloques_fourier(descripcion, [0])[0].real
    C = base_cluster(k)
    completa, _ = np.linalg.qr(C, mode="complete")
    resto = completa[:, len(k):]
    cluster = np.linalg.eigvalsh(C.T @ H0 @ C)
    cero = np.linalg.eigvalsh(resto.T @ H0 @ resto)
    frecuencias = np.arange(1, g // 2 + 1)
    minimos = np.empty(len(frecuencias))
    maximo = max(cluster.max(), cero.max())
    for inicio in range(0, len(frecuencias), lote):
        tramo = slice(inicio, min(inicio + lote, len(frecuencias)))
        valores = np.linalg.eigvalsh(bloques_fourier(descripcion, frecuencias[tramo]))
        minimos[tramo] = valores[:, 0]
        maximo = max(maximo, float(valores[:, -1].max()))
        if informar is not None:
            informar(tramo.stop, len(frecuencias))
    return {
        "cluster": cluster, "cero": cero, "frecuencias": frecuencias, "minimos": minimos,
        "acople": float(np.abs(C.T @ H0 @ resto).max()), "maximo": float(maximo),
    }


def curva_minimos(frecuencias, minimos, g, puntos=2000):
    """(m/g, menor autovalor) reducida a ~`puntos` tramos, tomando el mínimo de cada tramo."""
    tramos = np.array_split(np.arange(len(minimos)), min(puntos, len(minimos)))
    abscisas = []
    valores = []
    for tramo in tramos:
        mejor = tramo[np.argmin(minimos[tramo])]
        abscisas.append(float(frecuencias[mejor] / g))
        valores.append(float(minimos[mejor]))
    return abscisas, valores
