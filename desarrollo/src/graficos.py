"""Dibujos de fases sobre el círculo."""
import numpy as np

from .kuramoto import parametro_orden


def dibujar_circulo(ax, theta, titulo=""):
    """Partículas e^{iθ_v} sobre el círculo unitario y la flecha del parámetro de orden r."""
    s = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(s), np.sin(s), color="0.7", lw=1)
    ax.scatter(np.cos(theta), np.sin(theta), s=45, color="tab:orange", zorder=3, edgecolor="k", lw=0.5)
    r = parametro_orden(theta)
    ax.annotate("", xy=(r.real, r.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set(aspect="equal", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), xticks=[], yticks=[])
    ax.set_title(titulo, fontsize=9)


def dibujar_ciclo(ax, theta, titulo=""):
    """Como dibujar_circulo, más las aristas del ciclo v → v + 1."""
    z = np.exp(1j * np.append(theta, theta[0]))
    ax.plot(z.real, z.imag, color="tab:blue", lw=0.8, alpha=0.6, zorder=2)
    dibujar_circulo(ax, theta, titulo)
