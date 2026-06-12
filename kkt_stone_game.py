# -*- coding: utf-8 -*-
"""
Juego interactivo KKT: piedras, reparación al poliedro y convergencia al óptimo con momentum.

Ejecutar:
    python kkt_stone_game.py

Requisitos:
    pip install numpy matplotlib

Si estás en Jupyter:
    %matplotlib qt
    %run kkt_stone_game.py
"""

import itertools
import math
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button


# ============================================================
# 1. Problema
# ============================================================

Q = np.array([
    [0.08,  0.015, 0.000, 0.000, 0.000, 0.005],
    [0.015, 0.100, 0.020, 0.000, 0.000, 0.000],
    [0.000, 0.020, 0.120, 0.015, 0.000, 0.000],
    [0.000, 0.000, 0.015, 0.090, 0.020, 0.000],
    [0.000, 0.000, 0.000, 0.020, 0.110, 0.015],
    [0.005, 0.000, 0.000, 0.000, 0.015, 0.130]
], dtype=float)

b = np.array([-1.375, 0.815, 2.810, 3.400, 7.110, 9.375], dtype=float)

LOWER = 5.0
UPPER = 30.0
TOTAL = 100.0
N = 6

# Óptimo construido del problema de maximización
X_OPT = np.array([5, 5, 12, 18, 30, 30], dtype=float)
NU_OPT = 1.0

# Punto inicial: todos iguales
X0 = np.repeat(TOTAL / N, N)

H = 0.001


def f(x):
    """Función objetivo: max b'x - 1/2 x'Qx."""
    x = np.asarray(x, dtype=float)
    return float(b @ x - 0.5 * x @ Q @ x)


def grad_f(x):
    """Gradiente de f."""
    x = np.asarray(x, dtype=float)
    return b - Q @ x


def project_box_sum(y, lower=LOWER, upper=UPPER, total=TOTAL, max_iter=100):
    """
    Proyección euclidiana sobre:
        {x: lower <= x_i <= upper, sum(x)=total}

    Usa búsqueda binaria sobre el multiplicador de la suma.
    """
    y = np.asarray(y, dtype=float)

    lo = np.min(y - upper)
    hi = np.max(y - lower)

    for _ in range(max_iter):
        tau = 0.5 * (lo + hi)
        x = np.clip(y - tau, lower, upper)

        if x.sum() > total:
            lo = tau
        else:
            hi = tau

    return np.clip(y - 0.5 * (lo + hi), lower, upper)


def net_marginal_delta(x):
    """
    Eje y del gráfico:
        Δf_i - NU_OPT * 0.001

    Es una diferencia finita de aumentar x_i en 0.001,
    neta del precio sombra de la restricción de suma.
    """
    x = np.asarray(x, dtype=float)
    fx = f(x)

    out = []
    for i in range(N):
        x_new = x.copy()
        x_new[i] += H
        out.append((f(x_new) - fx) - NU_OPT * H)

    return np.array(out)


def enumerate_vertices_box_sum():
    """
    Vértices del poliedro:
        5 <= x_i <= 30, sum_i x_i = 100

    En dimensión 6, un vértice tiene 5 variables en cota
    y una variable libre determinada por la suma.
    """
    candidates = []

    for free in range(N):
        fixed = [j for j in range(N) if j != free]

        for bounds in itertools.product([LOWER, UPPER], repeat=N - 1):
            x = np.empty(N)
            x[fixed] = bounds
            x[free] = TOTAL - np.sum(bounds)

            if LOWER - 1e-10 <= x[free] <= UPPER + 1e-10:
                candidates.append(np.round(x, 12))

    # Por robustez: revisar también todos los puntos de cotas completas
    for bounds in itertools.product([LOWER, UPPER], repeat=N):
        x = np.array(bounds, dtype=float)
        if abs(x.sum() - TOTAL) < 1e-10:
            candidates.append(x)

    return np.unique(np.array(candidates), axis=0)


def find_feasible_minimum():
    """
    Como f es cóncava, el mínimo sobre un poliedro compacto
    ocurre en algún vértice.
    """
    vertices = enumerate_vertices_box_sum()
    vals = np.array([f(v) for v in vertices])
    idx = int(np.argmin(vals))
    return vertices[idx], float(vals[idx])


X_MIN, F_MIN = find_feasible_minimum()
F_OPT = f(X_OPT)


def progress_value(x):
    """
    Normalización de f:
        0% = mínimo factible del problema
        100% = óptimo de maximización
    """
    val = (f(x) - F_MIN) / (F_OPT - F_MIN)
    return float(np.clip(val, 0.0, 1.0))


def alpha_from_value(x):
    """
    Transparencia:
        f = mínimo  -> 80% transparente -> alpha = 0.20
        f = óptimo  -> 0% transparente  -> alpha = 1.00
    """
    p = progress_value(x)
    return 0.20 + 0.80 * p


def kkt_residual(x, eps=1e-4):
    """
    Residuo KKT ilustrativo usando el multiplicador óptimo NU_OPT = 1.

    En el óptimo:
        si x_i = lower,  grad_i - nu <= 0
        si interior,     grad_i - nu = 0
        si x_i = upper,  grad_i - nu >= 0
    """
    x = np.asarray(x, dtype=float)
    r = grad_f(x) - NU_OPT
    residuals = []

    for xi, ri in zip(x, r):
        if xi <= LOWER + eps:
            residuals.append(max(0.0, ri))
        elif xi >= UPPER - eps:
            residuals.append(max(0.0, -ri))
        else:
            residuals.append(abs(ri))

    return float(np.max(residuals))


# ============================================================
# 2. Sonidos
# ============================================================

def _try_winsound_beeps(seq):
    """
    Sonido no bloqueante.
    En Windows usa winsound. En otros sistemas intenta campana terminal.
    """
    try:
        import winsound
        for freq, dur in seq:
            winsound.Beep(int(freq), int(dur))
    except Exception:
        # Fallback simple: campana terminal
        print("\a", end="", flush=True)


def play_stone_sound():
    seq = [(170, 55), (95, 85)]
    threading.Thread(target=_try_winsound_beeps, args=(seq,), daemon=True).start()


def play_optimum_sound():
    seq = [(523, 110), (659, 110), (784, 160), (1046, 220)]
    threading.Thread(target=_try_winsound_beeps, args=(seq,), daemon=True).start()


# ============================================================
# 3. Juego interactivo
# ============================================================

class KKTStoneGame:
    def __init__(self):
        self.rng = np.random.default_rng()

        # Estado
        self.x = X0.copy()
        self.mode = "optimize"  # optimize, stone, repair, done
        self.iterations = 0
        self.stones = 0

        self.reset_allowed = False
        self.victory_played = False

        # Parámetros visuales / dinámicos
        self.interval_ms = 80
        self.stone_seconds = 1.00
        self.stone_frames = max(2, int(round(1000 * self.stone_seconds / self.interval_ms)))

        # Momentum adaptativo:
        # - Lejos: eta alta + momentum alto
        # - Cerca: eta baja + momentum bajo
        # - Muy cerca: aterrizaje lento en 5 pasos exactos
        self.eta_min = 0.035
        self.eta_max = 0.58
        self.momentum_min = 0.15
        self.momentum_max = 0.86
        self.velocity = np.zeros(N)

        self.landing_threshold = 0.50
        self.landing_steps_total = 5
        self.landing_remaining = 0
        self.landing_start = None

        self.repair_rate = 0.10  # suavidad al regresar a sum(x)=100
        self.opt_tol = 0.004

        self.stone_t = 0
        self.stone_start = None
        self.stone_target = None
        self.stone_origin = None
        self.repair_target = None

        self.colors = [
            "tab:blue", "tab:orange", "tab:green",
            "tab:red", "tab:purple", "tab:brown"
        ]

        self._setup_figure()
        self.update_reset_button()

        self.anim = FuncAnimation(
            self.fig,
            self.update,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False
        )

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title("Juego KKT con piedras")

        self.ax = self.fig.add_axes([0.08, 0.25, 0.70, 0.65])
        self.bar_ax = self.fig.add_axes([0.84, 0.25, 0.08, 0.65])

        self.ax_stone = self.fig.add_axes([0.08, 0.07, 0.20, 0.075])
        self.ax_reset = self.fig.add_axes([0.31, 0.07, 0.20, 0.075])

        self.btn_stone = Button(self.ax_stone, "Lanzar piedra")
        self.btn_reset = Button(self.ax_reset, "Reiniciar")

        self.btn_stone.on_clicked(self.throw_stone)
        self.btn_reset.on_clicked(self.reset_game)

        self.metrics_text = self.fig.text(
            0.08, 0.17, "",
            fontsize=11,
            ha="left",
            va="center"
        )

        self.info_text = self.fig.text(
            0.55, 0.08,
            f"Mínimo factible: f={F_MIN:.2f} en {np.round(X_MIN, 2)}     |     Óptimo: f={F_OPT:.2f}",
            fontsize=10,
            ha="left",
            va="center",
            color="0.25"
        )

    def update_reset_button(self):
        if self.reset_allowed:
            self.btn_reset.label.set_text("Reiniciar")
            self.ax_reset.set_facecolor("#DFF2E1")
        else:
            self.btn_reset.label.set_text("Reiniciar bloqueado")
            self.ax_reset.set_facecolor("#EEEEEE")

        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    # Eventos
    # --------------------------------------------------------

    def throw_stone(self, event=None):
        """
        La piedra manda el estado a un punto aleatorio del cubo [5,30]^6.
        Ese punto NO tiene por qué cumplir sum(x)=100.
        """
        self.stones += 1
        self.mode = "stone"
        self.victory_played = False
        self.reset_allowed = False
        self.update_reset_button()

        self.velocity[:] = 0.0
        self.landing_remaining = 0
        self.landing_start = None

        self.stone_t = 0
        self.stone_start = self.x.copy()
        self.stone_target = self.rng.uniform(LOWER, UPPER, size=N)
        self.stone_origin = np.array([32.0, 0.0050])

        play_stone_sound()

    def reset_game(self, event=None):
        if not self.reset_allowed:
            return

        self.x = X0.copy()
        self.mode = "optimize"
        self.iterations = 0
        self.stones = 0
        self.velocity[:] = 0.0
        self.landing_remaining = 0
        self.landing_start = None

        self.reset_allowed = False
        self.victory_played = False
        self.update_reset_button()

    # --------------------------------------------------------
    # Dinámica
    # --------------------------------------------------------

    def step_stone(self):
        """
        Piedra de 1 segundo:
        - El estado se mueve con easing elástico hacia un punto aleatorio.
        - Visualmente se dibuja una piedra parabólica con estela en draw_main_panel().
        """
        self.stone_t += 1
        u = min(1.0, self.stone_t / self.stone_frames)

        # Movimiento del sistema: smoothstep + leve overshoot amortiguado
        smooth = u * u * (3 - 2 * u)
        wobble = 0.06 * math.sin(5 * math.pi * u) * (1 - u)
        s = np.clip(smooth + wobble, 0.0, 1.0)

        self.x = self.stone_start + s * (self.stone_target - self.stone_start)

        if u >= 1.0:
            self.x = self.stone_target.copy()
            self.repair_target = project_box_sum(self.x)
            self.mode = "repair"

    def step_repair(self):
        """
        Transición suave al poliedro de restricciones.
        Respeta el cubo porque ambos extremos están en [5,30]^6.
        """
        if self.repair_target is None:
            self.repair_target = project_box_sum(self.x)

        self.x = self.x + self.repair_rate * (self.repair_target - self.x)
        self.iterations += 1

        close_to_polyhedron = (
            abs(self.x.sum() - TOTAL) < 0.02
            and np.max(np.abs(self.x - self.repair_target)) < 0.02
        )

        if close_to_polyhedron:
            self.x = self.repair_target.copy()
            self.velocity[:] = 0.0
            self.landing_remaining = 0
            self.landing_start = None
            self.mode = "optimize"

    def step_optimize(self):
        """
        Ascenso proyectado de f con momentum adaptativo.

        Regla:
        - Si está lejos del óptimo, se mueve rápido y con más momentum.
        - Si está cerca, baja velocidad y momentum.
        - Si max |x-x*| < 0.5, entra en aterrizaje de 5 pasos,
          acercándose lentamente al óptimo exacto.
        """
        if self.has_reached_optimum():
            self.reach_optimum()
            return

        max_dist = float(np.max(np.abs(self.x - X_OPT)))

        # Aterrizaje final: exactamente 5 pasos visuales lentos.
        if max_dist < self.landing_threshold:
            if self.landing_remaining <= 0:
                self.landing_remaining = self.landing_steps_total
                self.landing_start = self.x.copy()
                self.velocity[:] = 0.0

            # ease-out lento: cada frame consume una fracción del gap restante
            frac = 1.0 / self.landing_remaining
            self.x = self.x + frac * (X_OPT - self.x)
            self.landing_remaining -= 1
            self.iterations += 1

            if self.landing_remaining <= 0 or self.has_reached_optimum():
                self.reach_optimum()
            return

        # Momentum adaptativo, normalizado por distancia.
        closeness = np.clip(max_dist / 12.0, 0.0, 1.0)
        eta = self.eta_min + (self.eta_max - self.eta_min) * closeness
        mom = self.momentum_min + (self.momentum_max - self.momentum_min) * closeness

        proposed_velocity = mom * self.velocity + eta * grad_f(self.x)
        x_candidate = project_box_sum(self.x + proposed_velocity)

        # La velocidad real es el desplazamiento tras proyectar al poliedro.
        self.velocity = x_candidate - self.x
        self.x = x_candidate
        self.iterations += 1

        if self.has_reached_optimum():
            self.reach_optimum()

    def has_reached_optimum(self):
        return (
            abs(self.x.sum() - TOTAL) < 1e-6
            and np.max(np.abs(self.x - X_OPT)) < self.opt_tol
        )

    def reach_optimum(self):
        self.x = X_OPT.copy()
        self.mode = "done"
        self.reset_allowed = True
        self.update_reset_button()

        if not self.victory_played:
            play_optimum_sound()
            self.victory_played = True

    # --------------------------------------------------------
    # Dibujo
    # --------------------------------------------------------

    def draw_main_panel(self):
        self.ax.clear()

        y = net_marginal_delta(self.x)
        g_eff = grad_f(self.x) - NU_OPT

        alpha = alpha_from_value(self.x)

        self.ax.axvline(LOWER, linestyle="--", linewidth=1.5, color="steelblue")
        self.ax.axvline(UPPER, linestyle="--", linewidth=1.5, color="steelblue")
        self.ax.axhline(0, linewidth=1.2, color="steelblue", alpha=0.8)

        self.ax.text(LOWER + 0.2, 0.0042, "min = 5", color="steelblue", fontsize=10)
        self.ax.text(UPPER - 2.8, 0.0042, "max = 30", color="steelblue", fontsize=10)

        # Flechas y puntos
        for i in range(N):
            xi = self.x[i]
            yi = y[i]
            gi = g_eff[i]

            color = self.colors[i]

            # Flechas 2.5x grandes, proporcionales al gradiente efectivo
            if abs(gi) > 0.02:
                base_len = 0.40 + 1.30 * min(abs(gi) / 4.0, 1.0)
                arrow_len = 2.5 * base_len * np.sign(gi)

                self.ax.annotate(
                    "",
                    xy=(xi + arrow_len, yi),
                    xytext=(xi, yi),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=2.4,
                        color=color,
                        alpha=alpha,
                        shrinkA=7,
                        shrinkB=0,
                        mutation_scale=16
                    )
                )

            self.ax.scatter(
                xi, yi,
                s=150,
                color=color,
                alpha=alpha,
                edgecolor=color,
                linewidth=1.0,
                zorder=3
            )

            self.ax.text(
                xi + 0.35,
                yi + 0.00022,
                f"x{i+1}",
                fontsize=11,
                color=color,
                alpha=alpha,
                weight="bold"
            )

        # Piedra visual durante el impacto: arco parabólico + estela + ondas
        if self.mode == "stone":
            u = min(1.0, self.stone_t / self.stone_frames)
            smooth = u * u * (3 - 2 * u)

            # Punto visual de impacto: centro promedio del sistema actual
            target_x = float(np.mean(self.x))
            target_y = float(np.mean(y))
            start_x, start_y = 32.0, 0.0050

            rock_x = (1 - smooth) * start_x + smooth * target_x
            arc = 0.0042 * math.sin(math.pi * u)
            rock_y = (1 - smooth) * start_y + smooth * target_y + arc

            # Estela
            for k in range(1, 6):
                uk = max(0.0, u - 0.035 * k)
                sk = uk * uk * (3 - 2 * uk)
                tx = (1 - sk) * start_x + sk * target_x
                ty = (1 - sk) * start_y + sk * target_y + 0.0042 * math.sin(math.pi * uk)
                self.ax.scatter(
                    tx, ty,
                    s=max(20, 130 - 18 * k),
                    color="dimgray",
                    alpha=max(0.05, 0.22 - 0.03 * k),
                    zorder=4
                )

            # Piedra principal
            self.ax.scatter(
                rock_x, rock_y,
                s=420,
                color="#4B4B4B",
                alpha=0.92,
                marker="o",
                edgecolor="black",
                linewidth=1.0,
                zorder=6
            )

            # Brillo/volumen
            self.ax.scatter(
                rock_x - 0.18, rock_y + 0.00018,
                s=80,
                color="#8A8A8A",
                alpha=0.7,
                zorder=7
            )

            # Ondas de impacto hacia el final
            if u > 0.70:
                ring_alpha = (u - 0.70) / 0.30
                for rr in [0.35, 0.70, 1.05]:
                    self.ax.scatter(
                        target_x, target_y,
                        s=700 * rr * ring_alpha,
                        facecolors="none",
                        edgecolors="dimgray",
                        alpha=max(0.0, 0.35 * (1 - ring_alpha)),
                        linewidth=1.2,
                        zorder=3
                    )

            self.ax.text(
                rock_x + 0.45, rock_y + 0.00035,
                "piedra",
                fontsize=10,
                color="dimgray",
                weight="bold"
            )

        # Mensaje de óptimo
        if self.mode == "done":
            self.ax.text(
                0.5,
                0.52,
                "ÓPTIMO CONSEGUIDO\nKKT satisfecho",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                fontsize=22,
                weight="bold",
                color="#104E8B",
                bbox=dict(
                    boxstyle="round,pad=0.55",
                    facecolor="#EAF5FF",
                    edgecolor="#5DADE2",
                    linewidth=2,
                    alpha=0.95
                )
            )

        self.ax.set_xlim(LOWER - 2, UPPER + 2)
        self.ax.set_ylim(-0.0065, 0.0065)

        self.ax.set_xlabel("Asignación de cada variable")
        self.ax.set_ylabel("Utilidad marginal neta: Δf - ν·0.001")

        self.ax.set_title(
            f"Juego KKT interactivo     |     Piedras: {self.stones}     |     Iteraciones: {self.iterations}",
            fontsize=14,
            pad=12
        )

        self.ax.grid(True, alpha=0.25)

        # Recuadro centrado de suma y distancia
        state = f"sum(x) = {self.x.sum():.2f}     |     max |x-x*| = {np.max(np.abs(self.x - X_OPT)):.4f}"
        self.ax.text(
            0.5,
            0.94,
            state,
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#F4F4F4",
                edgecolor="#CCCCCC",
                alpha=0.95
            )
        )

    def draw_bar(self):
        self.bar_ax.clear()

        p = progress_value(self.x)
        fx = f(self.x)

        self.bar_ax.set_xlim(0, 1)
        self.bar_ax.set_ylim(0, 1)
        self.bar_ax.axis("off")

        # Contenedor
        self.bar_ax.add_patch(
            Rectangle(
                (0.22, 0.03),
                0.56,
                0.92,
                fill=False,
                edgecolor="#1F4E79",
                linewidth=2
            )
        )

        # Agua
        self.bar_ax.add_patch(
            Rectangle(
                (0.25, 0.03),
                0.50,
                0.92 * p,
                color="#1E88E5",
                alpha=0.82
            )
        )

        # Onda
        if p > 0:
            y0 = 0.03 + 0.92 * p
            xs = np.linspace(0.25, 0.75, 80)
            ys = y0 + 0.012 * np.sin(np.linspace(0, 2 * np.pi, len(xs)))
            self.bar_ax.plot(xs, ys, color="#90CAF9", lw=2)

        self.bar_ax.text(
            0.5,
            0.50,
            f"f={fx:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="black"
        )

        self.bar_ax.text(
            0.5,
            0.99,
            "f(x)",
            ha="center",
            va="top",
            fontsize=11
        )

        self.bar_ax.text(
            0.5,
            -0.01,
            f"{100 * p:.0f}%",
            ha="center",
            va="top",
            fontsize=10
        )

    def draw_metrics(self):
        fx = f(self.x)
        gap = F_OPT - fx
        sum_error = self.x.sum() - TOTAL
        residual = kkt_residual(self.x)

        self.metrics_text.set_text(
            "Datos KKT:  "
            f"gap objetivo f*−f(x) = {gap:.4f}     |     "
            f"violación suma = {sum_error:+.4f}     |     "
            f"residuo KKT máx. = {residual:.4f}"
        )

    # --------------------------------------------------------
    # Loop principal
    # --------------------------------------------------------

    def update(self, frame):
        if self.mode == "stone":
            self.step_stone()
        elif self.mode == "repair":
            self.step_repair()
        elif self.mode == "optimize":
            self.step_optimize()
        elif self.mode == "done":
            pass

        self.draw_main_panel()
        self.draw_bar()
        self.draw_metrics()

        return []


if __name__ == "__main__":
    print("Juego KKT iniciado.")
    print(f"Mínimo factible: x_min={X_MIN}, f_min={F_MIN:.4f}")
    print(f"Óptimo:          x_opt={X_OPT}, f_opt={F_OPT:.4f}")
    print("Usa el botón 'Lanzar piedra' para sacar el sistema de la senda.")
    game = KKTStoneGame()
    plt.show()
