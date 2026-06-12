# -*- coding: utf-8 -*-
"""
Juego interactivo KKT: bombas, reparación al poliedro y convergencia al óptimo.

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


def arrow_gradient(x, eps=1e-8):
    """
    Gradiente efectivo usado SOLO para dibujar flechas.

    En el óptimo, las variables en cota deben seguir mostrando flecha:
      x1,x2 en mínimo: flechas a la izquierda
      x5,x6 en máximo: flechas a la derecha
      x3,x4 interiores: sin flecha

    Esto evita que efectos numéricos o estados 'done' borren las flechas
    de las restricciones activas.
    """
    x = np.asarray(x, dtype=float)

    if np.max(np.abs(x - X_OPT)) < eps:
        return np.array([-3.0, -1.0, 0.0, 0.0, 2.0, 4.0])

    return grad_f(x) - NU_OPT


# ============================================================
# 2. Sonidos
# ============================================================

SOUND_VOLUME = 0.50  # 50% de amplitud


def _play_tone_sequence(seq, volume=SOUND_VOLUME, sample_rate=22050):
    """
    Reproduce una secuencia de tonos con volumen controlado por amplitud.

    En Windows usa winsound.PlaySound sobre un WAV temporal generado.
    En otros sistemas cae a campana terminal; ahí el volumen real depende del sistema.
    """
    try:
        import os
        import math as _math
        import wave
        import tempfile
        import winsound
        import struct

        volume = float(max(0.0, min(1.0, volume)))
        samples = []

        for freq, dur_ms in seq:
            n_samples = int(sample_rate * dur_ms / 1000)
            for k in range(n_samples):
                t = k / sample_rate
                # Envolvente simple para evitar clicks
                env = min(1.0, k / max(1, int(0.015 * sample_rate)))
                env *= min(1.0, (n_samples - k) / max(1, int(0.015 * sample_rate)))
                amp = int(32767 * volume * 0.35 * env)
                val = int(amp * _math.sin(2 * _math.pi * freq * t))
                samples.append(val)

            # mini silencio entre tonos
            samples.extend([0] * int(sample_rate * 0.025))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(struct.pack("<h", s) for s in samples))

        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

        # Limpieza diferida para no borrar el WAV antes de que Windows lo lea
        def _cleanup():
            time.sleep(2)
            try:
                os.remove(wav_path)
            except OSError:
                pass

        threading.Thread(target=_cleanup, daemon=True).start()

    except Exception:
        # Fallback simple; el volumen no es controlable con la campana terminal.
        print("", end="", flush=True)


def play_launch_sound():
    seq = [(360, 45), (430, 55)]
    threading.Thread(target=_play_tone_sequence, args=(seq,), daemon=True).start()


def play_impact_sound():
    seq = [(120, 80), (70, 120), (45, 180)]
    threading.Thread(target=_play_tone_sequence, args=(seq,), daemon=True).start()


def play_optimum_sound():
    seq = [(523, 110), (659, 110), (784, 160), (1046, 220)]
    threading.Thread(target=_play_tone_sequence, args=(seq,), daemon=True).start()


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
        self.paused = False

        self.reset_allowed = False
        self.victory_played = False

        # Parámetros visuales / dinámicos
        self.interval_ms = 30
        self.stone_seconds = 0.25
        self.stone_frames = max(2, int(round(1000 * self.stone_seconds / self.interval_ms)))

        # Sin momentum: paso proyectado simple.
        # Lejos avanza más rápido; cerca, más lento.
        self.eta_min = 0.035
        self.eta_max = 0.45

        # Máximo movimiento permitido:
        # 0.1 por variable por iteración = 1 unidad cada 10 iteraciones.
        self.max_step_per_iteration = 0.10

        self.landing_threshold = 0.50
        self.landing_steps_total = 5
        self.landing_remaining = 0
        self.landing_start = None

        self.repair_rate = 0.10  # suavidad al regresar a sum(x)=100
        self.explosion_frames_total = 8
        self.explosion_remaining = 0
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
        self.update_pause_button()

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
        self.fig.canvas.manager.set_window_title("Juego KKT con bombas")

        self.ax = self.fig.add_axes([0.08, 0.25, 0.70, 0.65])
        self.bar_ax = self.fig.add_axes([0.84, 0.25, 0.08, 0.65])

        self.ax_stone = self.fig.add_axes([0.08, 0.07, 0.20, 0.075])
        self.ax_reset = self.fig.add_axes([0.31, 0.07, 0.20, 0.075])
        self.ax_pause = self.fig.add_axes([0.54, 0.07, 0.20, 0.075])

        self.btn_stone = Button(self.ax_stone, "Lanzar bomba")
        self.btn_reset = Button(self.ax_reset, "↻")
        self.btn_pause = Button(self.ax_pause, "⏸")

        self.btn_stone.on_clicked(self.throw_stone)
        self.btn_reset.on_clicked(self.reset_game)
        self.btn_pause.on_clicked(self.toggle_pause)

        self.btn_reset.label.set_fontsize(18)
        self.btn_pause.label.set_fontsize(18)

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
        self.btn_reset.label.set_text("↻")
        self.ax_reset.set_facecolor("#DFF2E1")
        self.fig.canvas.draw_idle()

    def update_pause_button(self):
        if self.paused:
            self.btn_pause.label.set_text("▶")
            self.ax_pause.set_facecolor("#FFF2CC")
        else:
            self.btn_pause.label.set_text("⏸")
            self.ax_pause.set_facecolor("#EEEEEE")

        self.fig.canvas.draw_idle()

    def toggle_pause(self, event=None):
        self.paused = not self.paused
        self.update_pause_button()

    # --------------------------------------------------------
    # Eventos
    # --------------------------------------------------------

    def throw_stone(self, event=None):
        """
        La bomba manda el estado a un punto aleatorio del cubo [5,30]^6.
        Ese punto NO tiene por qué cumplir sum(x)=100.
        """
        self.stones += 1
        self.mode = "stone"
        self.victory_played = False
        self.reset_allowed = False
        self.update_reset_button()

        self.landing_remaining = 0
        self.landing_start = None

        self.stone_t = 0
        self.stone_start = self.x.copy()
        self.stone_target = self.rng.uniform(LOWER, UPPER, size=N)
        self.stone_origin = np.array([32.0, 0.0050])
        self.explosion_remaining = 0

        play_launch_sound()

    def reset_game(self, event=None):
        if not self.reset_allowed:
            return

        self.x = X0.copy()
        self.mode = "optimize"
        self.iterations = 0
        self.stones = 0
        self.paused = False
        self.update_pause_button()
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
        La bomba vuela durante 0.25s. Los puntos NO se mueven durante el vuelo.
        En el impacto:
        - suena explosión,
        - se activa efecto visual,
        - el sistema salta al punto aleatorio en [5,30]^6,
        - luego empieza la reparación hacia sum(x)=100.
        """
        self.stone_t += 1
        u = min(1.0, self.stone_t / self.stone_frames)

        # Durante el vuelo, mantener el estado original.
        self.x = self.stone_start.copy()

        if u >= 1.0:
            self.x = self.stone_target.copy()
            self.repair_target = project_box_sum(self.x)
            self.explosion_remaining = self.explosion_frames_total
            play_impact_sound()
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
            self.landing_remaining = 0
            self.landing_start = None
            self.mode = "optimize"

    def step_optimize(self):
        """
        Ascenso proyectado de f SIN momentum.

        - Lejos: paso más grande.
        - Cerca: paso más pequeño.
        - Si max |x-x*| < 0.5, entra en aterrizaje lento.
        """
        if self.has_reached_optimum():
            self.reach_optimum()
            return

        max_dist = float(np.max(np.abs(self.x - X_OPT)))

        if max_dist < self.landing_threshold:
            if self.landing_remaining <= 0:
                self.landing_remaining = self.landing_steps_total
                self.landing_start = self.x.copy()

            desired_step = (X_OPT - self.x) / self.landing_remaining
            max_abs_step = float(np.max(np.abs(desired_step)))

            if max_abs_step > self.max_step_per_iteration:
                desired_step *= self.max_step_per_iteration / max_abs_step

            self.x = self.x + desired_step
            self.x = project_box_sum(self.x)

            self.landing_remaining -= 1
            self.iterations += 1

            if self.landing_remaining <= 0 or self.has_reached_optimum():
                self.reach_optimum()
            return

        closeness = np.clip(max_dist / 12.0, 0.0, 1.0)
        eta = self.eta_min + (self.eta_max - self.eta_min) * closeness

        x_candidate = project_box_sum(self.x + eta * grad_f(self.x))

        raw_step = x_candidate - self.x
        max_abs_step = float(np.max(np.abs(raw_step)))

        if max_abs_step > self.max_step_per_iteration:
            raw_step = raw_step * (self.max_step_per_iteration / max_abs_step)
            x_candidate = self.x + raw_step
            x_candidate = project_box_sum(x_candidate)

            raw_step = x_candidate - self.x
            max_abs_step = float(np.max(np.abs(raw_step)))
            if max_abs_step > self.max_step_per_iteration:
                raw_step = raw_step * (self.max_step_per_iteration / max_abs_step)
                x_candidate = self.x + raw_step

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
        g_eff = arrow_gradient(self.x)

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

            # Flechas 2.5x grandes, proporcionales al gradiente efectivo.
            # Debug importante:
            # En el óptimo, las variables activas en cota NO deben perder flecha.
            # Solo las interiores con gradiente efectivo cercano a cero pueden quedar sin flecha.
            is_boundary = (xi <= LOWER + 1e-7) or (xi >= UPPER - 1e-7)
            draw_this_arrow = (abs(gi) > 0.02) or (self.mode == "done" and is_boundary)

            if draw_this_arrow:
                # Si por cualquier razón gi fuese casi cero pero está en cota,
                # darle dirección KKT correcta.
                if abs(gi) <= 0.02 and is_boundary:
                    gi = -1.0 if xi <= LOWER + 1e-7 else 1.0

                base_len = 0.40 + 1.30 * min(abs(gi) / 4.0, 1.0)
                arrow_len = 2.5 * base_len * np.sign(gi)

                self.ax.annotate(
                    "",
                    xy=(xi + arrow_len, yi),
                    xytext=(xi, yi),
                    clip_on=False,
                    zorder=4,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=2.6,
                        color=color,
                        alpha=max(alpha, 0.65),
                        shrinkA=4,
                        shrinkB=0,
                        mutation_scale=18
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

        # Bomba visual: vuelo de 0.25s. Los puntos se quedan quietos hasta el impacto.
        if self.mode == "stone":
            u = min(1.0, self.stone_t / self.stone_frames)
            smooth = u * u * (3 - 2 * u)

            target_x = float(np.mean(self.stone_start))
            target_y = float(np.mean(net_marginal_delta(self.stone_start)))
            start_x, start_y = 32.0, 0.0050

            rock_x = (1 - smooth) * start_x + smooth * target_x
            arc = 0.0048 * math.sin(math.pi * u)
            rock_y = (1 - smooth) * start_y + smooth * target_y + arc

            # Estela
            for k in range(1, 8):
                uk = max(0.0, u - 0.030 * k)
                sk = uk * uk * (3 - 2 * uk)
                tx = (1 - sk) * start_x + sk * target_x
                ty = (1 - sk) * start_y + sk * target_y + 0.0048 * math.sin(math.pi * uk)
                self.ax.scatter(
                    tx, ty,
                    s=max(18, 160 - 18 * k),
                    color="#555555",
                    alpha=max(0.04, 0.26 - 0.03 * k),
                    zorder=4
                )

            # Bomba principal
            self.ax.scatter(
                rock_x, rock_y,
                s=460,
                color="#3F3F3F",
                alpha=0.95,
                marker="o",
                edgecolor="black",
                linewidth=1.1,
                zorder=6
            )

            # Brillo
            self.ax.scatter(
                rock_x - 0.18, rock_y + 0.00020,
                s=90,
                color="#9A9A9A",
                alpha=0.75,
                zorder=7
            )

            self.ax.text(
                rock_x + 0.45, rock_y + 0.00035,
                "bomba",
                fontsize=10,
                color="dimgray",
                weight="bold"
            )

        # Explosión visual justo después del impacto
        if self.explosion_remaining > 0:
            q = 1.0 - self.explosion_remaining / self.explosion_frames_total
            cx = float(np.mean(self.x))
            cy = float(np.mean(y))

            self.ax.scatter(
                cx, cy,
                s=700 * (1 - q) + 120,
                color="#FFB000",
                alpha=max(0.05, 0.75 * (1 - q)),
                zorder=8
            )

            for rr, col in [(0.8, "#FF6D00"), (1.3, "#FFC400"), (1.8, "#777777")]:
                self.ax.scatter(
                    cx, cy,
                    s=1800 * rr * (q + 0.15),
                    facecolors="none",
                    edgecolors=col,
                    alpha=max(0.0, 0.55 * (1 - q)),
                    linewidth=2.0,
                    zorder=7
                )

            angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            for a in angles:
                px = cx + 1.8 * q * math.cos(a)
                py = cy + 0.0022 * q * math.sin(a)
                self.ax.scatter(
                    px, py,
                    s=45,
                    color="#5A5A5A",
                    alpha=max(0.0, 0.75 * (1 - q)),
                    zorder=9
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

        self.ax.set_xlim(LOWER - 6, UPPER + 6)
        self.ax.set_ylim(-0.0065, 0.0065)

        self.ax.set_xlabel("Asignación de cada variable")
        self.ax.set_ylabel("Utilidad marginal neta: Δf - ν·0.001")

        self.ax.set_title(
            f"Juego KKT interactivo     |     Bombas: {self.stones}     |     Iteraciones: {self.iterations}",
            fontsize=14,
            pad=12
        )

        self.ax.grid(True, alpha=0.25)

        # Recuadro centrado de suma y distancia.
        # ÚNICO elemento rojo cuando se viola sum(x)=100.
        sum_error = float(self.x.sum() - TOTAL)
        sum_violated = abs(sum_error) > 1e-3

        state = f"sum(x) = {self.x.sum():.2f}     |     max |x-x*| = {np.max(np.abs(self.x - X_OPT)):.4f}"
        self.ax.text(
            0.5,
            0.94,
            state,
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="white" if sum_violated else "black",
            weight="bold" if sum_violated else "normal",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#B00020" if sum_violated else "#F4F4F4",
                edgecolor="#700000" if sum_violated else "#CCCCCC",
                linewidth=1.4 if sum_violated else 1.0,
                alpha=0.96 if sum_violated else 0.95
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
        if not self.paused:
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

        if not self.paused and self.explosion_remaining > 0:
            self.explosion_remaining -= 1

        return []


if __name__ == "__main__":
    print("Juego KKT iniciado, sin momentum.")
    print(f"Mínimo factible: x_min={X_MIN}, f_min={F_MIN:.4f}")
    print(f"Óptimo:          x_opt={X_OPT}, f_opt={F_OPT:.4f}")
    print(f"Flechas en óptimo grad_f(x*)-nu = {arrow_gradient(X_OPT)}")
    print("Usa el botón 'Lanzar bomba' para sacar el sistema de la senda.")
    game = KKTStoneGame()
    plt.show()
