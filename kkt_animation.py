# -*- coding: utf-8 -*-
"""
Interactive KKT game: bombs, repair to the polyhedron, and convergence to the optimum.

Run:
    python kkt_animation.py

Requirements:
    pip install numpy matplotlib

If you are in Jupyter:
    %matplotlib qt
    %run kkt_animation.py
"""

import itertools
import math
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Button


# ============================================================
# 1. Problem
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

# Optimum constructed for the maximization problem
X_OPT = np.array([5, 5, 12, 18, 30, 30], dtype=float)
NU_OPT = 1.0

# Initial point: all variables equal
X0 = np.repeat(TOTAL / N, N)

H = 0.001


def f(x):
    """Objective function: max b'x - 1/2 x'Qx."""
    x = np.asarray(x, dtype=float)
    return float(b @ x - 0.5 * x @ Q @ x)


def grad_f(x):
    """Gradient of f."""
    x = np.asarray(x, dtype=float)
    return b - Q @ x


def project_box_sum(y, lower=LOWER, upper=UPPER, total=TOTAL, max_iter=100):
    """
    Euclidean projection onto:
        {x: lower <= x_i <= upper, sum(x)=total}

    Uses binary search over the sum multiplier.
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
    Chart y-axis:
        Δf_i - NU_OPT * 0.001

    It is a finite difference from increasing x_i by 0.001,
    net of the shadow price of the sum constraint.
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
    Polyhedron vertices:
        5 <= x_i <= 30, sum_i x_i = 100

    In dimension 6, a vertex has 5 variables at bounds
    and one free variable determined by the sum.
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

    # For robustness: also check all full-bound points
    for bounds in itertools.product([LOWER, UPPER], repeat=N):
        x = np.array(bounds, dtype=float)
        if abs(x.sum() - TOTAL) < 1e-10:
            candidates.append(x)

    return np.unique(np.array(candidates), axis=0)


def find_feasible_minimum():
    """
    Since f is concave, the minimum over a compact polyhedron
    occurs at a vertex.
    """
    vertices = enumerate_vertices_box_sum()
    vals = np.array([f(v) for v in vertices])
    idx = int(np.argmin(vals))
    return vertices[idx], float(vals[idx])


X_MIN, F_MIN = find_feasible_minimum()
F_OPT = f(X_OPT)


def progress_value(x):
    """
    Normalization of f:
        0% = feasible minimum of the problem
        100% = maximization optimum
    """
    val = (f(x) - F_MIN) / (F_OPT - F_MIN)
    return float(np.clip(val, 0.0, 1.0))


def alpha_from_value(x):
    """
    Transparency:
        f = minimum -> 80% transparent -> alpha = 0.20
        f = optimum -> 0% transparent -> alpha = 1.00
    """
    p = progress_value(x)
    return 0.20 + 0.80 * p


def kkt_residual(x, eps=1e-4):
    """
    Illustrative KKT residual using the optimal multiplier NU_OPT = 1.

    At the optimum:
        if x_i = lower,  grad_i - nu <= 0
        if interior,     grad_i - nu = 0
        if x_i = upper,  grad_i - nu >= 0
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
    Effective gradient used ONLY to draw arrows.

    At the optimum, variables at bounds must keep showing arrows:
      x1,x2 at minimum: arrows to the left
      x5,x6 at maximum: arrows to the right
      x3,x4 interior: no arrow

    This prevents numerical effects or 'done' states from removing arrows
    from active constraints.
    """
    x = np.asarray(x, dtype=float)

    if np.max(np.abs(x - X_OPT)) < eps:
        return np.array([-3.0, -1.0, 0.0, 0.0, 2.0, 4.0])

    return grad_f(x) - NU_OPT


# ============================================================
# 2. Sounds
# ============================================================

SOUND_VOLUME = 0.50  # 50% amplitude


def _play_tone_sequence(seq, volume=SOUND_VOLUME, sample_rate=22050):
    """
    Play a sequence of tones with amplitude-controlled volume.

    On Windows this uses winsound.PlaySound on a generated temporary WAV.
    On other systems it falls back to the terminal bell; in that case the actual
    volume depends on the system settings.
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
                # Simple envelope to avoid clicks
                env = min(1.0, k / max(1, int(0.015 * sample_rate)))
                env *= min(1.0, (n_samples - k) / max(1, int(0.015 * sample_rate)))
                amp = int(32767 * volume * 0.35 * env)
                val = int(amp * _math.sin(2 * _math.pi * freq * t))
                samples.append(val)

            # Short silence between tones
            samples.extend([0] * int(sample_rate * 0.025))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(struct.pack("<h", s) for s in samples))

        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

        # Deferred cleanup so Windows can read the WAV before it is removed
        def _cleanup():
            time.sleep(2)
            try:
                os.remove(wav_path)
            except OSError:
                pass

        threading.Thread(target=_cleanup, daemon=True).start()

    except Exception:
        # Simple fallback; terminal bell volume is not controllable here.
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
# 3. Interactive game
# ============================================================

class KKTStoneGame:
    def __init__(self):
        self.rng = np.random.default_rng()

        # State
        self.x = X0.copy()
        self.mode = "optimize"  # optimize, stone, repair, done
        self.iterations = 0
        self.stones = 0
        self.paused = False
        self.info_open = False

        self.reset_allowed = False
        self.victory_played = False

        # Visual / dynamic parameters
        self.interval_ms = 2      # velocidad del loop/iteraciones
        # Bomb duration is controlled by wall-clock time, not by frame count.
        # This keeps the bomb flight at 0.25 real seconds even if interval_ms changes.
        self.stone_seconds = 0.25
        self.stone_frames = None  # Intentionally unused; kept only to document the old approach.
        self.stone_start_time = None

        # No momentum: simple projected step.
        # Far away it moves faster; close to the optimum, slower.
        self.eta_min = 0.075
        self.eta_max = 6

        # Maximum allowed movement:
        # 0.1 per variable per iteration = 1 unit every 10 iterations.
        self.max_step_per_iteration = 0.10

        self.landing_threshold = 1.00
        self.landing_steps_total = 10
        self.landing_remaining = 0
        self.landing_start = None

        self.repair_rate = 0.10  # smoothness when returning to sum(x)=100
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
        self.update_info_button()

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
        self.fig.canvas.manager.set_window_title("KKT game with bombs")

        self.ax = self.fig.add_axes([0.08, 0.25, 0.70, 0.65])
        self.bar_ax = self.fig.add_axes([0.84, 0.25, 0.08, 0.65])

        self.ax_stone = self.fig.add_axes([0.08, 0.07, 0.20, 0.075])

        # Small controls: circular icons, side by side.
        self.ax_reset = self.fig.add_axes([0.315, 0.082, 0.045, 0.045])
        self.ax_pause = self.fig.add_axes([0.372, 0.082, 0.045, 0.045])
        self.ax_info = self.fig.add_axes([0.429, 0.082, 0.045, 0.045])

        self.btn_stone = Button(self.ax_stone, "Drop bomb")
        self.btn_reset = Button(self.ax_reset, "↻")
        self.btn_pause = Button(self.ax_pause, "⏸")
        self.btn_info = Button(self.ax_info, "ℹ")

        self.btn_stone.on_clicked(self.throw_stone)
        self.btn_reset.on_clicked(self.reset_game)
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_info.on_clicked(self.toggle_info_panel)

        self.btn_reset.label.set_fontsize(15)
        self.btn_pause.label.set_fontsize(15)
        self.btn_info.label.set_fontsize(15)

        self._style_round_icon_buttons()

        self.metrics_text = self.fig.text(
            0.08, 0.17, "",
            fontsize=11,
            ha="left",
            va="center"
        )

        self.info_text = self.fig.text(
            0.55, 0.08,
            f"Feasible minimum: f={F_MIN:.2f} at {np.round(X_MIN, 2)}     |     Optimum: f={F_OPT:.2f}",
            fontsize=10,
            ha="left",
            va="center",
            color="0.25"
        )

        # Collapsible objective/parameter panel.
        # Purely informational: it does not change timing, optimization steps,
        # bomb dynamics, or any animation state transition.
        self.objective_panel = self.fig.text(
            0.08, 0.965, "",
            fontsize=9.5,
            ha="left",
            va="top",
            family="monospace",
            color="#222222",
            visible=False,
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor="#FBFBFB",
                edgecolor="#777777",
                linewidth=1.2,
                alpha=0.97
            )
        )

    def _style_round_icon_buttons(self):
        """
        Style restart and pause as small circular buttons.
        The clickable area is still the square axes, but the rectangular box is hidden.
        """
        self._round_button_axes = [self.ax_reset, self.ax_pause, self.ax_info]
        self._round_button_circles = {}

        for ax, face in [
            (self.ax_reset, "#EAF6EA"),
            (self.ax_pause, "#EEEEEE"),
            (self.ax_info, "#EAF2FF"),
        ]:
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("none")

            for spine in ax.spines.values():
                spine.set_visible(False)

            # Hide the widget's rectangular box.
            ax.patch.set_alpha(0)

            circle = Circle(
                (0.5, 0.5),
                0.46,
                transform=ax.transAxes,
                facecolor=face,
                edgecolor="#777777",
                linewidth=1.2,
                zorder=-1
            )
            ax.add_patch(circle)
            self._round_button_circles[ax] = circle

    def _set_round_button_face(self, ax, facecolor):
        if hasattr(self, "_round_button_circles") and ax in self._round_button_circles:
            self._round_button_circles[ax].set_facecolor(facecolor)
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)

    def update_reset_button(self):
        self.btn_reset.label.set_text("↻")
        self._set_round_button_face(self.ax_reset, "#EAF6EA")
        self.fig.canvas.draw_idle()

    def update_pause_button(self):
        if self.paused:
            self.btn_pause.label.set_text("▶")
            self._set_round_button_face(self.ax_pause, "#FFF2CC")
        else:
            self.btn_pause.label.set_text("⏸")
            self._set_round_button_face(self.ax_pause, "#EEEEEE")

        self.fig.canvas.draw_idle()

    def toggle_pause(self, event=None):
        self.paused = not self.paused
        self.update_pause_button()

    def objective_panel_text(self):
        """Return the text shown in the collapsible objective/parameter panel."""
        q_text = np.array2string(Q, precision=3, suppress_small=True)
        b_text = np.array2string(b, precision=3, suppress_small=True)
        xopt_text = np.array2string(X_OPT, precision=3, suppress_small=True)

        return (
            "Objective function:\n"
            "    maximize  f(x) = b'x - 0.5 x'Qx\n\n"
            "Constraints:\n"
            f"    {LOWER:.0f} <= x_i <= {UPPER:.0f}\n"
            f"    sum_i x_i = {TOTAL:.0f}\n\n"
            "Parameters:\n"
            f"    b = {b_text}\n"
            f"    Q =\n{q_text}\n"
            f"    x* = {xopt_text}\n"
            f"    nu* = {NU_OPT:.3f}\n\n"
            "Animation settings:\n"
            f"    interval_ms = {self.interval_ms}\n"
            f"    bomb duration = {self.stone_seconds:.2f}s, measured with wall-clock time\n"
            f"    max step per iteration = {self.max_step_per_iteration:.3f}\n"
            f"    landing threshold = {self.landing_threshold:.2f}\n"
            f"    landing steps = {self.landing_steps_total}\n"
        )

    def update_info_button(self):
        """Update the info icon color according to whether the panel is open."""
        if self.info_open:
            self._set_round_button_face(self.ax_info, "#CFE3FF")
        else:
            self._set_round_button_face(self.ax_info, "#EAF2FF")

        self.fig.canvas.draw_idle()

    def toggle_info_panel(self, event=None):
        """Show/hide the objective and parameter panel."""
        self.info_open = not self.info_open
        self.objective_panel.set_text(self.objective_panel_text())
        self.objective_panel.set_visible(self.info_open)
        self.update_info_button()

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    def throw_stone(self, event=None):
        """
        The bomb sends the state to a random point in the cube [5,30]^6.
        That point does NOT have to satisfy sum(x)=100.
        """
        self.stones += 1
        self.mode = "stone"
        self.victory_played = False
        self.reset_allowed = False
        self.update_reset_button()

        self.landing_remaining = 0
        self.landing_start = None

        self.stone_t = 0  # Kept only as a frame counter/debug counter.
        self.stone_start_time = time.perf_counter()
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
    # Dynamics
    # --------------------------------------------------------

    def bomb_progress(self):
        """
        Return bomb flight progress in [0, 1] using real wall-clock time.

        Why not use stone_frames? Because stone_frames depends on interval_ms.
        If interval_ms is reduced to make the optimization iterate faster, the
        number of bomb frames increases. Matplotlib may fail to render those
        frames at the requested speed, making the bomb look slower. Wall-clock
        timing avoids that: stone_seconds is the real duration.
        """
        if self.stone_start_time is None:
            return 0.0

        elapsed = time.perf_counter() - self.stone_start_time
        return float(np.clip(elapsed / self.stone_seconds, 0.0, 1.0))

    def step_stone(self):
        """
        The bomb flies for 0.25 real seconds. The points do NOT move during the flight.
        On impact:
        - an explosion sound plays,
        - a visual effect is triggered,
        - the system jumps to the random point in [5,30]^6,
        - then repair toward sum(x)=100 begins.
        """
        self.stone_t += 1
        u = self.bomb_progress()

        # During flight, keep the original state.
        self.x = self.stone_start.copy()

        if u >= 1.0:
            self.x = self.stone_target.copy()
            self.repair_target = project_box_sum(self.x)
            self.explosion_remaining = self.explosion_frames_total
            play_impact_sound()
            self.mode = "repair"

    def step_repair(self):
        """
        Smooth transition to the constraint polyhedron.
        It respects the box because both endpoints are in [5,30]^6.
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
        Projected ascent on f WITHOUT momentum.

        - Far away: larger step.
        - Close: smaller step.
        - If max |x-x*| < 1.0, it enters slow landing.
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
    # Drawing
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

        # Arrows and points
        for i in range(N):
            xi = self.x[i]
            yi = y[i]
            gi = g_eff[i]

            color = self.colors[i]

            # Arrows are 2.5x large, proportional to the effective gradient.
            # Important rendering debug:
            # At the optimum, active bound variables must NOT lose their arrows.
            # Only interior variables with near-zero effective gradient may have no arrow.
            is_boundary = (xi <= LOWER + 1e-7) or (xi >= UPPER - 1e-7)
            draw_this_arrow = (abs(gi) > 0.02) or (self.mode == "done" and is_boundary)

            if draw_this_arrow:
                # If for any reason gi is almost zero but the variable is at a bound,
                # give it the correct KKT direction.
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

        # Bomb visual: 0.25s flight. Points stay still until impact.
        if self.mode == "stone":
            u = self.bomb_progress()
            smooth = u * u * (3 - 2 * u)

            target_x = float(np.mean(self.stone_start))
            target_y = float(np.mean(net_marginal_delta(self.stone_start)))
            start_x, start_y = 32.0, 0.0050

            rock_x = (1 - smooth) * start_x + smooth * target_x
            arc = 0.0048 * math.sin(math.pi * u)
            rock_y = (1 - smooth) * start_y + smooth * target_y + arc

            # Trail
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

            # Bomb principal
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

            # Highlight
            self.ax.scatter(
                rock_x - 0.18, rock_y + 0.00020,
                s=90,
                color="#9A9A9A",
                alpha=0.75,
                zorder=7
            )

            self.ax.text(
                rock_x + 0.45, rock_y + 0.00035,
                "bomb",
                fontsize=10,
                color="dimgray",
                weight="bold"
            )

        # Visual explosion right after impact
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

        # Optimum message
        if self.mode == "done":
            self.ax.text(
                0.5,
                0.52,
                "OPTIMUM REACHED\nKKT satisfied",
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

        self.ax.set_xlabel("Allocation of each variable")
        self.ax.set_ylabel("Net marginal utility: Δf - ν·0.001")

        self.ax.set_title(
            f"Interactive KKT game     |     Bombs: {self.stones}     |     Iterations: {self.iterations}",
            fontsize=14,
            pad=12
        )

        self.ax.grid(True, alpha=0.25)

        # Centered box with sum and distance.
        # ONLY red element when sum(x)=100 is violated.
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

        # Container
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

        # Water
        self.bar_ax.add_patch(
            Rectangle(
                (0.25, 0.03),
                0.50,
                0.92 * p,
                color="#1E88E5",
                alpha=0.82
            )
        )

        # Wave
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
            "KKT data:  "
            f"objective gap f*−f(x) = {gap:.4f}     |     "
            f"sum violation = {sum_error:+.4f}     |     "
            f"max KKT residual = {residual:.4f}"
        )

    # --------------------------------------------------------
    # Main loop
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
    print("KKT game started, without momentum.")
    print(f"Feasible minimum: x_min={X_MIN}, f_min={F_MIN:.4f}")
    print(f"Optimum:          x_opt={X_OPT}, f_opt={F_OPT:.4f}")
    print(f"Arrows at optimum grad_f(x*)-nu = {arrow_gradient(X_OPT)}")
    print("Use the 'Drop bomb' button to knock the system off path.")
    game = KKTStoneGame()
    plt.show()
