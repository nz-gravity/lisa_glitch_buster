"""
Make a FRED animation gif.

Install https://github.com/3b1b/manim

manimgl -lo make_fred_anim.py FREDPulseAnimation
"""

from functools import partial

import numpy as np
from manimlib import (
    Scene,
    Axes,
    Write,
    UP,
    DOWN,
    RIGHT,
    TexText,
    BLUE,
    RED,
    FadeOut,
    Dot,
    GREEN,
    DashedLine,
)

from lisa_glitch_buster.backend.model.fred_pulse import FRED_pulse

T0, T1 = 0, 100
TIMES = (T0, T1, 0.5)
T0_RANGE = (0, 50.0, 1)
SCALE_RANGE = (0, 10, 1)
TAU_RANGE = (0.1, 20.0, 99)
XI_RANGE = (0.1, 10.0, 91)


def delta_txt(start):
    return f"$\\Delta = {start:.1f}$"


def scale_txt(scale):
    return f"$A = {scale:.1f}$"


def tau_txt(tau):
    return f"$\\tau = {tau:.1f}$"


def xi_txt(xi):
    return f"$\\xi = {xi:.1f}$"


def get_yoyo_values(current, start, end, n=50):
    return np.concatenate(
        (
            np.linspace(current, start, n),
            np.linspace(start, end, n),
            np.linspace(end, current, n),
        )
    )


class FREDPulseAnimation(Scene):
    WAITTIME = 0.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.axes = Axes(
            x_range=(T0, T1, 10),
            y_range=SCALE_RANGE,
            axis_config={"color": BLUE},
            x_axis_config={"numbers_to_include": np.arange(0, 11, 2)},
            y_axis_config={"numbers_to_include": np.arange(0, 1.2, 0.2)},
        )
        self.start, self.scale, self.tau, self.xi = 30, 5, 10, 5
        self.graph = self.get_graph()

    def get_graph(self):
        pulse = partial(
            FRED_pulse,
            start=self.start,
            scale=self.scale,
            tau=self.tau,
            xi=self.xi,
        )
        kwgs = dict(function=pulse, x_range=TIMES, color=RED)
        return self.axes.get_graph(**kwgs)

    def animate_graph(self):
        return self.graph.animate.become(self.get_graph())

    def animate_dot(self, d, x, y):
        return d.animate.move_to(self.axes.coords_to_point(x, y))

    def construct(self):

        eqn = TexText(
            r"$S(t|A,\Delta,\tau,\xi) = A \exp \left[ - \xi \left(\frac{t - \Delta}{\tau} + \frac{\tau}{t-\Delta}  \right) -2  \right]$"
        )
        eqn.to_edge(UP)

        # write label on LHS of axis
        axes_labels = self.axes.get_axis_labels(
            x_label_tex="t", y_label_tex=""
        )
        self.play(
            Write(eqn),
            Write(self.axes),
            Write(axes_labels),
            Write(self.graph),
            run_time=1,
        )

        self.wait(self.WAITTIME)
        self.play_t0_anim()

        self.wait(self.WAITTIME)
        self.play_scale_anim()

        self.wait(self.WAITTIME)
        self.play_tau_anim()

        self.wait(self.WAITTIME)
        self.play_xi_anim()

        self.wait()

    def play_t0_anim(self):
        # Start point + label

        def get_txt():
            return TexText(delta_txt(self.start)).next_to(start_dot, DOWN)

        start_dot = Dot(self.axes.coords_to_point(self.start, 0), color=GREEN)
        start_label = get_txt()

        self.play(Write(start_dot), Write(start_label), run_time=self.WAITTIME)

        for self.start in get_yoyo_values(
            self.start, T0_RANGE[0], T0_RANGE[1]
        ):
            self.play(
                self.animate_graph(),
                self.animate_dot(start_dot, self.start, 0),
                start_label.animate.become(get_txt()),
                run_time=0.01,
            )
        self.wait(self.WAITTIME)

        # fade out start point + label
        self.play(FadeOut(start_dot), FadeOut(start_label), run_time=0.5)

    def play_scale_anim(self):
        # Start point + label

        scale_dot = Dot(self.axes.coords_to_point(0, self.scale), color=GREEN)

        def get_txt():
            return TexText(scale_txt(self.scale)).next_to(scale_dot, RIGHT)

        scale_label = get_txt()

        self.play(Write(scale_label), Write(scale_dot), run_time=self.WAITTIME)

        for self.scale in get_yoyo_values(
            self.scale, SCALE_RANGE[0] + 0.5, SCALE_RANGE[1] - 0.5
        ):
            self.play(
                self.animate_graph(),
                self.animate_dot(scale_dot, 0, self.scale),
                scale_label.animate.become(get_txt()),
                run_time=0.01,
            )
        self.wait(self.WAITTIME)

        # fade out start point + label
        self.play(FadeOut(scale_dot), FadeOut(scale_label), run_time=0.5)

    def play_tau_anim(self):

        y = self.scale
        tau_dot1 = Dot(self.axes.coords_to_point(self.start, y), color=GREEN)
        tau_dot2 = Dot(
            self.axes.coords_to_point(self.start + self.tau, y), color=GREEN
        )

        def get_txt():
            return TexText(tau_txt(self.tau)).next_to(tau_dot1, UP)

        tau_label = get_txt()

        start_dot = Dot(self.axes.coords_to_point(self.start, 0), color=GREEN)
        start_label = TexText(delta_txt(self.start)).next_to(start_dot, DOWN)
        # dashed line from y=0 to y=scale
        dashed_line = DashedLine(
            self.axes.coords_to_point(self.start, 0),
            self.axes.coords_to_point(self.start, y),
            color=GREEN,
        )

        self.play(
            Write(tau_dot1),
            Write(tau_dot2),
            Write(tau_label),
            Write(start_dot),
            Write(start_label),
            Write(dashed_line),
            run_time=self.WAITTIME,
        )

        def rng():
            return [self.start, self.start + self.tau]

        tau_graph = self.axes.get_graph(
            lambda x: self.scale, x_range=rng(), color=GREEN
        )

        def animate_tau():
            return tau_graph.animate.become(
                self.axes.get_graph(
                    lambda x: self.scale, x_range=rng(), color=GREEN
                )
            )

        taus = get_yoyo_values(self.tau, TAU_RANGE[0], TAU_RANGE[1])
        for new_tau in taus:
            self.tau = new_tau
            self.play(
                animate_tau(),
                self.animate_graph(),
                self.animate_dot(tau_dot1, self.start, y),
                self.animate_dot(tau_dot2, self.start + self.tau, y),
                tau_label.animate.become(get_txt()),
                run_time=0.01,
            )

        self.wait(self.WAITTIME)

        self.play(
            FadeOut(tau_dot1),
            FadeOut(tau_dot2),
            FadeOut(tau_graph),
            FadeOut(tau_label),
            FadeOut(start_dot),
            FadeOut(start_label),
            FadeOut(dashed_line),
            run_time=0.5,
        )

    def play_xi_anim(self):
        def get_txt():
            return TexText(xi_txt(self.xi)).next_to(
                self.axes.coords_to_point(self.tau + self.start, self.scale),
                UP + RIGHT,
            )

        xi_lbl = get_txt()

        self.play(Write(xi_lbl), run_time=self.WAITTIME)

        for i, self.xi in enumerate(
            get_yoyo_values(self.xi, XI_RANGE[0], XI_RANGE[1])
        ):
            self.play(
                self.animate_graph(),
                xi_lbl.animate.become(get_txt()),
                run_time=0.01,
            )

        self.wait(self.WAITTIME)
        self.play(FadeOut(xi_lbl), run_time=0.5)
