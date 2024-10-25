from manim import *
from math import exp, e
import numpy as np
from scipy.ndimage import gaussian_filter1d  # for smoothing

def LIF(tau=10, t0=20, t1=40, t2=60, w=0.1, threshold=1.0, reset=0.0, smoothing=False):
    # Spike times, keep sorted because it's more efficient to pop the last value off the list
    times = [t0, t1, t2]
    times.sort(reverse=True)
    
    # set some default parameters
    duration = 100  # total time in ms
    dt = 0.1  # timestep in ms
    alpha = np.exp(-dt/tau)  # decay factor per time step
    
    V_rec = []  # list to record membrane potentials
    T_rec = []  # list to record time points
    V = 0.0  # initial membrane potential
    
    T = np.arange(0, duration, dt)  # array of time points
    spikes = []  # list to store spike times

    # run the simulation
    for t in T:
        T_rec.append(t)  # record time
        V_rec.append(V)  # record voltage before update

        # Integrate equations
        V *= alpha
        
        # Check if there is an input spike
        if times and t > times[-1]:
            V += w
            times.pop()  # remove spike time after processing
        
        # If the potential exceeds the threshold, reset
        if V > threshold:
            V = reset
            spikes.append(t)

    # Smoothing the graph
    if smoothing:
        V_rec = gaussian_filter1d(V_rec, sigma=5)

    # Return the lists of time and voltage values
    return T_rec, V_rec

class Main(Scene):
    def construct(self):
        # ==== explain neuron ====
        neuron = ImageMobject("neuron.png").scale(0.5)

        self.play(FadeIn(neuron))
        self.wait(1)
        self.play(neuron.animate.to_edge(LEFT).scale(0.5))
        desc = Paragraph("- your brain has many neurons""- neurons sends signals to other neurons in a complex network").scale(0.5).next_to(neuron, RIGHT)

        self.play(Write(desc))
        self.wait(2)

        self.play(FadeOut(neuron), FadeOut(desc))


        # ====== introduce SNN =====
        intr =  Paragraph("To fully understand the brain,\nwe have to begin by better understanding the brain's neural signalling\ninside its neural networks", font="Arial").scale(0.5)
        self.play(Write(intr))
        self.wait(2)
        self.play(FadeOut(intr))


        # ==== SNN definition ====

        word = Text("Spiking Neural Networks (SNN)", font="Arial").scale(1.5)
        definition = Paragraph("Are made to simulate the kinds of networks the neurons in your brain form", "act as a model to provide insight into how our brain works", font="Arial").scale(0.5)

        self.play(Write(word))

        self.play(
            word.animate.shift(UP * 2),
            run_time=2
        )
        self.play(Write(definition))
        self.wait(2)
        self.play(FadeOut(word), FadeOut(definition))


        diff_eq = Paragraph("SNNs are defined by differential equations", font="Arial").scale(0.5)
        self.play(Write(diff_eq))
        self.play(diff_eq.animate.shift(UP * 2))

        eq1 = MathTex(r"\tau \frac{dV}{dt} = -V ").next_to(diff_eq, DOWN) 
        eq2 = MathTex(r"V \leftarrow V + w").next_to(eq1, DOWN*2)
        self.play(Write(eq1))
        self.wait(5)
        self.play(Write(eq2))
        self.wait(5)
        self.play(FadeOut(diff_eq))

        self.play(eq1.animate.shift(UP * 2).scale(0.8))
        self.play(eq1.animate.shift(LEFT), eq2.animate.shift(UP * 2).scale(0.8).next_to(eq1, RIGHT))


        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1, 0.1],
            axis_config={"include_tip": False, "numbers_to_exclude": [0]},
        )

        labels = VGroup(
            axes.get_x_axis_label("Time (ms)", edge=DOWN, direction=DOWN),
            axes.get_y_axis_label("Membrane Potential (V)")
        )

        graph = VGroup(
            axes,
            labels
        ).next_to(eq1, DOWN).scale(0.7)

        self.play(Create(graph))
        self.wait(2)

        activation = axes.plot(lambda x: 0, color=BLUE, stroke_width=4, x_range=[0, 2])
        self.play(Create(activation))
        self.wait(2)


        stimulus = axes.get_vertical_line(axes.c2p(2, 1, 0), color=RED)
        self.play(Create(stimulus))
        self.wait(2)

        response = axes.plot(lambda x: np.exp(-(x-1)), color=BLUE, stroke_width=4, x_range=[2, 10])
        self.play(Create(response))
        self.wait(2)

        self.play(FadeOut(graph), FadeOut(eq1), FadeOut(eq2), FadeOut(activation), FadeOut(stimulus), FadeOut(response))


        # ================= LIF =================
        # Set up axes for the graph
        axes = Axes(
            x_range=[0, 100, 10],
            y_range=[0, 1, 0.1],
            axis_config={"include_tip": False, "numbers_to_exclude": [0]},
        )

        # Labels for axes
        x_label = axes.get_x_axis_label(Text("Time (ms)"), edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label(Text("Membrane Potential (V)"))

        # Adding axes to the scene
        self.play(Create(axes), Write(x_label), Write(y_label))

        times, voltages = LIF()

        activation_graph = axes.plot_line_graph(times, voltages, line_color=BLUE, stroke_width=4, add_vertex_dots=False)

        threshold_line = axes.plot(lambda x: 0.6)
        self.play(Write(threshold_line))
        self.wait(1)

        times, voltages = LIF(tau=5, t0=10, t1=25, t2=50, w=0.4, threshold=0.6, reset=0.0)

        for i in range(30):
            times, voltages = LIF(tau=5, t0=10 + i, t1=25, t2=50, w=0.4, threshold=0.6, reset=0.0)
            activation_graph.become(axes.plot_line_graph(times, voltages, line_color=BLUE, stroke_width=4, add_vertex_dots=False))

            if i == 0:
                self.play(Write(activation_graph))
                self.wait(1)
            elif i == 11:
                self.wait(5)
            else:
                self.wait(0.1)

        self.wait(1)

        items = VGroup(axes, x_label, y_label, activation_graph, threshold_line)
        self.play(items.animate.scale(0.5).to_edge(LEFT))

        hyper_text = Paragraph("The hyperparameters of the LIF model are:", "- tau: the time constant of the neuron", "- w: the weight of the spikes", "- threshold: the threshold at which the neuron fires", font="Arial").scale(0.5).next_to(axes, RIGHT)
        self.play(Write(hyper_text))
        self.wait(5)
        self.play(FadeOut(hyper_text))

        hyper_text2 = Paragraph("sensitive to the hyperparameters", "\n- changing the hyperparameters\ncan change the behaviour of the neuron", font="Arial").scale(0.5).next_to(items, RIGHT)

        self.play(Write(hyper_text2))
        self.wait(2)
        self.play(FadeOut(hyper_text2))

        detail = Paragraph("increasing the threshold will only\nlet the strongest signals through").scale(0.5).next_to(items, RIGHT)
        self.play(Write(detail))

        for t in [0.7, 0.8, 0.9]:
            threshold_line.become(axes.plot(lambda x: t))
            for i in range(20):
                times, voltages = LIF(tau=5, t0=10 + i, t1=25, t2=50, w=0.4, threshold=t, reset=0.0)
                activation_graph.become(axes.plot_line_graph(times, voltages, line_color=BLUE, stroke_width=4, add_vertex_dots=False))

                if i == 0:
                    self.play(Write(activation_graph))
                else:
                    self.wait(0.3)
            self.wait(2)


        self.play(FadeOut(hyper_text), FadeOut(items), FadeOut(detail))


        # ======== Training SNNs ======

        title = Paragraph("Training Spiking Neural Networks", font="Arial").scale(1.5)
        self.play(Write(title))

        self.wait(2)
        self.play(title.animate.shift(UP * 2).scale(0.5))

        desc = Paragraph("- Many ways to train SNNs", "\t- Reservoir Computing", "\t- Evolutionary Algorithmss", "\t- Surrogate gradient descent", font="Arial").scale(0.5).next_to(title, DOWN)
        self.play(Write(desc))
        self.wait(2)

        self.play(FadeOut(title), FadeOut(desc))

        # ==== Surrogate Gradient Descent ====

        title = Paragraph("Surrogate Gradient Descent", font="Arial").scale(1.5)
        self.play(Write(title))


        self.wait(2)
        self.play(title.animate.shift(UP * 2).scale(0.5))

        desc = Paragraph("- Each neuron has a different effect on the error of the network", "- We have methods to minimize the error (Gradient Descent)", "- However it only works for nonlinear functions", font="Arial").scale(0.5).next_to(title, DOWN)

        self.play(Write(desc))
        self.wait(2)

        self.play(FadeOut(title), FadeOut(desc))

        question = Text("Does this look linear to you?").scale(1.5)

        self.play(Write(question))
        self.wait(2)

        self.play(FadeOut(question))

        
        axes = Axes(
            x_range=[0, 100, 10],
            y_range=[0, 1, 0.1],
            axis_config={"include_tip": False, "numbers_to_exclude": [0]},
        )

        # Labels for axes
        x_label = axes.get_x_axis_label(Text("Time (ms)"), edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label(Text("Membrane Potential (V)"))

        times, voltages = LIF(tau=5, t0=10, t1=25, t2=50, w=0.4, threshold=0.6, reset=0.0, smoothing=False)

        activation_graph = axes.plot_line_graph(times, voltages, line_color=BLUE, stroke_width=4, add_vertex_dots=False)

        plot_nonlinear = VGroup(axes, x_label, y_label, activation_graph).scale(0.5)

        self.play(Create(axes), Create(activation_graph), Write(x_label), Write(y_label))
        self.wait(2)

        self.play(FadeOut(plot_nonlinear))

        question = Text("What about now?").scale(1.5)
        self.play(Write(question))
        self.wait(2)
        self.play(FadeOut(question))


        axes = Axes(
            x_range=[0, 100, 10],
            y_range=[0, 1, 0.1],
            axis_config={"include_tip": False, "numbers_to_exclude": [0]},
        )

        # Labels for axes
        x_label = axes.get_x_axis_label(Text("Time (ms)"), edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label(Text("Membrane Potential (V)"))

        times, voltages = LIF(tau=5, t0=10, t1=25, t2=50, w=0.4, threshold=0.6, reset=0.0, smoothing=True)

        activation_graph = axes.plot_line_graph(times, voltages, line_color=BLUE, stroke_width=4, add_vertex_dots=False)

        plot_linear = VGroup(axes, x_label, y_label, activation_graph).scale(0.5)

        self.play(Create(axes), Create(activation_graph), Write(x_label), Write(y_label))
        self.wait(2)
        self.play(FadeOut(plot_linear))


        desc = Paragraph("We can get predictions from the network\n using the non-linear, spiking activations.", font="Arial").scale(0.5).to_edge(UP)
        self.play(Write(desc))
        plot_nonlinear.next_to(desc, DOWN)
        self.play(Create(plot_nonlinear))
        self.wait(2)
        self.play(FadeOut(plot_nonlinear), FadeOut(desc))

        desc = Paragraph("And we can train the network using the linear,\nsmoothed activations.", font="Arial").scale(0.5).to_edge(UP)
        self.play(Write(desc))
        plot_linear.next_to(desc, DOWN)
        self.play(Create(plot_linear))
        self.wait(2)
        self.play(FadeOut(plot_linear), FadeOut(desc))


        self.wait(2)

        cons = Paragraph("In practice this works well", "but, importantly, we are optimizing a different function", font="Arial").scale(0.5)
        self.play(Write(cons))
        self.wait(2)
        self.play(FadeOut(cons))



