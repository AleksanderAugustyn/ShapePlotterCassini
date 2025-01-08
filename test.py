"""
Nuclear Shape Plotter using Cassini Ovals - A program to visualize and analyze nuclear shapes.
This version implements an object-oriented design for better organization and maintainability.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

matplotlib.use('TkAgg')


def double_factorial(n):
    """Calculate double factorial n!!"""
    return math.prod(range(n, 0, -2))


@dataclass
class CassiniParameters:
    """Class to store Cassini shape parameters."""
    protons: int
    neutrons: int
    alpha: float = 0.0
    alpha_params: List[float] = field(default_factory=lambda: [0.0] * 5)  # Now includes α₂
    r0: float = 1.16  # Radius constant in fm

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not isinstance(self.alpha_params, list):
            raise TypeError("alpha_params must be a list")

        if len(self.alpha_params) != 5:  # Updated for 5 parameters
            original_length = len(self.alpha_params)
            if len(self.alpha_params) < 5:
                self.alpha_params.extend([0.0] * (5 - len(self.alpha_params)))
            else:
                self.alpha_params = self.alpha_params[:5]

    @property
    def nucleons(self) -> int:
        """Total number of nucleons."""
        return self.protons + self.neutrons


class CassiniShapeCalculator:
    """Class for performing Cassini shape calculations."""

    def __init__(self, params: CassiniParameters):
        self.params = params

    def calculate_epsilon(self) -> float:
        """Calculate epsilon parameter from alpha and alpha parameters."""
        alpha = self.params.alpha
        alpha_params = self.params.alpha_params

        sum_all = sum(alpha_params)
        sum_alternating = sum((-1) ** n * val for n, val in enumerate(alpha_params, 1))

        # Calculate factorial sum term
        sum_factorial = 0
        for n in range(1, 3):  # For α₂ and α₄
            idx = 2 * n - 1  # Convert to 0-based index
            if idx < len(alpha_params):
                val = alpha_params[idx]
                sum_factorial += ((-1) ** n * val *
                                  double_factorial(2 * n - 1) /
                                  (2 ** n * math.factorial(n)))


        print(f"sum_factorial: {sum_factorial}")

        epsilon = ((alpha - 1) / 4 * ((1 + sum_all) ** 2 + (1 + sum_alternating) ** 2) +
                   (alpha + 1) / 2 * (1 + sum_factorial) ** 2)

        return epsilon

    def calculate_coordinates(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cylindrical coordinates using Cassini oval parametrization."""
        R_0 = self.params.r0 * (self.params.nucleons ** (1 / 3))
        epsilon = self.calculate_epsilon()
        s = epsilon * R_0 ** 2

        # Calculate R(x) using Legendre polynomials
        R = R_0 * (1 + sum(alpha_n * np.polynomial.legendre.Legendre.basis(n + 1)(x)
                           for n, alpha_n in enumerate(self.params.alpha_params)))

        # Calculate p(x)
        p2 = R ** 4 + 2 * s * R ** 2 * (2 * x ** 2 - 1) + s ** 2
        p = np.sqrt(p2)

        # Calculate ρ and z
        rho = np.sqrt(np.maximum(0, p - R ** 2 * (2 * x ** 2 - 1) - s)) / np.sqrt(2)
        z = np.sign(x) * np.sqrt(np.maximum(0, p + R ** 2 * (2 * x ** 2 - 1) + s)) / np.sqrt(2)

        return rho, z

    def calculate_zcm(self, n_points: int = 1000) -> float:
        """Calculate the z-coordinate of the center of mass."""
        x = np.linspace(-1, 1, n_points)
        rho, z = self.calculate_coordinates(x)

        # Calculate differential elements
        dz = np.diff(z)
        rho_midpoints = (rho[1:] + rho[:-1]) / 2
        z_midpoints = (z[1:] + z[:-1]) / 2

        # Volume element dV = πρ²dz for constant density
        volume_elements = rho_midpoints * rho_midpoints * dz

        # Calculate center of mass
        total_volume = np.sum(volume_elements)
        z_cm = np.sum(volume_elements * z_midpoints) / total_volume

        return z_cm


class CassiniShapePlotter:
    """Class for handling the plotting interface and user interaction."""

    def __init__(self):
        """Initialize the plotter with default settings."""
        # Define all instance attributes
        self.initial_z = 92  # Uranium
        self.initial_n = 144
        self.initial_alpha = 0.0
        self.initial_alphas = [0.0, 0.0, 0.0, 0.0, 0.0]  # α₁, α₂, α₃, α₄
        self.x_points = np.linspace(-1, 1, 2000)

        # UI elements
        self.fig = None
        self.ax_plot = None
        self.line = None
        self.line_mirror = None
        self.slider_z = None
        self.slider_n = None
        self.btn_z_increase = None
        self.btn_z_decrease = None
        self.btn_n_increase = None
        self.btn_n_decrease = None
        self.slider_alpha = None
        self.sliders = []
        self.buttons = []
        self.reset_button = None
        self.save_button = None

        # Initialize nuclear parameters
        self.nuclear_params = CassiniParameters(
            protons=self.initial_z,
            neutrons=self.initial_n,
            alpha=self.initial_alpha,
            alpha_params=self.initial_alphas
        )

        # Set up the interface
        self.create_figure()
        self.setup_controls()
        self.setup_event_handlers()

    def create_figure(self):
        """Create and set up the matplotlib figure."""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_plot = self.fig.add_subplot(111)

        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9)

        # Set up the main plot
        self.ax_plot.set_aspect('equal')
        self.ax_plot.grid(True)
        self.ax_plot.set_title('Nuclear Shape (Cassini Parametrization)', fontsize=14)
        self.ax_plot.set_xlabel('X (fm)', fontsize=12)
        self.ax_plot.set_ylabel('Y (fm)', fontsize=12)

        # Initialize the shape plot
        calculator = CassiniShapeCalculator(self.nuclear_params)
        rho, z = calculator.calculate_coordinates(self.x_points)
        z_cm = calculator.calculate_zcm()
        z = (z - z_cm)  # Center the shape

        self.line, = self.ax_plot.plot(z, rho)
        self.line_mirror, = self.ax_plot.plot(z, -rho)

    def setup_controls(self):
        """Set up all UI controls."""
        # Create proton (Z) controls
        ax_z = plt.axes((0.25, 0.00, 0.5, 0.02))
        ax_z_decrease = plt.axes((0.16, 0.00, 0.04, 0.02))
        ax_z_increase = plt.axes((0.80, 0.00, 0.04, 0.02))

        self.slider_z = Slider(ax=ax_z, label='Z', valmin=82, valmax=120,
                               valinit=self.initial_z, valstep=1)
        self.btn_z_decrease = Button(ax_z_decrease, '-')
        self.btn_z_increase = Button(ax_z_increase, '+')

        # Create neutron (N) controls
        ax_n = plt.axes((0.25, 0.02, 0.5, 0.02))
        ax_n_decrease = plt.axes((0.16, 0.02, 0.04, 0.02))
        ax_n_increase = plt.axes((0.80, 0.02, 0.04, 0.02))

        self.slider_n = Slider(ax=ax_n, label='N', valmin=100, valmax=180,
                               valinit=self.initial_n, valstep=1)
        self.btn_n_decrease = Button(ax_n_decrease, '-')
        self.btn_n_increase = Button(ax_n_increase, '+')

        # Style settings for Z and N controls
        for slider in [self.slider_z, self.slider_n]:
            slider.label.set_fontsize(12)
            slider.valtext.set_fontsize(12)

        # Create slider for main alpha parameter
        ax_alpha = plt.axes((0.25, 0.04, 0.5, 0.02))
        self.slider_alpha = Slider(ax=ax_alpha, label='α',
                                   valmin=0.0, valmax=1.05,
                                   valinit=self.initial_alpha, valstep=0.025)

        # Create sliders for alpha parameters with appropriate ranges
        param_ranges = [
            ('α₁', -1.0, 1.0),
            ('α₂', -1.0, 1.0),
            ('α₃', -1.0, 1.0),
            ('α₄', -1.0, 1.0)
        ]

        for i, (label, min_val, max_val) in enumerate(param_ranges):
            ax = plt.axes((0.25, 0.06 + i * 0.02, 0.5, 0.02))
            slider = Slider(ax=ax, label=label,
                            valmin=min_val, valmax=max_val,
                            valinit=self.initial_alphas[i], valstep=0.025)
            self.sliders.append(slider)

        # Create buttons
        ax_reset = plt.axes((0.8, 0.15, 0.1, 0.04))
        self.reset_button = Button(ax=ax_reset, label='Reset')

        ax_save = plt.axes((0.8, 0.1, 0.1, 0.04))
        self.save_button = Button(ax=ax_save, label='Save Plot')

    def setup_event_handlers(self):
        """Set up all event handlers for controls."""
        # Connect slider update functions
        self.slider_z.on_changed(self.update_plot)
        self.slider_n.on_changed(self.update_plot)
        self.slider_alpha.on_changed(self.update_plot)
        for slider in self.sliders:
            slider.on_changed(self.update_plot)

        # Connect proton/neutron button handlers
        self.btn_z_decrease.on_clicked(self.create_button_handler(self.slider_z, -1))
        self.btn_z_increase.on_clicked(self.create_button_handler(self.slider_z, 1))
        self.btn_n_decrease.on_clicked(self.create_button_handler(self.slider_n, -1))
        self.btn_n_increase.on_clicked(self.create_button_handler(self.slider_n, 1))

        # Connect action buttons
        self.reset_button.on_clicked(self.reset_values)
        self.save_button.on_clicked(self.save_plot)

    @staticmethod
    def create_button_handler(slider_obj: Slider, increment: int):
        """Create a button click handler for a slider object."""

        def handler(_):
            """Handle button click event."""
            new_val = slider_obj.val + increment * slider_obj.valstep
            if slider_obj.valmin <= new_val <= slider_obj.valmax:
                slider_obj.set_val(new_val)

        return handler

    def reset_values(self, _):
        """Reset all sliders to their initial values."""
        self.slider_z.set_val(self.initial_z)
        self.slider_n.set_val(self.initial_n)
        self.slider_alpha.set_val(self.initial_alpha)
        for slider, init_val in zip(self.sliders, self.initial_alphas):
            slider.set_val(init_val)

    def save_plot(self, _):
        """Save the current plot to a file."""
        number_of_protons = int(self.slider_z.val)
        number_of_neutrons = int(self.slider_n.val)
        params = [self.slider_alpha.val] + [s.val for s in self.sliders]
        filename = f"cassini_shape_{number_of_protons}_{number_of_neutrons}_{'_'.join(f'{p:.2f}' for p in params)}.png"
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def update_plot(self, _):
        """Update the plot with new parameters."""
        # Get current parameters
        current_params = CassiniParameters(
            protons=int(self.slider_z.val),
            neutrons=int(self.slider_n.val),
            alpha=self.slider_alpha.val,
            alpha_params=[s.val for s in self.sliders]
        )

        # Calculate new shape
        calculator = CassiniShapeCalculator(current_params)
        rho, z = calculator.calculate_coordinates(self.x_points)
        z_cm = calculator.calculate_zcm()
        z = (z - z_cm)  # Center the shape

        # Update plot
        self.line.set_data(z, rho)
        self.line_mirror.set_data(z, -rho)

        # Update plot limits
        max_val = max(np.max(np.abs(z)), np.max(np.abs(rho))) * 1.2
        self.ax_plot.set_xlim(-max_val, max_val)
        self.ax_plot.set_ylim(-max_val, max_val)

        # Update title with current nuclear information
        self.ax_plot.set_title(f'Nuclear Shape (Z={current_params.protons}, N={current_params.neutrons}, A={current_params.nucleons})',
                               fontsize=14)

        self.fig.canvas.draw_idle()

    def run(self):
        """Start the interactive plotting interface."""
        self.update_plot(None)
        plt.show(block=True)


def main():
    """Main entry point for the application."""
    plotter = CassiniShapePlotter()
    plotter.run()


if __name__ == '__main__':
    main()
