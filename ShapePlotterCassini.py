"""This script plots the shape of a nucleus with volume conservation using Cassini oval parametrization."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox
from scipy import integrate
from scipy.special import legendre

matplotlib.use('TkAgg')

# Constants
r0 = 1.16  # Radius constant in fm


def calculate_legendre_sum(x, alpha_params):
    """Calculate the sum of Legendre polynomials multiplied by alpha parameters."""
    result = 0
    for n, alpha in enumerate(alpha_params, start=1):
        P_n = legendre(n)
        result += alpha * P_n(x)
    return result


def calculate_base_radius(theta, alpha_params, number_of_nucleons):
    """Calculate the base nuclear radius without volume fixing."""
    cos_theta = np.cos(theta)
    deformation_term = 1 + calculate_legendre_sum(cos_theta, alpha_params)
    base_radius = r0 * number_of_nucleons ** (1 / 3)
    return base_radius * deformation_term


def calculate_radius(theta, alpha_params, number_of_nucleons, volume_fix):
    """Calculate the nuclear radius with volume fixing applied."""
    base_r = calculate_base_radius(theta, alpha_params, number_of_nucleons)
    return base_r * volume_fix ** (1 / 3)


def calculate_volume(number_of_nucleons, alpha_params):
    """Calculate the volume of the nucleus by numerical integration."""
    n_theta, n_phi = 400, 400
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)

    r = calculate_base_radius(theta_mesh, alpha_params, number_of_nucleons)
    integrand = (r ** 3 * np.sin(theta_mesh)) / 3

    return integrate.trapezoid(integrate.trapezoid(integrand, theta, axis=1), phi)


def calculate_sphere_volume(number_of_nucleons):
    """Calculate the volume of a spherical nucleus."""
    return 4 / 3 * np.pi * number_of_nucleons * r0 ** 3


def calculate_volume_fixing_factor(number_of_nucleons, alpha_params):
    """Calculate the volume fixing factor to conserve volume."""
    initial_volume = calculate_volume(number_of_nucleons, alpha_params)
    sphere_volume = calculate_sphere_volume(number_of_nucleons)
    return (sphere_volume / initial_volume) ** (1 / 3)


def find_neck_thickness(x_coords, y_coords, theta_vals, degree_range):
    """Find the neck thickness between specified degree range."""
    start_rad, end_rad = np.radians(degree_range)
    mask = (theta_vals >= start_rad) & (theta_vals <= end_rad)
    relevant_x = x_coords[mask]
    relevant_y = y_coords[mask]

    distances = np.abs(relevant_y)
    neck_idx = np.argmin(distances)
    neck_thickness = distances[neck_idx] * 2

    return neck_thickness, relevant_x[neck_idx], relevant_y[neck_idx]


def main():
    """Main function to create and display the nuclear shape plot."""
    # Set up the figure
    fig = plt.figure(figsize=(15, 8))
    ax_plot = fig.add_subplot(121)
    ax_text = fig.add_subplot(122)
    ax_text.axis('off')

    plt.subplots_adjust(left=0.1, bottom=0.48, right=0.9, top=0.98)

    # Add text for keyboard input instructions
    ax_text.text(0.1, 0.35, 'Keyboard Input Format (works with Ctrl+V):\nZ N α1 α2 α3 α4 α5 α6\nExample: 102 154 0.0 0.5 0.0 0.0 0.0 0.0',
                 fontsize=12, verticalalignment='top')

    # Add error message text (initially empty)
    error_text = ax_text.text(0.1, 0.20, '', color='red', fontsize=12, verticalalignment='top')

    # Initial parameters
    num_params = 6  # Number of alpha parameters
    initial_params = (0.0,) * num_params
    initial_z = 102
    initial_n = 154
    theta = np.linspace(0, 2 * np.pi, 2000)

    initial_volume_fix = calculate_volume_fixing_factor(initial_z + initial_n, initial_params)

    # Calculate and plot initial shape
    radius = calculate_radius(theta, initial_params, initial_z + initial_n, initial_volume_fix)
    x = radius * np.sin(theta)
    y = radius * np.cos(theta)
    line, = ax_plot.plot(x, y)

    # Set up the plot
    ax_plot.set_aspect('equal')
    ax_plot.grid(True)
    ax_plot.set_title('Nuclear Shape (Cassini Parametrization)', fontsize=18)
    ax_plot.set_xlabel('X (fm)', fontsize=18)
    ax_plot.set_ylabel('Y (fm)', fontsize=18)

    # Create a text box for volume information
    volume_text = ax_text.text(0.1, 0.4, '', fontsize=24)

    # Create sliders for protons and neutrons
    ax_z = plt.axes((0.25, 0.00, 0.5, 0.02))
    ax_n = plt.axes((0.25, 0.03, 0.5, 0.02))

    slider_z = Slider(ax=ax_z, label='Z', valmin=82, valmax=120, valinit=initial_z, valstep=1)
    slider_n = Slider(ax=ax_n, label='N', valmin=100, valmax=180, valinit=initial_n, valstep=1)

    slider_z.label.set_fontsize(18)
    slider_z.valtext.set_fontsize(18)

    slider_n.label.set_fontsize(18)
    slider_n.valtext.set_fontsize(18)

    # Create decrease/increase buttons for Z and N
    ax_z_decrease = plt.axes((0.16, 0.00, 0.04, 0.02))
    ax_z_increase = plt.axes((0.80, 0.00, 0.04, 0.02))
    ax_n_decrease = plt.axes((0.16, 0.03, 0.04, 0.02))
    ax_n_increase = plt.axes((0.80, 0.03, 0.04, 0.02))

    btn_z_decrease = Button(ax_z_decrease, '-')
    btn_z_increase = Button(ax_z_increase, '+')
    btn_n_decrease = Button(ax_n_decrease, '-')
    btn_n_increase = Button(ax_n_increase, '+')

    # Create sliders and buttons for alpha parameters
    slider_height = 0.03
    sliders = []
    decrease_buttons = []
    increase_buttons = []

    for i in range(num_params):
        ax_decrease = plt.axes((0.16, 0.06 + i * slider_height, 0.04, 0.02))
        ax_slider = plt.axes((0.25, 0.06 + i * slider_height, 0.5, 0.02))
        ax_increase = plt.axes((0.80, 0.06 + i * slider_height, 0.04, 0.02))

        slider = Slider(
            ax=ax_slider,
            label=f'α{i + 1}',
            valmin=-1.0,
            valmax=1.0,
            valinit=initial_params[i],
            valstep=0.01
        )

        btn_decrease = Button(ax_decrease, '-')
        btn_increase = Button(ax_increase, '+')

        slider.label.set_fontsize(18)
        slider.valtext.set_fontsize(18)

        sliders.append(slider)
        decrease_buttons.append(btn_decrease)
        increase_buttons.append(btn_increase)

    # Create save and reset buttons
    ax_save = plt.axes((0.75, 0.45, 0.1, 0.04))
    save_button = Button(ax=ax_save, label='Save Plot')

    ax_reset = plt.axes((0.86, 0.45, 0.1, 0.04))
    reset_button = Button(ax=ax_reset, label='Reset')

    # Create text input field and submit button
    ax_input = plt.axes((0.25, 0.42, 0.5, 0.02))
    text_box = TextBox(ax_input, 'Parameters')
    text_box.label.set_fontsize(12)

    ax_submit = plt.axes((0.80, 0.42, 0.1, 0.02))
    submit_button = Button(ax_submit, 'Submit')

    def reset_values(_):
        """Reset all sliders to their initial values."""
        for slider in sliders:
            slider.set_val(0.0)
        slider_z.set_val(initial_z)
        slider_n.set_val(initial_n)
        text_box.set_val('')

    def submit_parameters(_):
        """Handle parameter submission from text input."""
        try:
            values = [float(val) for val in text_box.text.split()]
            if len(values) != 8:  # 2 for Z,N + 6 for alphas
                raise ValueError("Expected 8 values: Z N α1 α2 α3 α4 α5 α6")

            if not (82 <= values[0] <= 120 and 100 <= values[1] <= 180):
                raise ValueError("Z must be between 82-120 and N between 100-180")

            slider_z.set_val(int(values[0]))
            slider_n.set_val(int(values[1]))

            for i, slider in enumerate(sliders):
                if not (-1.0 <= values[i + 2] <= 1.0):
                    raise ValueError(f"α{i + 1} must be between -1.0 and 1.0")
                slider.set_val(values[i + 2])

            text_box.set_val('')
            error_text.set_text('')
            fig.canvas.draw_idle()

        except (ValueError, IndexError) as e:
            error_text.set_text(f"Error: {str(e)}")
            fig.canvas.draw_idle()

    def save_plot(_):
        """Save the current plot to a file."""
        params = [s.val for s in sliders]
        z_val = int(slider_z.val)
        n_val = int(slider_n.val)
        alpha_values = "_".join(f"{p:.2f}" for p in params)
        filename = f"cassini_{z_val}_{n_val}_{alpha_values}.png"
        fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def find_nearest_point(plot_x, plot_y, angle):
        """Find the nearest point on the curve to a given angle."""
        angles = np.arctan2(plot_y, plot_x)
        angle_diff = np.abs(angles - angle)
        nearest_index = np.argmin(angle_diff)
        return plot_x[nearest_index], plot_y[nearest_index]

    def update(_):
        """Update the plot with new parameters."""
        params = [s.val for s in sliders]
        z_val = int(slider_z.val)
        n_val = int(slider_n.val)
        number_of_nucleons = z_val + n_val

        # Calculate volume fixing factor and apply it
        volume_fix = calculate_volume_fixing_factor(number_of_nucleons, params)

        # Calculate new shape with volume conservation
        plot_radius = calculate_radius(theta, params, number_of_nucleons, volume_fix)
        plot_x = plot_radius * np.cos(theta)
        plot_y = plot_radius * np.sin(theta)
        line.set_data(plot_x, plot_y)

        # Find intersection points and draw axis lines
        x_axis_positive = find_nearest_point(plot_x, plot_y, 0)
        x_axis_negative = find_nearest_point(plot_x, plot_y, np.pi)
        y_axis_positive = find_nearest_point(plot_x, plot_y, np.pi / 2)
        y_axis_negative = find_nearest_point(plot_x, plot_y, -np.pi / 2)

        # Remove previous lines if they exist
        for attr in ['x_axis_line', 'y_axis_line']:
            if hasattr(ax_plot, attr):
                getattr(ax_plot, attr).remove()

        # Draw axis lines
        ax_plot.x_axis_line = ax_plot.plot(
            [x_axis_negative[0], x_axis_positive[0]],
            [x_axis_negative[1], x_axis_positive[1]],
            color='red'
        )[0]

        ax_plot.y_axis_line = ax_plot.plot(
            [y_axis_negative[0], y_axis_positive[0]],
            [y_axis_negative[1], y_axis_positive[1]],
            color='blue'
        )[0]

        # Calculate and draw necks
        neck_thickness_45_135, neck_x_45_135, neck_y_45_135 = find_neck_thickness(
            plot_x, plot_y, theta, (45, 135)
        )
        neck_thickness_30_150, neck_x_30_150, neck_y_30_150 = find_neck_thickness(
            plot_x, plot_y, theta, (30, 150)
        )

        # Remove previous neck lines if they exist
        for attr in ['neck_line_45_135', 'neck_line_30_150']:
            if hasattr(ax_plot, attr):
                getattr(ax_plot, attr).remove()

        # Draw neck lines
        ax_plot.neck_line_45_135 = ax_plot.plot(
            [neck_x_45_135, neck_x_45_135],
            [-neck_thickness_45_135 / 2, neck_thickness_45_135 / 2],
            color='green',
            linewidth=2,
            label='Neck (45°-135°)'
        )[0]

        ax_plot.neck_line_30_150 = ax_plot.plot(
            [neck_x_30_150, neck_x_30_150],
            [-neck_thickness_30_150 / 2, neck_thickness_30_150 / 2],
            color='purple',
            linewidth=2,
            label='Neck (30°-150°)'
        )[0]

        # Update plot limits
        max_radius = np.max(np.abs(plot_radius)) * 1.5
        ax_plot.set_xlim(-max_radius, max_radius)
        ax_plot.set_ylim(-max_radius, max_radius)

        # Calculate dimensions
        max_x_length = np.max(plot_y) - np.min(plot_y)
        max_y_length = np.max(plot_x) - np.min(plot_x)
        along_x_length = calculate_radius(0.0, params, number_of_nucleons, volume_fix) + calculate_radius(np.pi, params, number_of_nucleons, volume_fix)
        along_y_length = calculate_radius(np.pi / 2, params, number_of_nucleons, volume_fix) + calculate_radius(-np.pi / 2, params, number_of_nucleons, volume_fix)

        # Calculate volumes
        sphere_volume = calculate_sphere_volume(number_of_nucleons)
        shape_volume = calculate_volume(number_of_nucleons, params)

        # Check for negative radius
        negative_radius = False
        if np.any(plot_radius < 0):
            negative_radius = True

        # Update information display
        volume_text.set_text(
            f'Sphere Volume: {sphere_volume:.4f} fm³\n'
            f'Shape Volume: {shape_volume:.4f} fm³\n'
            f'Volume Fixing Factor: {volume_fix:.8f}\n'
            f'Max X Length: {max_x_length:.2f} fm\n'
            f'Max Y Length: {max_y_length:.2f} fm\n'
            f'Length Along X Axis (red): {along_x_length:.2f} fm\n'
            f'Length Along Y Axis (blue): {along_y_length:.2f} fm\n'
            f'Neck Thickness (45°-135°, green): {neck_thickness_45_135:.2f} fm\n'
            f'Neck Thickness (30°-150°, purple): {neck_thickness_30_150:.2f} fm\n' +
            ('Negative radius detected!\n' if negative_radius else '')
        )

        # Update the legend
        ax_plot.legend(fontsize='small', loc='upper right')
        fig.canvas.draw_idle()

    # Function to create button click handlers
    def create_button_handler(slider_obj, increment):
        """Create a button click handler for a slider object."""

        def handler(_):
            """Handle button click event."""
            new_val = slider_obj.val + increment * slider_obj.valstep
            if slider_obj.valmin <= new_val <= slider_obj.valmax:
                slider_obj.set_val(new_val)

        return handler

    # Connect button click handlers
    for i, slider in enumerate(sliders):
        decrease_buttons[i].on_clicked(create_button_handler(slider, -1))
        increase_buttons[i].on_clicked(create_button_handler(slider, 1))

    # Connect proton and neutron button handlers
    btn_z_decrease.on_clicked(create_button_handler(slider_z, -1))
    btn_z_increase.on_clicked(create_button_handler(slider_z, 1))
    btn_n_decrease.on_clicked(create_button_handler(slider_n, -1))
    btn_n_increase.on_clicked(create_button_handler(slider_n, 1))

    # Enable key events for the text box
    text_box_widget = text_box.ax.figure.canvas.get_tk_widget()
    root = text_box_widget.master

    def handle_paste(_):
        """Handle paste events from clipboard."""
        try:
            clipboard_text = root.clipboard_get()
            text_box.set_val(clipboard_text)
            return "break"  # Prevents default paste behavior
        except Exception as e:
            print(f"Error pasting from clipboard: {e}")
            return "break"

    # Bind Ctrl+V to the root window
    root.bind_all('<Control-v>', handle_paste)

    # Connect all control buttons
    submit_button.on_clicked(submit_parameters)
    save_button.on_clicked(save_plot)
    reset_button.on_clicked(reset_values)

    # Connect slider update functions
    for slider in sliders:
        slider.on_changed(update)
    slider_z.on_changed(update)
    slider_n.on_changed(update)

    # Update plot with initial values
    update(None)

    plt.show(block=True)


if __name__ == '__main__':
    main()
