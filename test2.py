import matplotlib.pyplot as plt
import numpy as np


def cassini_coordinates(R, x, alpha, alpha1, alpha3, alpha4):
    """
    Calculate cylindrical coordinates using Cassini oval parametrization.

    Parameters:
    R, x: Lemniscate coordinate system parameters
    alpha: Main deformation parameter (0 to 1)
    alpha1, alpha3, alpha4: Shape modification parameters
    """
    # Calculate epsilon using the formula from equation (6)
    eps = (alpha - 1) / 4 * ((1 + alpha1 + alpha3 + alpha4) ** 2 +
                             (1 - alpha1 + alpha3 - alpha4) ** 2) + \
          (alpha + 1) / 2 * (1 + sum((-1) ** n * alpha_n * (2 * n - 1) / (2 ** n * np.math.factorial(n))
                                     for n, alpha_n in [(1, alpha1), (3, alpha3), (4, alpha4)])) ** 2

    # Calculate s parameter
    s = eps * R ** 2

    # Calculate p(x) according to equation (3)
    p2 = R ** 4 + 2 * s * R ** 2 * (2 * x ** 2 - 1) + s ** 2
    p = np.sqrt(p2)

    # Calculate rho and z according to equations (3)
    rho = 1 / np.sqrt(2) * np.sqrt(p - R ** 2 * (2 * x ** 2 - 1) - s)
    z = np.sign(x) / np.sqrt(2) * np.sqrt(p + R ** 2 * (2 * x ** 2 - 1) + s)

    return rho, z


def generate_nuclear_shape(alpha, alpha1, alpha3, alpha4, n_points=1000):
    """
    Generate points for nuclear shape using Cassini parametrization.
    """
    x = np.linspace(-1, 1, n_points)
    R0 = 1.0  # Base radius

    # Calculate R(x) using equation (4)
    R = R0 * (1 + alpha1 * x + alpha3 * (5 * x ** 3 - 3 * x) / 2 +
              alpha4 * (35 * x ** 4 - 30 * x ** 2 + 3) / 8)

    rho_points = []
    z_points = []

    for i in range(len(x)):
        rho, z = cassini_coordinates(R[i], x[i], alpha, alpha1, alpha3, alpha4)
        rho_points.append(rho)
        z_points.append(z)

    return np.array(rho_points), np.array(z_points)


def plot_nuclear_shapes():
    """
    Create plots for different nuclear configurations from the paper.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Nuclear Shapes in Fission Process (236U)', fontsize=14)

    # Ground state configuration
    rho1, z1 = generate_nuclear_shape(0.250, 0.000, 0.000, 0.075)
    axs[0, 0].plot(z1, rho1, 'b-', label='Nuclear Surface')
    axs[0, 0].plot(z1, -rho1, 'b-')
    axs[0, 0].set_title('Ground State')
    axs[0, 0].set_aspect('equal')

    # First saddle point
    rho2, z2 = generate_nuclear_shape(0.350, 0.000, 0.000, 0.075)
    axs[0, 1].plot(z2, rho2, 'r-', label='Nuclear Surface')
    axs[0, 1].plot(z2, -rho2, 'r-')
    axs[0, 1].set_title('First Saddle Point')
    axs[0, 1].set_aspect('equal')

    # Secondary minimum
    rho3, z3 = generate_nuclear_shape(0.525, 0.000, 0.000, 0.025)
    axs[1, 0].plot(z3, rho3, 'g-', label='Nuclear Surface')
    axs[1, 0].plot(z3, -rho3, 'g-')
    axs[1, 0].set_title('Secondary Minimum')
    axs[1, 0].set_aspect('equal')

    # Second saddle point
    rho4, z4 = generate_nuclear_shape(0.650, 0.200, 0.025, 0.050)
    axs[1, 1].plot(z4, rho4, 'm-', label='Nuclear Surface')
    axs[1, 1].plot(z4, -rho4, 'm-')
    axs[1, 1].set_title('Second Saddle Point')
    axs[1, 1].set_aspect('equal')

    # Add grid and labels
    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlabel('z/R₀')
        ax.set_ylabel('ρ/R₀')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_nuclear_shapes()
