"""
Nuclear Shapes in Fission Process (236U)
"""

import math

import matplotlib.pyplot as plt
import numpy as np


def double_factorial(n):
    """Calculate double factorial n!!"""
    return math.prod(range(n, 0, -2))


def cassini_coordinates(R, x, alpha, alpha_params):
    """
    Calculate cylindrical coordinates using Cassini oval parametrization.

    Parameters:
    R, x: Lemniscate coordinate system parameters
    alpha: Main deformation parameter (0 to 1)
    alpha_params: Array of alpha parameters where index+1 corresponds to the parameter number
                 e.g., alpha_params[0] is α₁, alpha_params[2] is α₃, etc.
    """
    # Create dictionary of non-zero alpha parameters
    alpha_dict = {i + 1: val for i, val in enumerate(alpha_params)}

    # Calculate first sum term
    sum_all = sum(alpha_dict.values())

    # Calculate second sum term (with alternating signs)
    sum_alternating = sum((-1) ** n * val for n, val in alpha_dict.items())

    # Calculate third sum term (for the factorial part)
    # For nth term in sum, use α₂ₙ
    # e.g., n=1: use α₂, n=2: use α₄, n=3: use α₆, etc., if available, otherwise use 0
    sum_factorial = sum((-1) ** n * alpha_dict.get(2 * n, 0) * double_factorial(2 * n - 1) / (2 ** n * math.factorial(n))
                        for n in range(1, (max(alpha_dict.keys()) + 1) // 2 + 1))

    # Calculate epsilon using the formula from equation (6)
    eps = (alpha - 1) / 4 * ((1 + sum_all) ** 2 + (1 + sum_alternating) ** 2) + (alpha + 1) / 2 * (1 + sum_factorial) ** 2

    R_0 = 1.16 * 236 ** (1 / 3)  # Base radius

    # Calculate s parameter
    s = eps * R_0 ** 2

    # Calculate p(x) according to equation (3)
    p2 = R ** 4 + 2 * s * R ** 2 * (2 * x ** 2 - 1) + s ** 2
    p = np.sqrt(p2)

    # Calculate rho and z according to equations (3)

    under_sqrt_rho = p - R ** 2 * (2 * x ** 2 - 1) - s
    if under_sqrt_rho <= abs(1e-10):
        under_sqrt_rho = 0

    under_sqrt_z = p + R ** 2 * (2 * x ** 2 - 1) + s
    if under_sqrt_z <= abs(1e-10):
        under_sqrt_z = 0

    # if p - R ** 2 * (2 * x ** 2 - 1) - s < 0:
    #     print("Warning: p - R ** 2 * (2 * x ** 2 - 1) - s < 0")
    #     print(f"alpha: {alpha}, alpha_params: {alpha_params}")
    #     print(f"p: {p}, R: {R}, x: {x}, s: {s}")
    #     print(f"p - R ** 2 * (2 * x ** 2 - 1) - s: {p - R ** 2 * (2 * x ** 2 - 1) - s}")

    rho = 1 / np.sqrt(2) * np.sqrt(under_sqrt_rho)
    z = np.sign(x) / np.sqrt(2) * np.sqrt(under_sqrt_z)

    return rho, z


def generate_nuclear_shape(alpha, alpha_params, n_points=2000):
    """
    Generate points for nuclear shape using Cassini parametrization.

    Parameters:
    alpha: Main deformation parameter
    alpha_params: Array of alpha parameters
    n_points: Number of points for shape discretization
    """
    x = np.linspace(-1, 1, n_points)
    R_0 = 1.16 * 236 ** (1 / 3)  # Base radius

    # Calculate Legendre polynomials for each alpha parameter using numpy
    R = np.zeros_like(x)
    legendre_sum = np.zeros_like(x)
    for n, alpha_n in enumerate(alpha_params, start=1):
        if alpha_n != 0:
            # Add contribution of each non-zero alpha parameter using Legendre polynomial
            legendre = np.polynomial.legendre.Legendre.basis(n)
            legendre_sum += alpha_n * legendre(x)

    R = R_0 * (1 + legendre_sum)

    rho_points = []
    z_points = []

    for i in range(len(x)):
        rho, z = cassini_coordinates(R[i], x[i], alpha, alpha_params)
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
    # [α₁, α₂, α₃, α₄] - using 0 for α₂ as it's not used in the paper
    alpha_params1 = np.array([0.000, 0.000, 0.000, 0.075])
    rho1, z1 = generate_nuclear_shape(0.250, alpha_params1)
    axs[0, 0].plot(z1, rho1, 'b-', label='Nuclear Surface')
    axs[0, 0].plot(z1, -rho1, 'b-')
    axs[0, 0].set_title('Ground State')
    axs[0, 0].set_aspect('equal')

    # First saddle point
    alpha_params2 = np.array([0.000, 0.000, 0.000, -0.075])
    rho2, z2 = generate_nuclear_shape(0.350, alpha_params2)
    axs[0, 1].plot(z2, rho2, 'r-', label='Nuclear Surface')
    axs[0, 1].plot(z2, -rho2, 'r-')
    axs[0, 1].set_title('First Saddle Point')
    axs[0, 1].set_aspect('equal')

    # Secondary minimum
    alpha_params3 = np.array([0.000, 0.000, 0.000, 0.025])
    rho3, z3 = generate_nuclear_shape(0.525, alpha_params3)
    axs[1, 0].plot(z3, rho3, 'g-', label='Nuclear Surface')
    axs[1, 0].plot(z3, -rho3, 'g-')
    axs[1, 0].set_title('Secondary Minimum')
    axs[1, 0].set_aspect('equal')

    # Second saddle point
    alpha_params4 = np.array([0.200, 0.000, 0.025, 0.050])
    rho4, z4 = generate_nuclear_shape(0.650, alpha_params4)
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
