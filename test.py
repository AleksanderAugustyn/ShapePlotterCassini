import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import legendre


def calculate_epsilon(alpha, alpha_n):
    """Calculate epsilon from alpha and other deformation parameters."""
    sum_alpha = np.sum(alpha_n)
    sum_alpha_alternating = np.sum([(-1) ** n * a for n, a in enumerate(alpha_n)])

    # First term
    term1 = (alpha - 1) / 4 * ((1 + sum_alpha) ** 2 + (1 + sum_alpha_alternating) ** 2)

    # Second term - simplified version considering only provided alphas
    sum_even = np.sum([(-1) ** n * alpha_n[n] * (2 * n + 1) / (2 ** (n + 1) * math.factorial(n + 1))
                       for n in range(len(alpha_n))])
    term2 = (alpha + 1) / 2 * (1 + sum_even) ** 2

    return term1 + term2


def calculate_radius(x, R0, alpha_n, Pn):
    """Calculate R(x) using the expansion in Legendre polynomials."""
    expansion = sum(a * P(x) for a, P in zip(alpha_n, Pn))
    return R0 * (1 + expansion)


def get_coordinates(x_vals, R0, alpha, alpha_n):
    """Calculate the physical coordinates (ρ, z) for given parameters."""
    # Calculate epsilon and s
    epsilon = calculate_epsilon(alpha, alpha_n)
    s = epsilon * R0 ** 2

    # Prepare Legendre polynomials
    Pn = [legendre(n) for n in range(1, len(alpha_n) + 1)]

    # Calculate R(x) for each x
    rho_coords = []
    z_coords = []

    for x in x_vals:
        R = calculate_radius(x, R0, alpha_n, Pn)

        # Calculate p²(x)
        p_squared = R ** 4 + 2 * s * R ** 2 * (2 * x ** 2 - 1) + s ** 2

        # Calculate ρ and z
        if p_squared >= 0:
            p = np.sqrt(p_squared)
            term1 = p - R ** 2 * (2 * x ** 2 - 1)
            term2 = p + R ** 2 * (2 * x ** 2 - 1)

            if term1 >= 0:
                rho = np.sqrt(term1) / np.sqrt(2) - s
                z = np.sign(x) * np.sqrt(term2) / np.sqrt(2) + s

                rho_coords.append(rho)
                z_coords.append(z)

    return np.array(rho_coords), np.array(z_coords)


def plot_nuclear_shape(alpha, alpha1, alpha3, alpha4, R0=1.0):
    """Plot the nuclear shape for given deformation parameters."""
    x_vals = np.linspace(-1, 1, 1000)
    alpha_n = [alpha1, 0, alpha3, alpha4]  # α₂=0 included for proper indexing

    rho, z = get_coordinates(x_vals, R0, alpha, alpha_n)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the positive and negative ρ parts to show the full cross-section
    plt.plot(z, rho, 'b-', label='Surface')
    plt.plot(z, -rho, 'b-')

    # Fill the shape
    plt.fill_between(z, -rho, rho, alpha=0.3, color='blue')

    plt.title(f'Nuclear Shape (α={alpha:.2f}, α₁={alpha1:.2f}, α₃={alpha3:.2f}, α₄={alpha4:.2f})')
    plt.xlabel('z/R₀')
    plt.ylabel('ρ/R₀')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Parameters for a sample shape (e.g., from Fig. 6 in the paper)
    plot_nuclear_shape(
        alpha=0.650,  # elongation
        alpha1=0.200,  # dipole
        alpha3=0.025,  # octupole (asymmetry)
        alpha4=-0.050  # hexadecapole (neck)
    )
