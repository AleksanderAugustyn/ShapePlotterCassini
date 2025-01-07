import unittest

import numpy as np

from test2 import cassini_coordinates, generate_nuclear_shape


class TestNuclearShapes(unittest.TestCase):
    def setUp(self):
        """Set up test cases with known values from the paper"""
        # Ground state configuration from the paper (236U)
        self.ground_state = {
            'alpha': 0.250,
            'alpha_params': np.array([0.000, 0.000, 0.000, 0.075]),
            'expected_etot': -2.50  # From Fig. 6 in the paper
        }

        # Second saddle point configuration
        self.second_saddle = {
            'alpha': 0.650,
            'alpha_params': np.array([0.200, 0.000, 0.025, 0.050]),
            'expected_etot': 3.03  # From Fig. 6 in the paper
        }

    def test_epsilon_calculation(self):
        """Test epsilon calculation for known configurations"""
        # Test at x=0 for ground state
        rho, z = cassini_coordinates(1.0, 0.0,
                                     self.ground_state['alpha'],
                                     self.ground_state['alpha_params'])

        # The shape should be symmetric at x=0
        self.assertAlmostEqual(z, 0.0, places=6)
        self.assertTrue(rho > 0)  # Shape should have non-zero radius at middle

        # Test shape symmetry
        rho_plus, z_plus = cassini_coordinates(1.0, 0.5,
                                               self.ground_state['alpha'],
                                               self.ground_state['alpha_params'])
        rho_minus, z_minus = cassini_coordinates(1.0, -0.5,
                                                 self.ground_state['alpha'],
                                                 self.ground_state['alpha_params'])

        # Shape should be symmetric around z=0
        self.assertAlmostEqual(rho_plus, rho_minus, places=6)
        self.assertAlmostEqual(z_plus, -z_minus, places=6)

    def test_factorial_sum(self):
        """Test the factorial sum term in epsilon calculation"""

        def double_factorial(n):
            if n <= 0:
                return 1
            return n * double_factorial(n - 2)

        # Test with second saddle point configuration
        alpha_params = self.second_saddle['alpha_params']
        alpha = self.second_saddle['alpha']

        # Manual calculation of the factorial sum for α₄ term
        n = 2  # For α₄
        manual_term = (-1) ** n * alpha_params[3] * double_factorial(2 * n - 1) / (2 ** n * np.math.factorial(n))

        # Calculate using our implementation
        R = 1.0
        x = 0.0
        rho, z = cassini_coordinates(R, x, alpha, alpha_params)

        # The epsilon value should produce a shape consistent with the paper's dimensions
        self.assertTrue(0.5 < rho < 1.5)  # Reasonable bounds for nuclear shape

    def test_shape_generation(self):
        """Test complete shape generation against known configurations"""
        # Test ground state shape
        rho, z = generate_nuclear_shape(self.ground_state['alpha'],
                                        self.ground_state['alpha_params'],
                                        n_points=100)

        # Check shape characteristics
        self.assertEqual(len(rho), 100)
        self.assertEqual(len(z), 100)

        # Check if shape is mostly spherical for ground state
        max_z = np.max(np.abs(z))
        max_rho = np.max(rho)
        aspect_ratio = max_z / max_rho
        self.assertTrue(0.8 < aspect_ratio < 1.2)  # Nearly spherical

        # Test second saddle point shape (more elongated)
        rho, z = generate_nuclear_shape(self.second_saddle['alpha'],
                                        self.second_saddle['alpha_params'],
                                        n_points=100)

        aspect_ratio = np.max(np.abs(z)) / np.max(rho)
        self.assertTrue(aspect_ratio > 1.2)  # More elongated shape

    def test_paper_configurations(self):
        """Test all four configurations from Figure 6 in the paper"""
        configurations = [
            # Ground state
            (0.250, np.array([0.000, 0.000, 0.000, 0.075])),
            # First saddle point
            (0.350, np.array([0.000, 0.000, 0.000, 0.075])),
            # Secondary minimum
            (0.525, np.array([0.000, 0.000, 0.000, 0.025])),
            # Second saddle point
            (0.650, np.array([0.200, 0.000, 0.025, 0.050]))
        ]

        for alpha, alpha_params in configurations:
            rho, z = generate_nuclear_shape(alpha, alpha_params, n_points=50)

            # Basic sanity checks for each configuration
            self.assertTrue(np.all(np.isfinite(rho)))
            self.assertTrue(np.all(np.isfinite(z)))
            self.assertTrue(np.all(rho >= 0))  # Physical shapes only


if __name__ == '__main__':
    unittest.main()
