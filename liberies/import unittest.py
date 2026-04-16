import unittest
from unittest.mock import patch
import numpy as np
from numpy.linalg import LinAlgError
from liberies.A_matrix import MMr, NNr, NNq, NNl, S_matrix, S_prime, inverse_MM

# Absolute import based on your project structure

class TestAMatrix(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.U = np.linspace(0.1, 0.9, self.N)
        self.C = np.linspace(0.1, 0.9, self.N)
        self.xi = np.linspace(0.5, 1.0, self.N)
        self.args = (4.25, 0.25, 5/3) # omega, delt, gamma

    @patch('liberies.A_matrix.P_val')
    @patch('liberies.A_matrix.G_val')
    def test_MMr_shape(self, mock_G_val, mock_P_val):
        mock_G_val.return_value = np.ones(self.N)
        mock_P_val.return_value = np.ones(self.N)

        result = MMr(self.U, self.C, self.xi, self.args)
        
        # MMr builds a 4x4 list of arrays of size N, resulting in shape (4, 4, N)
        self.assertEqual(result.shape, (4, 4, self.N))

    @patch('liberies.A_matrix.P_val')
    @patch('liberies.A_matrix.G_val')
    def test_inverse_MM_debug_fix(self, mock_G_val, mock_P_val):
        mock_G_val.return_value = np.ones(self.N)
        mock_P_val.return_value = np.ones(self.N)

        MM_result = MMr(self.U, self.C, self.xi, self.args)
        
        # Reproduce the bug
        with self.assertRaises(LinAlgError):
            inverse_MM(MM_result)
            
        # The fix: np.linalg.inv expects the last two dimensions to be square (e.g., N, 4, 4)
        # So we transpose (4, 4, N) -> (N, 4, 4), invert, then transpose back to (4, 4, N)
        def fixed_inverse_MM(MM):
            # Transpose to (N, 4, 4)
            MM_T = np.transpose(MM, (2, 0, 1))
            # Invert resulting in (N, 4, 4)
            inv_MM_T = np.linalg.inv(MM_T)
            # Transpose back to (4, 4, N)
            return np.transpose(inv_MM_T, (1, 2, 0))

        # Test the fix works smoothly without throwing LinAlgError
        fixed_result = fixed_inverse_MM(MM_result)
        self.assertEqual(fixed_result.shape, (4, 4, self.N))

    @patch('liberies.A_matrix.P_val')
    @patch('liberies.A_matrix.G_val')
    def test_NNq_shape(self, mock_G_val, mock_P_val):
        mock_G_val.return_value = np.ones(self.N)
        mock_P_val.return_value = np.ones(self.N)

        result = NNq(self.U, self.C, self.xi, self.args)
        self.assertEqual(result.shape, (4, 4, self.N))

    @patch('liberies.A_matrix.G_val')
    def test_NNl_shape(self, mock_G_val):
        mock_G_val.return_value = np.ones(self.N)

        result = NNl(self.U, self.C, self.xi, self.args)
        self.assertEqual(result.shape, (4, 4, self.N))

    def test_S_matrix_shape(self):
        result = S_matrix(self.U, self.C, self.xi, self.args)
        self.assertEqual(result.shape, (4, 4, self.N))
        
    def test_S_prime_shape(self):
        result = S_prime(self.U, self.C, self.xi, self.args)
        self.assertEqual(result.shape, (4, 4, self.N))


if __name__ == '__main__':
    unittest.main()