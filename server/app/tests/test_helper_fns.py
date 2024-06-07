import numpy as np
from math import sqrt
import datetime
import unittest

from server.app.utils.helpers import calculate_angle, to_timestamp, calculate_distance

class TestCalculateAngle(unittest.TestCase):
    def test_calculate_angle(self):
        # Define points
        a = (1, 1)
        b = (2, 2)
        c = (3, 3)
        # Expected angle is 0 because it's a straight line
        expected_angle = 0
        calculated_angle = calculate_angle(a, b, c)
        # Use assertAlmostEqual due to floating point arithmetic
        self.assertAlmostEqual(calculated_angle, expected_angle, places=1)

class TestToTimestamp(unittest.TestCase):
    def test_to_timestamp(self):
        # Define a known timestamp
        time_float = 1609459200  # Corresponds to 2021-01-01 00:00:00 UTC
        expected_timestamp = "2021-01-01 00:00:00,000000"
        calculated_timestamp = to_timestamp(time_float)
        self.assertEqual(calculated_timestamp, expected_timestamp)

class TestCalculateDistance(unittest.TestCase):
    def test_calculate_distance(self):
        # Define points
        p1 = (0, 0)
        p2 = (3, 4)
        # Expected distance is 5 (3-4-5 triangle)
        expected_distance = 5
        calculated_distance = calculate_distance(p1, p2)
        self.assertEqual(calculated_distance, expected_distance)


if __name__ == '__main__':
    unittest.main()
