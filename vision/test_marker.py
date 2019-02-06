import unittest
import marker as marker
from math import pi


class TestMarker(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_angle_trunc(self):
        c1 = marker.Marker.angle_trunc(1.0)
        c2 = marker.Marker.angle_trunc(-1.0)
        c3 = marker.Marker.angle_trunc(0)
        c4 = marker.Marker.angle_trunc(-7.0)

        self.assertAlmostEqual(c1, 1.0, 'positive angle error')
        self.assertAlmostEqual(
            c2, -1.0 + (pi*2), 'negative, no-loop, angle error')
        self.assertAlmostEqual(c3, 0, 'zero angle error')
        self.assertAlmostEqual(
            c4, -7.0 + (pi*4), 'negative, loop, angle error')

    def test_angle_between_points(self):
        c1 = marker.Marker.angle_between_points(0, 0, 10, 10)

        self.assertAlmostEqual(c1, pi/4, 'angle calculation error')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMarker)
    unittest.TextTestRunner(verbosity=2).run(suite)
