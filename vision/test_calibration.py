import unittest
import calibration


class TestCalibration(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_save_and_restore(self):
        pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCalibration)
    unittest.TextTestRunner(verbosity=2).run(suite)
