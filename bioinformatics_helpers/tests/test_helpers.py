from bioinformatics_helpers.utils import get_iqr_outliers
import unittest
import numpy as np
class TestHelpers(unittest.TestCase):
    def testGetIqrOutliers(self):
         #https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots/a/identifying-outliers-iqr-rule
        data = np.array([5,7,10,15,19,21,21,22,22,23,23,23,23,23,24,24,24,24,25])
        q1 = 19
        q3 = 24
        iqr = q3 - q1
        low_iqr = q1 - (1.5 * iqr)
        high_iqr = q3 + (1.5 * iqr)
        high_outliers = data[data > high_iqr]
        low_outliers = data[data < low_iqr]

        result = get_iqr_outliers(data)
        
        self.assertAlmostEqual(low_iqr, result["low_value"], delta=5)
        self.assertAlmostEqual(high_iqr, result["high_value"], delta=5)
        self.assertListEqual(high_outliers.tolist(), result["high_outliers"].tolist())
        self.assertListEqual(low_outliers.tolist(), result["low_outliers"].tolist())

if __name__ == '__main__':
    unittest.main()
        