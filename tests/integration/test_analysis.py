import unittest
from bjsfm.analysis import MaxStrain
from tests.test_data import *


class TestMaxStrain(unittest.TestCase):

    def test_quasi_max_strain(self):
        analysis = MaxStrain(
            QUASI, QUASI_THICK,
            DIAMETER,
            et0=QUASI_UNT, et90=QUASI_UNT, et45=QUASI_UNT, etn45=QUASI_UNT,
            ec0=QUASI_UNC, ec90=QUASI_UNC, ec45=QUASI_UNC, ecn45=QUASI_UNC,
            es0=QUASI_SBS, es90=QUASI_SBS, es45=QUASI_SBS, esn45=QUASI_SBS,
        )

