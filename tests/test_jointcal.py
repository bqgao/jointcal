# This file is part of jointcal.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest
import unittest.mock

import numpy as np

import lsst.log
import lsst.utils

import lsst.jointcal
from lsst.jointcal import MinimizeResult
import lsst.jointcal.chi2
import lsst.jointcal.testUtils


# for MemoryTestCase
def setup_module(module):
    lsst.utils.tests.init()


class TestJointcalIterateFit(lsst.utils.tests.TestCase):
    def setUp(self):
        struct = lsst.jointcal.testUtils.createTwoFakeCcdImages(100, 100)
        self.ccdImageList = struct.ccdImageList
        # so that countStars() returns nonzero results
        for ccdImage in self.ccdImageList:
            ccdImage.resetCatalogForFit()

        self.config = lsst.jointcal.jointcal.JointcalConfig()
        # disable both, so it doesn't configure any refObjLoaders
        self.config.doAstrometry = False
        self.config.doPhotometry = False
        self.jointcal = lsst.jointcal.JointcalTask(config=self.config)

        self.goodChi2 = lsst.jointcal.chi2.Chi2Statistic()
        # chi2/ndof == 2.0 should be non-bad
        self.goodChi2.chi2 = 200.0
        self.goodChi2.ndof = 100

        self.badChi2 = lsst.jointcal.chi2.Chi2Statistic()
        self.badChi2.chi2 = 600.0
        self.badChi2.ndof = 100

        self.nanChi2 = lsst.jointcal.chi2.Chi2Statistic()
        self.nanChi2.chi2 = np.nan
        self.nanChi2.ndof = 100

        self.maxSteps = 20
        self.name = "testing"
        self.whatToFit = ""  # unneeded, since we're mocking the fitter

        # Mock the fitter, association manager, and model, so we can force particular
        # return values/exceptions. Default to "good" return values.
        self.fitter = unittest.mock.Mock(spec=lsst.jointcal.PhotometryFit)
        self.fitter.computeChi2.return_value = self.goodChi2
        self.fitter.minimize.return_value = MinimizeResult.Converged
        self.associations = unittest.mock.Mock(spec=lsst.jointcal.Associations)
        self.associations.getCcdImageList.return_value = self.ccdImageList
        self.model = unittest.mock.Mock(spec=lsst.jointcal.SimpleFluxModel)

    def test_iterateFit_success(self):
        chi2 = self.jointcal._iterate_fit(self.associations, self.fitter,
                                          self.maxSteps, self.name, self.whatToFit)
        self.assertEqual(chi2, self.goodChi2)
        # Once for the for loop, the second time for the rank update.
        self.assertEqual(self.fitter.minimize.call_count, 2)

    def test_iterateFit_failed(self):
        self.fitter.minimize.return_value = MinimizeResult.Failed

        with self.assertRaises(RuntimeError):
            self.jointcal._iterate_fit(self.associations, self.fitter,
                                       self.maxSteps, self.name, self.whatToFit)
        self.assertEqual(self.fitter.minimize.call_count, 1)

    def test_iterateFit_badFinalChi2(self):
        log = unittest.mock.Mock(spec=lsst.log.Log)
        self.jointcal.log = log
        self.fitter.computeChi2.return_value = self.badChi2

        chi2 = self.jointcal._iterate_fit(self.associations, self.fitter,
                                          self.maxSteps, self.name, self.whatToFit)
        self.assertEqual(chi2, self.badChi2)
        log.info.assert_called_with("%s %s", "Fit completed", self.badChi2)
        log.error.assert_called_with("Potentially bad fit: High chi-squared/ndof.")

    def test_iterateFit_exceedMaxSteps(self):
        log = unittest.mock.Mock(spec=lsst.log.Log)
        self.jointcal.log = log
        self.fitter.minimize.return_value = MinimizeResult.Chi2Increased
        maxSteps = 3

        chi2 = self.jointcal._iterate_fit(self.associations, self.fitter,
                                          maxSteps, self.name, self.whatToFit)
        self.assertEqual(chi2, self.goodChi2)
        self.assertEqual(self.fitter.minimize.call_count, maxSteps)
        log.error.assert_called_with("testing failed to converge after %s steps" % maxSteps)

    def test_invalid_model(self):
        self.model.validate.return_value = False
        with(self.assertRaises(ValueError)):
            self.jointcal._logChi2AndValidate(self.associations, self.fitter, self.model)

    def test_nonfinite_chi2(self):
        self.fitter.computeChi2.return_value = self.nanChi2
        with(self.assertRaises(FloatingPointError)):
            self.jointcal._logChi2AndValidate(self.associations, self.fitter, self.model)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
