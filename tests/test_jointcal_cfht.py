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
import os

from astropy import units as u

import lsst.afw.geom
import lsst.utils
import lsst.pex.exceptions

import jointcalTestBase


# for MemoryTestCase
def setup_module(module):
    lsst.utils.tests.init()


class JointcalTestCFHT(jointcalTestBase.JointcalTestBase, lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.data_dir = lsst.utils.getPackageDir('testdata_jointcal')
            os.environ['ASTROMETRY_NET_DATA_DIR'] = os.path.join(cls.data_dir, 'cfht_and_index')
        except lsst.pex.exceptions.NotFoundError:
            raise unittest.SkipTest("testdata_jointcal not setup")

    def setUp(self):
        # We don't want the absolute astrometry to become significantly worse
        # than the single-epoch astrometry (about 0.040").
        # See Readme for an explanation of this empirical value.
        self.dist_rms_absolute = 48.6e-3*u.arcsecond

        do_plot = False

        # center of the cfht validation_data catalog
        center = lsst.afw.geom.SpherePoint(214.884832, 52.6622199, lsst.afw.geom.degrees)
        radius = 3*lsst.afw.geom.degrees

        input_dir = os.path.join(self.data_dir, 'cfht')
        all_visits = [849375, 850587]

        self.setUp_base(center, radius,
                        input_dir=input_dir,
                        all_visits=all_visits,
                        do_plot=do_plot,
                        log_level="DEBUG")

    def test_jointcalTask_2_visits(self):
        self.config = lsst.jointcal.jointcal.JointcalConfig()
        self.config.astrometryModel = "simple"
        self.config.photometryModel = "simpleFlux"
        self.config.sourceSelector['astrometry'].badFlags.append("base_PixelFlags_flag_interpolated")

        # to test whether we got the expected chi2 contribution files.
        self.other_args.extend(['--config', 'writeChi2ContributionFiles=True'])

        # See Readme for an explanation of these empirical values.
        dist_rms_relative = 11e-3*u.arcsecond
        pa1 = 0.014
        metrics = {'collected_astrometry_refStars': 825,
                   'collected_photometry_refStars': 825,
                   'selected_astrometry_refStars': 350,
                   'selected_photometry_refStars': 350,
                   'associated_astrometry_fittedStars': 2269,
                   'associated_photometry_fittedStars': 2269,
                   'selected_astrometry_fittedStars': 1239,
                   'selected_photometry_fittedStars': 1239,
                   'selected_astrometry_ccdImages': 12,
                   'selected_photometry_ccdImages': 12,
                   'astrometry_final_chi2': 1150.62,
                   'astrometry_final_ndof': 2550,
                   'photometry_final_chi2': 2824.86,
                   'photometry_final_ndof': 1388
                   }

        self._testJointcalTask(2, dist_rms_relative, self.dist_rms_absolute, pa1, metrics=metrics)

        # Check for the existence of the chi2 contribution files.
        expected = ['photometry_initial_chi2-0_r', 'astrometry_initial_chi2-0_r',
                    'photometry_final_chi2-0_r', 'astrometry_final_chi2-0_r']
        for partial in expected:
            name = partial+'-ref.csv'
            self.assertTrue(os.path.exists(name), msg="Did not find file %s"%name)
            os.remove(name)
            name = partial+'-meas.csv'
            self.assertTrue(os.path.exists(name), msg='Did not find file %s'%name)
            os.remove(name)

    def setup_jointcalTask_2_visits_constrainedAstrometry(self):
        """Help keep the constrainedAstrometry tests consistent and make
        the differences between them more obvious.
        """
        self.config = lsst.jointcal.jointcal.JointcalConfig()
        self.config.astrometryModel = "constrained"
        self.config.doPhotometry = False
        self.config.sourceSelector['astrometry'].badFlags.append("base_PixelFlags_flag_interpolated")
        self.jointcalStatistics.do_photometry = False

        # See Readme for an explanation of these empirical values.
        dist_rms_relative = 12e-3*u.arcsecond
        metrics = {'collected_astrometry_refStars': 825,
                   'selected_astrometry_refStars': 350,
                   'associated_astrometry_fittedStars': 2269,
                   'selected_astrometry_fittedStars': 1239,
                   'selected_astrometry_ccdImages': 12,
                   'astrometry_final_chi2': 1253.80,
                   'astrometry_final_ndof': 2660,
                   }

        return dist_rms_relative, metrics

    def test_jointcalTask_2_visits_constrainedAstrometry_no_photometry(self):
        dist_rms_relative, metrics = self.setup_jointcalTask_2_visits_constrainedAstrometry()
        self._testJointcalTask(2, dist_rms_relative, self.dist_rms_absolute, None, metrics=metrics)

    def test_jointcalTask_2_visits_constrainedAstrometry_no_rank_update(self):
        """Demonstrate that skipping the rank update doesn't affect astrometry.
        """
        relative_error, metrics = self.setup_jointcalTask_2_visits_constrainedAstrometry()
        self.config.astrometryDoRankUpdate = False

        self._testJointcalTask(2, relative_error, self.dist_rms_absolute, None, metrics=metrics)

    def test_jointcalTask_2_visits_constrainedAstrometry_4sigma_outliers(self):
        """4 sigma outlier rejection means fewer available sources after the
        fitter converges, resulting in a smaller ndof and chi2.
        """
        dist_rms_relative, metrics = self.setup_jointcalTask_2_visits_constrainedAstrometry()
        self.config.outlierRejectSigma = 4
        metrics['astrometry_final_chi2'] = 922.76
        metrics['astrometry_final_ndof'] = 2486

        self._testJointcalTask(2, dist_rms_relative, self.dist_rms_absolute, None, metrics=metrics)

    def setup_jointcalTask_2_visits_constrainedPhotometry(self):
        """Help keep the constrainedPhotometry tests consistent and make
        the differences between them more obvious.
        """
        self.config = lsst.jointcal.jointcal.JointcalConfig()
        self.config.photometryModel = "constrainedFlux"
        self.config.doAstrometry = False
        self.config.sourceSelector['astrometry'].badFlags.append("base_PixelFlags_flag_interpolated")
        self.jointcalStatistics.do_astrometry = False

        # See Readme for an explanation of these empirical values.
        pa1 = 0.017
        metrics = {'collected_photometry_refStars': 825,
                   'selected_photometry_refStars': 350,
                   'associated_photometry_fittedStars': 2269,
                   'selected_photometry_fittedStars': 1239,
                   'selected_photometry_ccdImages': 12,
                   'photometry_final_chi2': 2655.86,
                   'photometry_final_ndof': 1328
                   }
        return pa1, metrics

    def test_jointcalTask_2_visits_constrainedPhotometry_no_astrometry(self):
        pa1, metrics = self.setup_jointcalTask_2_visits_constrainedPhotometry()

        self._testJointcalTask(2, None, None, pa1, metrics=metrics)

    def test_jointcalTask_2_visits_constrainedPhotometry_no_rank_update(self):
        """Demonstrate that skipping the rank update doesn't affect photometry.
        """
        pa1, metrics = self.setup_jointcalTask_2_visits_constrainedPhotometry()
        self.config.photometryDoRankUpdate = False

        self._testJointcalTask(2, None, None, pa1, metrics=metrics)

    def test_jointcalTask_2_visits_constrainedPhotometry_lineSearch(self):
        """Activating the line search should only slightly change the chi2 in this case."""
        pa1, metrics = self.setup_jointcalTask_2_visits_constrainedPhotometry()
        self.config.allowLineSearch = True

        # Only this value should differ from the metrics defined in setup above.
        metrics['photometry_final_chi2'] = 2642.47

        self._testJointcalTask(2, None, None, pa1, metrics=metrics)

    def test_jointcalTask_2_visits_constrainedPhotometry_flagged(self):
        """Test the use of the FlaggedSourceSelector."""
        pa1, metrics = self.setup_jointcalTask_2_visits_constrainedPhotometry()
        self.config.sourceSelector.name = "flagged"
        # Calib flag names changed with RFC-498 (DM-14997).  The following sets the config to use the
        # old names associated with the current data in testdata_jointcal that was processed pre-RFC-498.
        # Remove line if the data in testdata_jointcal are ever reprocessed post-RFC-498.
        self.config.sourceSelector.active.field = "calib_psfUsed"
        # Reduce warnings due to flaggedSourceSelector having fewer sources than astrometrySourceSelector.
        self.config.minMeasuredStarsPerCcd = 30
        self.config.minRefStarsPerCcd = 20

        pa1 = 0.026
        metrics['selected_photometry_refStars'] = 212
        metrics['associated_photometry_fittedStars'] = 270
        metrics['selected_photometry_fittedStars'] = 244
        metrics['photometry_final_chi2'] = 369.964
        metrics['photometry_final_ndof'] = 252

        self._testJointcalTask(2, None, None, pa1, metrics=metrics)

    def test_jointcalTask_2_visits_constrainedMagnitude_no_astrometry(self):
        pa1, metrics = self.setup_jointcalTask_2_visits_constrainedPhotometry()
        self.config.photometryModel = "constrainedMagnitude"

        metrics['photometry_final_chi2'] = 2597.4
        metrics['photometry_final_ndof'] = 1326

        self._testJointcalTask(2, None, None, pa1, metrics=metrics)

    def test_jointcalTask_2_visits_constrainedFlux_pedestal(self):
        pa1, metrics = self.setup_jointcalTask_2_visits_constrainedPhotometry()
        self.config.photometryErrorPedestal = 0.02

        # We're allowing more error in the fit, so PA1 may be worse.
        pa1 = 0.018
        # Final chi2 is much lower, because all sources contribute more error.
        metrics['photometry_final_chi2'] = 1800.08
        # ndof shouldn't change much; slightly different likelihood contours
        metrics['photometry_final_ndof'] = 1338

        self._testJointcalTask(2, None, None, pa1, metrics=metrics)

    def test_jointcalTask_2_visits_constrainedMagnitude_pedestal(self):
        pa1, metrics = self.setup_jointcalTask_2_visits_constrainedPhotometry()
        self.config.photometryModel = "constrainedMagnitude"
        self.config.photometryErrorPedestal = 0.02

        # We're allowing more error in the fit, so PA1 may be worse.
        pa1 = 0.019
        # Final chi2 is much lower, because all sources contribute more error.
        metrics['photometry_final_chi2'] = 1632.83
        # ndof shouldn't change much; slightly different likelihood contours
        metrics['photometry_final_ndof'] = 1322

        self._testJointcalTask(2, None, None, pa1, metrics=metrics)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
