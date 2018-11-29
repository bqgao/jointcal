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

import abc

import numpy as np

import unittest
import lsst.utils.tests

import lsst.afw.geom
from lsst.jointcal import photometryTransfo


CHEBYSHEV_T = [
    lambda x: 1,
    lambda x: x,
    lambda x: 2*x**2 - 1,
    lambda x: (4*x**2 - 3)*x,
    lambda x: (8*x**2 - 8)*x**2 + 1,
    lambda x: ((16*x**2 - 20)*x**2 + 5)*x,
]


class PhotometryTransfoTestBase:
    def setUp(self):
        self.value = 5.0
        self.valueError = 0.3
        self.point = [1., 5.]


class SpatiallyInvariantTestBase(PhotometryTransfoTestBase):
    """Tests for PhotometryTransfoSpatiallyInvariant.
     Subclasses need to call setUp to define:
         self.transfo1 == a default initalized PhotometryTransfoSpatiallyInvariant.
         self.transfo2 == a transfo initialized with self.t2InitValue.
    """
    def setUp(self):
        super().setUp()
        # initial values for self.transfo2
        self.t2InitValue = 1000.0
        self.t2InitError = 70.0

    def _test_transform(self, transfo, expect):
        result = transfo.transform(self.point[0], self.point[1], self.value)
        self.assertEqual(result, expect)  # yes, I really mean exactly equal

    def _test_transformError(self, transfo, expect):
        result = transfo.transformError(self.point[0], self.point[1], self.value, self.valueError)
        self.assertFloatsAlmostEqual(result, expect)

    def _offsetParams(self, delta, value, expect):
        self.transfo1.offsetParams(delta)
        result = self.transfo1.transform(self.point[0], self.point[1], value)
        self.assertFloatsAlmostEqual(result, expect)

    def _test_offsetParams(self, expect):
        """Test offsetting; note that offsetParams offsets by +1."""
        # check that offset by 0 doesn't change anything.
        delta = np.zeros(1, dtype=float)
        self._offsetParams(delta, self.value, self.value)

        # offset by +1 should result in `expect`
        delta -= 1
        self._offsetParams(delta, self.value, expect)

    def test_clone(self):
        clone1 = self.transfo1.clone()
        self.assertEqual(self.transfo1.getParameters(), clone1.getParameters())
        clone2 = self.transfo2.clone()
        self.assertEqual(self.transfo2.getParameters(), clone2.getParameters())
        self.assertNotEqual(clone1.getParameters(), clone2.getParameters())

    def _test_computeParameterDerivatives(self, expect):
        """The derivative of a spatially invariant transform is always the same.
        Should be indepdendent of position
        """
        result = self.transfo1.computeParameterDerivatives(1, 2, self.value)
        self.assertEqual(expect, result)
        result = self.transfo1.computeParameterDerivatives(-5, -100, self.value)
        self.assertEqual(expect, result)
        result = self.transfo2.computeParameterDerivatives(-1000, 150, self.value)
        self.assertEqual(expect, result)


class FluxTransfoSpatiallyInvariantTestCase(SpatiallyInvariantTestBase, lsst.utils.tests.TestCase):
    def setUp(self):
        super().setUp()
        self.transfo1 = photometryTransfo.FluxTransfoSpatiallyInvariant()
        self.transfo2 = photometryTransfo.FluxTransfoSpatiallyInvariant(self.t2InitValue)

    def test_transform(self):
        self._test_transform(self.transfo1, self.value)
        self._test_transform(self.transfo2, self.value*self.t2InitValue)

    def test_transformError(self):
        expect = (self.valueError*1)
        self._test_transformError(self.transfo1, expect)
        expect = (self.valueError*self.t2InitValue)
        self._test_transformError(self.transfo2, expect)

    def test_offsetParams(self):
        """Offset by +1 means transform by 2."""
        self._test_offsetParams(self.value*2)

    def test_computeParameterDerivatives(self):
        """Should be indepdendent of position, and equal to the flux."""
        self._test_computeParameterDerivatives(self.value)


class MagnitudeTransfoSpatiallyInvariantTestCase(SpatiallyInvariantTestBase, lsst.utils.tests.TestCase):
    def setUp(self):
        super().setUp()
        self.transfo1 = photometryTransfo.MagnitudeTransfoSpatiallyInvariant()
        self.transfo2 = photometryTransfo.MagnitudeTransfoSpatiallyInvariant(self.t2InitValue)

    def test_transform(self):
        self._test_transform(self.transfo1, self.value)
        self._test_transform(self.transfo2, self.value + self.t2InitValue)

    def test_transformError(self):
        expect = self.valueError
        self._test_transformError(self.transfo1, expect)
        expect = self.valueError
        self._test_transformError(self.transfo2, expect)

    def test_offsetParams(self):
        """Offset by +1 means transform by +1."""
        self._test_offsetParams(self.value + 1)

    def test_computeParameterDerivatives(self):
        """Should always be identically 1."""
        self._test_computeParameterDerivatives(1.0)


class PhotometryTransfoChebyshevTestCase(PhotometryTransfoTestBase, abc.ABC):
    def setUp(self):
        """Call this first, then construct self.transfo1 from self.order1,
        and self.transfo2 from self.coefficients.
        """
        super().setUp()
        self.bbox = lsst.afw.geom.Box2D(lsst.afw.geom.Point2D(-5, -6), lsst.afw.geom.Point2D(7, 8))
        self.order1 = 2
        self.coefficients = np.array([[5, 3], [4, 0]], dtype=float)

        # self.transfo1 will have 6 parameters, by construction
        self.delta = np.arange(6, dtype=float)
        # make one of them have opposite sign to check +/- consistency
        self.delta[0] = -self.delta[0]

    def test_getNpar(self):
        self.assertEqual(self.transfo1.getNpar(), 6)
        self.assertEqual(self.transfo2.getNpar(), 3)

    def _evaluate_chebyshev(self, x, y):
        """Evaluate the chebyshev defined by self.coefficients at (x,y)"""
        # sx, sy: transform from self.bbox range to [-1, -1]
        cx = (self.bbox.getMinX() + self.bbox.getMaxX())/2.0
        cy = (self.bbox.getMinY() + self.bbox.getMaxY())/2.0
        sx = 2.0 / self.bbox.getWidth()
        sy = 2.0 / self.bbox.getHeight()
        result = 0
        order = len(self.coefficients)
        for j in range(order):
            for i in range(0, order-j):
                Tx = CHEBYSHEV_T[i](sx*(x - cx))
                Ty = CHEBYSHEV_T[j](sy*(y - cy))
                result += self.coefficients[j, i]*Tx*Ty
        return result

    def _test_offsetParams(self, expect):
        """Test offsetting; note that offsetParams offsets by `-delta`.

        Parameters
        ----------
        expect1 : `numpy.ndarray`, (N,2)
            Expected coefficients from an offset by 0.
        expect2 : `numpy.ndarray`, (N,2)
            Expected coefficients from an offset by self.delta.
        """
        # first offset by all zeros: nothing should change
        delta = np.zeros(self.transfo1.getNpar(), dtype=float)
        self.transfo1.offsetParams(delta)
        self.assertFloatsAlmostEqual(expect, self.transfo1.getCoefficients())

        # now offset by self.delta
        expect[0, 0] -= self.delta[0]
        expect[0, 1] -= self.delta[1]
        expect[0, 2] -= self.delta[2]
        expect[1, 0] -= self.delta[3]
        expect[1, 1] -= self.delta[4]
        expect[2, 0] -= self.delta[5]
        self.transfo1.offsetParams(self.delta)
        self.assertFloatsAlmostEqual(expect, self.transfo1.getCoefficients())

    def test_clone(self):
        clone1 = self.transfo1.clone()
        self.assertFloatsEqual(self.transfo1.getParameters(), clone1.getParameters())
        self.assertEqual(self.transfo1.getOrder(), clone1.getOrder())
        self.assertEqual(self.transfo1.getBBox(), clone1.getBBox())
        clone2 = self.transfo2.clone()
        self.assertFloatsEqual(self.transfo2.getParameters(), clone2.getParameters())
        self.assertEqual(self.transfo2.getOrder(), clone2.getOrder())
        self.assertEqual(self.transfo2.getBBox(), clone2.getBBox())

    @abc.abstractmethod
    def _computeChebyshevDerivative(self, Tx, Ty, value):
        """Return the derivative of chebyshev component Tx, Ty."""
        pass

    def test_computeParameterDerivatives(self):
        cx = (self.bbox.getMinX() + self.bbox.getMaxX())/2.0
        cy = (self.bbox.getMinY() + self.bbox.getMaxY())/2.0
        sx = 2.0 / self.bbox.getWidth()
        sy = 2.0 / self.bbox.getHeight()
        result = self.transfo1.computeParameterDerivatives(self.point[0], self.point[1], self.value)
        Tx = np.array([CHEBYSHEV_T[i](sx*(self.point[0] - cx)) for i in range(self.order1+1)], dtype=float)
        Ty = np.array([CHEBYSHEV_T[i](sy*(self.point[1] - cy)) for i in range(self.order1+1)], dtype=float)
        expect = []
        for j in range(len(Ty)):
            for i in range(0, self.order1-j+1):
                expect.append(self._computeChebyshevDerivative(Ty[j], Tx[i], self.value))
        self.assertFloatsAlmostEqual(np.array(expect), result)

    def testIntegrateBox(self):
        r"""Test integrating over an "interesting" box.

        The values of these integrals were checked in Mathematica. The code
        block below can be pasted into Mathematica to re-do those calculations.

        .. code-block:: mathematica

            f[x_, y_, n_, m_] := \!\(
            \*UnderoverscriptBox[\(\[Sum]\), \(i = 0\), \(n\)]\(
            \*UnderoverscriptBox[\(\[Sum]\), \(j = 0\), \(m\)]
            \*SubscriptBox[\(a\), \(i, j\)]*ChebyshevT[i, x]*ChebyshevT[j, y]\)\)
            integrate2dBox[n_, m_, xmin_, xmax_, ymin_, ymax_, x0_, x1_, y0_,
              y1_] := \!\(
            \*SubsuperscriptBox[\(\[Integral]\), \(y0\), \(y1\)]\(
            \*SubsuperscriptBox[\(\[Integral]\), \(x0\), \(x1\)]f[
            \*FractionBox[\(2  x - xmin - xmax\), \(xmax - xmin\)],
            \*FractionBox[\(2  y - ymin - ymax\), \(ymax - ymin\)], n,
                 m] \[DifferentialD]x \[DifferentialD]y\)\)
            integrate2dBox[0, 0, -5, 7, -6, 8, 0, 7, 0, 8]
            integrate2dBox[0, 0, -5, 7, -6, 8, 2, 6, 3, 5]
            # integrate2dBox[1, 0, -5, 7, -6, 8, 0, 6, 0, 5]
            # integrate2dBox[0, 1, -5, 7, -6, 8, 0, 6, 0, 5]
            integrate2dBox[1, 1, -5, 7, -6, 8, -1, 5., 2, 7]
            integrate2dBox[2, 2, -5, 7, -6, 8, 0, 2, 0, 3]
        """
        # for 0th order
        coeffs = np.array([[3.]], dtype=float)
        transform = photometryTransfo.FluxTransfoChebyshev(coeffs, self.bbox)

        # a box that goes to the x/y maximum
        box = lsst.geom.Box2D(lsst.geom.Point2D(0, 0),
                              lsst.geom.Point2D(self.bbox.getMaxX(), self.bbox.getMaxY()))
        expect = 56*coeffs[0]
        result = transform.integrate(box)
        self.assertFloatsAlmostEqual(result, expect)

        box = lsst.geom.Box2D(lsst.geom.Point2D(2, 3), lsst.geom.Point2D(6, 5))
        expect = 8*coeffs[0]
        result = transform.integrate(box)
        self.assertFloatsAlmostEqual(result, expect)

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # box = lsst.geom.Box2D(lsst.geom.Point2D(0, 0), lsst.geom.Point2D(6, 5))
        # # 1st order in x:
        # coeffs = np.array([[2.], [5.]], dtype=float)
        # transform = photometryTransfo.FluxTransfoChebyshev(coeffs, self.bbox)
        # # transform.offsetParams([[2., 5.], [0, 0]])
        # # 30*a00 + 10*a10
        # expect = 30*coeffs[0, 0] + 10*coeffs[1, 0]
        # result = transform.integrate(box)
        # self.assertFloatsAlmostEqual(result, expect)
        # # 1st order in y:
        # transform = photometryTransfo.FluxTransfoChebyshev(1, self.bbox)
        # transform.offsetParams([2., 0., 5.])
        # # 30*a00 + 45/7*a01
        # expect = 30*coeffs[0, 0] + 45./7.*coeffs[0, 1]
        # result = transform.integrate(box)
        # self.assertFloatsAlmostEqual(result, expect)

        # 1st order in both x and y
        transform = photometryTransfo.FluxTransfoChebyshev(2, self.bbox)
        # zero, then set the parameters
        transform.offsetParams(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float))
        coeffs = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]], dtype=float)
        transform.offsetParams(-coeffs.flatten())

        # integrating on the full box should match the standard integral
        expect = transform.integrate()
        result = transform.integrate(self.bbox)
        self.assertFloatsAlmostEqual(result, expect)

        # integrate on the smaller box:
        box = lsst.geom.Box2D(lsst.geom.Point2D(-1, 2), lsst.geom.Point2D(5, 7))
        # 3/8*(56*a00 + 20*a0,1 + 14*a1,0 + 5*a11)
        expect = 3.0/8.0*(56*coeffs[0, 0] + 20*coeffs[0, 1] + 14*coeffs[1, 0] + 5*coeffs[1, 1])

        import os; print(os.getpid()); import ipdb; ipdb.set_trace();
        result = transform.integrate(box)

        # self.assertFloatsAlmostEqual(result, expect)

        # for 2nd order in both x and y
        box = lsst.geom.Box2D(lsst.geom.Point2D(0, 0), lsst.geom.Point2D(2, 3))
        coeffs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        transform = photometryTransfo.FluxTransfoChebyshev(2, self.bbox)
        # zero, then set the parameters
        transform.offsetParams(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float))
        transform.offsetParams(-coeffs.flatten())
        # 1/882 * (5292*a00 + 378*a01 - 5076*a02 - 5194*a20 - 371*a21 + 4982*a22)
        expect = 1./882. * (5292*coeffs[0, 0] + 378*coeffs[0, 1] -
                            5076*coeffs[0, 2] - 5194*coeffs[2, 0] -
                            371*coeffs[2, 1] + 4982*coeffs[2, 2])
        result = transform.integrate(box)
        self.assertFloatsAlmostEqual(result, expect)


class FluxTransfoChebyshevTestCase(PhotometryTransfoChebyshevTestCase, lsst.utils.tests.TestCase):
    def setUp(self):
        super().setUp()
        self.transfo1 = photometryTransfo.FluxTransfoChebyshev(self.order1, self.bbox)
        self.transfo2 = photometryTransfo.FluxTransfoChebyshev(self.coefficients, self.bbox)

    def test_transform(self):
        result = self.transfo1.transform(self.point[0], self.point[1], self.value)
        self.assertEqual(result, self.value)  # transfo1 is the identity

        result = self.transfo2.transform(self.point[0], self.point[1], self.value)
        expect = self.value*self._evaluate_chebyshev(self.point[0], self.point[1])
        self.assertEqual(result, expect)

    def test_offsetParams(self):
        # an offset by 0 means we will still have 1 only in the 0th parameter
        expect = np.zeros((self.order1+1, self.order1+1), dtype=float)
        expect[0, 0] = 1
        self._test_offsetParams(expect)

    def _computeChebyshevDerivative(self, x, y, value):
        return x * y * value


class MagnitudeTransfoChebyshevTestCase(PhotometryTransfoChebyshevTestCase, lsst.utils.tests.TestCase):
    def setUp(self):
        super().setUp()
        self.transfo1 = photometryTransfo.MagnitudeTransfoChebyshev(self.order1, self.bbox)
        self.transfo2 = photometryTransfo.MagnitudeTransfoChebyshev(self.coefficients, self.bbox)

    def test_transform(self):
        result = self.transfo1.transform(self.point[0], self.point[1], self.value)
        self.assertEqual(result, self.value)  # transfo1 is the identity

        result = self.transfo2.transform(self.point[0], self.point[1], self.value)
        expect = self.value + self._evaluate_chebyshev(self.point[0], self.point[1])
        self.assertEqual(result, expect)

    def test_offsetParams(self):
        # an offset by 0 means all parameters still 0
        expect = np.zeros((self.order1+1, self.order1+1), dtype=float)
        self._test_offsetParams(expect)

    def _computeChebyshevDerivative(self, x, y, value):
        return x * y


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
