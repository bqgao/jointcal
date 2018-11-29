// -*- LSST-C++ -*-
/*
 * This file is part of jointcal.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "ndarray.h"
#include "Eigen/Core"

#include "lsst/afw/math/detail/TrapezoidalPacker.h"

#include "lsst/jointcal/Point.h"
#include "lsst/jointcal/PhotometryTransfo.h"

namespace lsst {
namespace jointcal {

// ------------------ PhotometryTransfoChebyshev helpers ---------------------------------------------------

namespace {

// To evaluate a 1-d Chebyshev function without needing to have workspace, we use the
// Clenshaw algorith, which is like going through the recurrence relation in reverse.
// The CoeffGetter argument g is something that behaves like an array, providing access
// to the coefficients.
template <typename CoeffGetter>
double evaluateFunction1d(CoeffGetter g, double x, int size) {
    double b_kp2 = 0.0, b_kp1 = 0.0;
    for (int k = (size - 1); k > 0; --k) {
        double b_k = g[k] + 2 * x * b_kp1 - b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }
    return g[0] + x * b_kp1 - b_kp2;
}

// This class imitates a 1-d array, by running evaluateFunction1d on a nested dimension;
// this lets us reuse the logic in evaluateFunction1d for both dimensions.  Essentially,
// we run evaluateFunction1d on a column of coefficients to evaluate T_i(x), then pass
// the result of that to evaluateFunction1d with the results as the "coefficients" associated
// with the T_j(y) functions.
struct RecursionArrayImitator {
    double operator[](int i) const {
        return evaluateFunction1d(coefficients[i], x, coefficients.getSize<1>());
    }

    RecursionArrayImitator(ndarray::Array<double const, 2, 2> const &coefficients_, double x_)
            : coefficients(coefficients_), x(x_) {}

    ndarray::Array<double const, 2, 2> coefficients;
    double x;
};

// Compute an affine transform that maps an arbitrary box to [-1,1]x[-1,1]
lsst::geom::AffineTransform makeChebyshevRangeTransform(lsst::geom::Box2D const &bbox) {
    return lsst::geom::AffineTransform(
            lsst::geom::LinearTransform::makeScaling(2.0 / bbox.getWidth(), 2.0 / bbox.getHeight()),
            lsst::geom::Extent2D(-(2.0 * bbox.getCenterX()) / bbox.getWidth(),
                                 -(2.0 * bbox.getCenterY()) / bbox.getHeight()));
}

// Initialize a "unit" Chebyshev
ndarray::Array<double, 2, 2> _initializeChebyshev(size_t order, bool identity) {
    ndarray::Array<double, 2, 2> coeffs = ndarray::allocate(ndarray::makeVector(order + 1, order + 1));
    coeffs.deep() = 0.0;
    if (identity) {
        coeffs[0][0] = 1;
    }
    return coeffs;
}
}  // namespace

PhotometryTransfoChebyshev::PhotometryTransfoChebyshev(size_t order, lsst::geom::Box2D const &bbox,
                                                       bool identity)
        : _bbox(bbox),
          _toChebyshevRange(makeChebyshevRangeTransform(bbox)),
          _coefficients(_initializeChebyshev(order, identity)),
          _order(order),
          _nParameters((order + 1) * (order + 2) / 2) {}

PhotometryTransfoChebyshev::PhotometryTransfoChebyshev(ndarray::Array<double, 2, 2> const &coefficients,
                                                       lsst::geom::Box2D const &bbox)
        : _bbox(bbox),
          _toChebyshevRange(makeChebyshevRangeTransform(bbox)),
          _coefficients(coefficients),
          _order(coefficients.size() - 1),
          _nParameters((_order + 1) * (_order + 2) / 2) {}

void PhotometryTransfoChebyshev::offsetParams(Eigen::VectorXd const &delta) {
    // NOTE: the indexing in this method and computeParameterDerivatives must be kept consistent!
    Eigen::VectorXd::Index k = 0;
    for (ndarray::Size j = 0; j <= _order; ++j) {
        ndarray::Size const iMax = _order - j;  // to save re-computing `i+j <= order` every inner step.
        for (ndarray::Size i = 0; i <= iMax; ++i, ++k) {
            _coefficients[j][i] -= delta[k];
        }
    }
}

namespace {
// The integral of T_n(x) over [-1,1]:
// https://en.wikipedia.org/wiki/Chebyshev_polynomials#Differentiation_and_integration
double integrateTn(int n) {
    if (n % 2 == 1)
        return 0;
    else
        return 2.0 / (1.0 - static_cast<double>(n * n));
}
}  // namespace

double PhotometryTransfoChebyshev::oneIntegral(double x, double y) const {
    lsst::geom::Point2D point = _toChebyshevRange(lsst::geom::Point2D(x, y));

    // The nth/mth x/y chebyshev polynomial
    Eigen::VectorXd Tnx(_order + 2);
    Eigen::VectorXd Tmy(_order + 2);

    // Algorithm: compute all the individual components recursively (since we'll need them anyway),
    // then combine them into the final answer.
    Tnx[0] = 1;
    Tmy[0] = 1;
    Tnx[1] = point.getX();
    Tmy[1] = point.getY();
    for (ndarray::Size i = 2; i <= _order + 1; ++i) {
        Tnx[i] = 2 * point.getX() * Tnx[i - 1] - Tnx[i - 2];
        Tmy[i] = 2 * point.getY() * Tmy[i - 1] - Tmy[i - 2];
    }

    // NOTE: the indexing in this method and offsetParams must be kept consistent!
    double result = 0;
    result += _coefficients[0][0] * point.getX() * point.getY();
    std::cout << "x y: " << x << " " << y << "; " << point.getX() << " " << point.getY() << std::endl;
    std::cout << "0,0 " << result << std::endl;
    // NOTE:
    if (_order >= 1) {
        // integral of T1(x)T0(y)
        result += _coefficients[0][1] * 0.5 * Tnx[1] * point.getX() * point.getY();
        std::cout << "0,1 " << result << std::endl;
        // integral of T0(x)T1(y)
        result += _coefficients[1][0] * 0.5 * point.getX() * Tmy[1] * point.getY();
        std::cout << "1,0 " << result << std::endl;
    }
    // 1,1 is part of order >= 2, but can't be computed with the relation below.
    if (_order > 1) {
        // integral of T1(x)T1(y)
        result += _coefficients[1][1] * 0.5 * 0.5 * point.getX() * Tnx[1] * point.getY() * Tmy[1];
        std::cout << "1,1 " << result << std::endl;
    }
    for (ndarray::Size j = 2; j <= _order; ++j) {
        ndarray::Size const iMax = _order - j;  // to save re-computing `i+j <= order` every inner step.
        for (ndarray::Size i = 0; i <= iMax; ++i) {
            // n*T[n+1]/(n^2 - 1) - x*T[n]/(n-1)
            double tempy = j * Tmy[j + 1] / (j * j - 1) + point.getY() * Tmy[j] / (j - 1);
            double tempx = i * Tnx[i + 1] / (i * i - 1) + point.getX() * Tnx[i] / (i - 1);
            std::cout << i << " " << j << " " << tempx << " " << tempy << std::endl;
            result += _coefficients[i][j] * tempx * tempy;
            std::cout << i << "," << j << " " << result << std::endl;
        }
    }
    std::cout << _toChebyshevRange.getLinear().computeDeterminant() << std::endl;
    result /= _toChebyshevRange.getLinear().computeDeterminant();
    std::cout << result << " %%%%%%%%%%%%%%%%%%%%%" << std::endl;
    return result;
}

double PhotometryTransfoChebyshev::integrate(lsst::geom::Box2D const &bbox) const {
    double result = 0;

    result += oneIntegral(bbox.getMaxX(), bbox.getMaxY());
    result += oneIntegral(bbox.getMinX(), bbox.getMinY());
    result -= oneIntegral(bbox.getMaxX(), bbox.getMinY());
    result -= oneIntegral(bbox.getMinX(), bbox.getMaxY());

    std::cout << "result: " << result << std::endl;
    std::cout << "###############################################" << std::endl;

    return result;
}

double PhotometryTransfoChebyshev::integrate() const {
    double result = 0;
    double determinant = _bbox.getArea() / 4.0;
    for (ndarray::Size j = 0; j < _coefficients.getSize<0>(); j++) {
        for (ndarray::Size i = 0; i < _coefficients.getSize<1>(); i++) {
            result += _coefficients[j][i] * integrateTn(i) * integrateTn(j);
        }
    }
    return result * determinant;
}

double PhotometryTransfoChebyshev::mean(lsst::geom::Box2D const &bbox) const {
    return integrate(bbox) / bbox.getArea();
}

double PhotometryTransfoChebyshev::mean() const { return integrate() / _bbox.getArea(); }

Eigen::VectorXd PhotometryTransfoChebyshev::getParameters() const {
    Eigen::VectorXd parameters(_nParameters);
    // NOTE: the indexing in this method and offsetParams must be kept consistent!
    Eigen::VectorXd::Index k = 0;
    for (ndarray::Size j = 0; j <= _order; ++j) {
        ndarray::Size const iMax = _order - j;  // to save re-computing `i+j <= order` every inner step.
        for (ndarray::Size i = 0; i <= iMax; ++i, ++k) {
            parameters[k] = _coefficients[j][i];
        }
    }

    return parameters;
}

double PhotometryTransfoChebyshev::computeChebyshev(double x, double y) const {
    lsst::geom::Point2D p = _toChebyshevRange(lsst::geom::Point2D(x, y));
    return evaluateFunction1d(RecursionArrayImitator(_coefficients, p.getX()), p.getY(),
                              _coefficients.getSize<0>());
}

void PhotometryTransfoChebyshev::computeChebyshevDerivatives(double x, double y,
                                                             Eigen::Ref<Eigen::VectorXd> derivatives) const {
    lsst::geom::Point2D p = _toChebyshevRange(lsst::geom::Point2D(x, y));
    // Algorithm: compute all the individual components recursively (since we'll need them anyway),
    // then combine them into the final answer vectors.
    Eigen::VectorXd Tnx(_order + 1);
    Eigen::VectorXd Tmy(_order + 1);
    Tnx[0] = 1;
    Tmy[0] = 1;
    if (_order >= 1) {
        Tnx[1] = p.getX();
        Tmy[1] = p.getY();
    }
    for (ndarray::Size i = 2; i <= _order; ++i) {
        Tnx[i] = 2 * p.getX() * Tnx[i - 1] - Tnx[i - 2];
        Tmy[i] = 2 * p.getY() * Tmy[i - 1] - Tmy[i - 2];
    }

    // NOTE: the indexing in this method and offsetParams must be kept consistent!
    Eigen::VectorXd::Index k = 0;
    for (ndarray::Size j = 0; j <= _order; ++j) {
        ndarray::Size const iMax = _order - j;  // to save re-computing `i+j <= order` every inner step.
        for (ndarray::Size i = 0; i <= iMax; ++i, ++k) {
            derivatives[k] = Tmy[j] * Tnx[i];
        }
    }
}

}  // namespace jointcal
}  // namespace lsst
