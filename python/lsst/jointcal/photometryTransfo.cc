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

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "ndarray/pybind11.h"
#include "ndarray/eigen.h"
#include "Eigen/Core"

#include "lsst/utils/python.h"

#include "lsst/jointcal/PhotometryTransfo.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace jointcal {
namespace {

void declarePhotometryTransfo(py::module &mod) {
    py::class_<PhotometryTransfo, std::shared_ptr<PhotometryTransfo>> cls(mod, "PhotometryTransfo");

    cls.def("transform",
            (double (PhotometryTransfo::*)(double, double, double) const) & PhotometryTransfo::transform,
            "x"_a, "y"_a, "value"_a);
    cls.def("transformError",
            (double (PhotometryTransfo::*)(double, double, double, double) const) &
                    PhotometryTransfo::transformError,
            "x"_a, "y"_a, "value"_a, "valueErr"_a);
    cls.def("offsetParams", &PhotometryTransfo::offsetParams);
    cls.def("clone", &PhotometryTransfo::clone);
    cls.def("getNpar", &PhotometryTransfo::getNpar);
    cls.def("getParameters", &PhotometryTransfo::getParameters);
    cls.def("computeParameterDerivatives",
            [](PhotometryTransfo const &self, double x, double y, double instFlux) {
                Eigen::VectorXd derivatives(self.getNpar());
                self.computeParameterDerivatives(x, y, instFlux, derivatives);
                return derivatives;
            });

    utils::python::addOutputOp(cls, "__str__");
    utils::python::addOutputOp(cls, "__repr__");
}

void declarePhotometryTransfoSpatiallyInvariant(py::module &mod) {
    py::class_<PhotometryTransfoSpatiallyInvariant, std::shared_ptr<PhotometryTransfoSpatiallyInvariant>,
               PhotometryTransfo>
            cls(mod, "PhotometryTransfoSpatiallyInvariant");
}

void declareFluxTransfoSpatiallyInvariant(py::module &mod) {
    py::class_<FluxTransfoSpatiallyInvariant, std::shared_ptr<FluxTransfoSpatiallyInvariant>,
               PhotometryTransfoSpatiallyInvariant, PhotometryTransfo>
            cls(mod, "FluxTransfoSpatiallyInvariant");

    cls.def(py::init<double>(), "value"_a = 1);
}

void declareMagnitudeTransfoSpatiallyInvariant(py::module &mod) {
    py::class_<MagnitudeTransfoSpatiallyInvariant, std::shared_ptr<MagnitudeTransfoSpatiallyInvariant>,
               PhotometryTransfoSpatiallyInvariant, PhotometryTransfo>
            cls(mod, "MagnitudeTransfoSpatiallyInvariant");

    cls.def(py::init<double>(), "value"_a = 0);
}

void declarePhotometryTransfoChebyshev(py::module &mod) {
    py::class_<PhotometryTransfoChebyshev, std::shared_ptr<PhotometryTransfoChebyshev>, PhotometryTransfo>
            cls(mod, "PhotometryTransfoChebyshev");

    cls.def("getCoefficients", &PhotometryTransfoChebyshev::getCoefficients);
    cls.def("getOrder", &PhotometryTransfoChebyshev::getOrder);
    cls.def("getBBox", &PhotometryTransfoChebyshev::getBBox);
    cls.def("integrate", py::overload_cast<>(&PhotometryTransfoChebyshev::integrate, py::const_));
    cls.def("integrate",
            py::overload_cast<afw::geom::Box2D const &>(&PhotometryTransfoChebyshev::integrate, py::const_),
            "box"_a);
}

void declareFluxTransfoChebyshev(py::module &mod) {
    py::class_<FluxTransfoChebyshev, std::shared_ptr<FluxTransfoChebyshev>, PhotometryTransfoChebyshev> cls(
            mod, "FluxTransfoChebyshev");

    cls.def(py::init<size_t, afw::geom::Box2D const &>(), "order"_a, "bbox"_a);
    cls.def(py::init<ndarray::Array<double, 2, 2> const &, afw::geom::Box2D const &>(), "coefficients"_a,
            "bbox"_a);
}

void declareMagnitudeTransfoChebyshev(py::module &mod) {
    py::class_<MagnitudeTransfoChebyshev, std::shared_ptr<MagnitudeTransfoChebyshev>,
               PhotometryTransfoChebyshev>
            cls(mod, "MagnitudeTransfoChebyshev");

    cls.def(py::init<size_t, afw::geom::Box2D const &>(), "order"_a, "bbox"_a);
    cls.def(py::init<ndarray::Array<double, 2, 2> const &, afw::geom::Box2D const &>(), "coefficients"_a,
            "bbox"_a);
}

PYBIND11_MODULE(photometryTransfo, mod) {
    declarePhotometryTransfo(mod);

    declarePhotometryTransfoSpatiallyInvariant(mod);
    declareFluxTransfoSpatiallyInvariant(mod);
    declareMagnitudeTransfoSpatiallyInvariant(mod);
    declarePhotometryTransfoChebyshev(mod);
    declareFluxTransfoChebyshev(mod);
    declareMagnitudeTransfoChebyshev(mod);
}

}  // namespace
}  // namespace jointcal
}  // namespace lsst
