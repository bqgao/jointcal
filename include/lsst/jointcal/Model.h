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

#ifndef LSST_JOINTCAL_MODEL_H
#define LSST_JOINTCAL_MODEL_H

#include <string>
#include <vector>

#include "lsst/log/Log.h"

#include "lsst/jointcal/CcdImage.h"
#include "lsst/jointcal/FittedStar.h"
#include "lsst/jointcal/MeasuredStar.h"
#include "lsst/jointcal/RefStar.h"

namespace lsst {
namespace jointcal {

class Model {
public:
    Model(LOG_LOGGER log, double errorPedestal_ = 0) : _log(log), errorPedestal(errorPedestal_) {}

    /**
     * Assign indices in the full matrix to the parameters being fit in the mappings, starting at firstIndex.
     *
     * @param[in]  whatToFit   String containing parameters to fit.
     * @param[in]  firstIndex  Index to start assigning at.
     *
     * @return     The highest assigned index.
     */
    virtual unsigned assignIndices(std::string const &whatToFit, unsigned firstIndex) = 0;

    /**
     * Offset the parameters by the provided amounts (by -delta).
     *
     * The shifts are applied according to the indices given in assignIndices.
     *
     * @param[in]  delta  vector of offsets to apply
     */
    virtual void offsetParams(Eigen::VectorXd const &delta) = 0;

    // NO??
    /**
     * Offset the appropriate flux or magnitude (by -delta).
     *
     * @param fittedStar The star to update.
     * @param delta The amount to update by.
     */
    virtual void offsetFittedStar(FittedStar &fittedStar, double delta) const = 0;

    /**
     * Compute the residual between the model applied to a star and its associated fittedStar.
     *
     * @f[
     *     residual = Model(measuredStar) - fittedStar
     * @f]
     *
     * @param ccdImage The ccdImage where measuredStar resides.
     * @param measuredStar The measured star position to compute the residual of.
     *
     * @return The residual.
     */
    virtual double computeResidual(CcdImage const &ccdImage, MeasuredStar const &measuredStar) const = 0;

    // NO??
    /**
     * Return the on-sky transformed flux for measuredStar on ccdImage.
     *
     * @param[in]  ccdImage     The ccdImage where measuredStar resides.
     * @param[in]  measuredStar The measured star to transform.
     *
     * @return     The on-sky flux transformed from instFlux at measuredStar's position.
     */
    virtual double transform(CcdImage const &ccdImage, MeasuredStar const &measuredStar) const = 0;

    // NO??
    /**
     * Return the on-sky transformed flux uncertainty for measuredStar on ccdImage.
     * Identical to transform() until freezeErrorTransform() is called.
     *
     * @param[in]  ccdImage     The ccdImage where measuredStar resides.
     * @param[in]  measuredStar The measured star to transform.
     *
     * @return     The on-sky flux transformed from instFlux at measuredStar's position.
     */
    virtual double transformError(CcdImage const &ccdImage, MeasuredStar const &measuredStar) const = 0;

    /**
     * Once this routine has been called, the error transform is not modified by offsetParams().
     *
     * The routine can be called when the mappings are roughly in place. After the call, the transformations
     * used to propagate errors are no longer affected when updating the mappings. This allows an exactly
     * linear fit, which can be necessary for some model+data combinations.
     */
    virtual void freezeErrorTransform() = 0;

    /**
     * Get how this set of parameters (of length Npar()) map into the "grand" fit.
     *
     * @param[in]  ccdImage  The ccdImage to look up.
     * @param[out] indices   The indices of the mapping associated with ccdImage.
     */
    virtual void getMappingIndices(CcdImage const &ccdImage, std::vector<unsigned> &indices) const = 0;

    /**
     * Compute the parametric derivatives of this model.
     *
     * @param[in]   measuredStar  The measured star with the position and flux to compute at.
     * @param[in]   ccdImage      The ccdImage containing the measured star, to find the correct mapping.
     * @param[out]  derivatives   The computed derivatives. Must be pre-allocated to the correct size.
     */
    virtual void computeParameterDerivatives(MeasuredStar const &measuredStar, CcdImage const &ccdImage,
                                             Eigen::VectorXd &derivatives) const = 0;

    // NO?? (err on astrometry should also be double, no?)
    /// Return the refStar error appropriate for this model (e.g. fluxErr or magErr).
    virtual double getRefError(RefStar const &refStar) const = 0;

    // NO?? (residuals on astrometry should also be double, no?)
    /// Return the fittedStar - refStar residual appropriate for this model (e.g. flux - flux or mag - mag).
    virtual double computeRefResidual(FittedStar const &fittedStar, RefStar const &refStar) const = 0;

    /// Return the number of parameters in the mapping of CcdImage
    unsigned getNpar(CcdImage const &ccdImage) const { return findMapping(ccdImage)->getNpar(); }

    /// Get the mapping associated with ccdImage.
    PhotometryMappingBase const &getMapping(CcdImage const &ccdImage) const {
        return *(findMapping(ccdImage));
    }

    /// Return the total number of parameters in this model.
    virtual int getTotalParameters() const = 0;

    /// Dump the contents of the transforms, for debugging.
    virtual void dump(std::ostream &stream = std::cout) const = 0;

    /**
     * Return true if this is a "reasonable" model.
     *
     * @param ccdImageList The ccdImages to test the model validity on.
     * @param ndof The number of degrees of freedom in the model.
     * @return True if the model is valid on all ccdImages.
     */
    bool validate(CcdImageList const &ccdImageList, int ndof) const;

    friend std::ostream &operator<<(std::ostream &s, Model const &model) {
        model.dump(s);
        return s;
    }

protected:
    /// Return a pointer to the mapping associated with this ccdImage.
    /// TODO: make this a shared_ptr?
    virtual MappingBase *findMapping(CcdImage const &ccdImage) const = 0;

    /// lsst.logging instance, to be created by a subclass so that messages have consistent name.
    LOG_LOGGER _log;

    // Pedestal on flux/magnitude error (percent of flux or delta magnitude)
    double errorPedestal;
};
}  // namespace jointcal
}  // namespace lsst

#endif  // LSST_JOINTCAL_MODEL_H