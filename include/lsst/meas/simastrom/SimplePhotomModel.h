#ifndef SIMPLEPHOTOMMODEL__H
#define SIMPLEPHOTOMMODEL__H

#include "lsst/meas/simastrom/Eigenstuff.h"
#include "lsst/meas/simastrom/PhotomModel.h"
#include <map>

namespace lsst {
namespace meas {
namespace simastrom {

class CcdImageList;
class CcdImage;
class Point;

//! Photometric response model which has a single photometric factor per CcdImage.
/*! It considers a full exposure as reference. */  
 class SimplePhotomModel : public PhotomModel
{

  struct PhotomStuff
  {
    unsigned index;
    double factor;
    bool fixed;
  PhotomStuff(const unsigned I=0, const double F=1) : index(I), factor(F), fixed(false) {};
  };

  typedef std::map<const CcdImage*,PhotomStuff> mapType;
  mapType _myMap;

  PhotomStuff& find(const CcdImage &C);
  const PhotomStuff& find(const CcdImage &C) const;

public :

  SimplePhotomModel(const CcdImageList &L);

  //! Assign indices to parameters involved in mappings, starting at FirstIndex. Returns the highest assigned index.
  unsigned AssignIndices(unsigned FirstIndex);

  //! Offset the parameters by the provided amounts. 
  /*! The shifts are applied according to the indices given in
      AssignIndices. */
  void OffsetParams(const Eigen::VectorXd &Delta);

  //! This model ignores "Where".
  double PhotomFactor(const CcdImage& C, const Point &Where) const;

  virtual void GetIndicesAndDerivatives(const MeasuredStar &M,
					const CcdImage &Ccd, 
					std::vector<unsigned> &Indices,
					Eigen::VectorXd &D);

};


}}}

#endif /*SIMPLEPHOTOMMODEL__H */ 
