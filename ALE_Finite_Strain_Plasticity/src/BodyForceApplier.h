/*
 * BodyForceApplier.h
 *
 *  Created on: 14 Jan 2015
 *      Author: maien
 */

#ifndef BODYFORCEAPPLIER_H_
#define BODYFORCEAPPLIER_H_

namespace PlasticityLab {

  template <int dim, typename Number = double>
  class BodyForceApplier {
   public:
    BodyForceApplier();
    BodyForceApplier(int direction, Number bodyForceMagnitude = 0);
    virtual ~BodyForceApplier();
    inline Number apply(const unsigned int direction,
                        const Number shapeFunctionValue,
                        const Number JxW) const;
   private:
    const unsigned int direction;
    const Number bodyForceMagnitude;
  };


  template <int dim, typename Number>
  BodyForceApplier<dim, Number>::
  BodyForceApplier(int direction, Number bodyForceMagnitude)
    : direction(direction), bodyForceMagnitude(bodyForceMagnitude) {
  }


  template <int dim, typename Number>
  BodyForceApplier<dim, Number>::~BodyForceApplier() {
  }

  template <int dim, typename Number>
  Number BodyForceApplier<dim, Number>::
  apply(const unsigned int direction,
        const Number shapeFunctionValue,
        const Number JxW) const {
    if (this->direction == direction)
      return -shapeFunctionValue * this->bodyForceMagnitude * JxW;
    return 0.0;
  }

} /* namespace PlasticityLab */

#endif /* BODYFORCEAPPLIER_H_ */
