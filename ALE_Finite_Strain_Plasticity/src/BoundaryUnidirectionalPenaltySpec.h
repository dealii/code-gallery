/*
 * BoundaryUnidirectionalPenaltySpec.h
 *
 *  Created on: 04 May 2021
 *      Author: maien
 */

#ifndef BOUNDARYUNIDIRECTIONALPENALTYSPEC_H_
#define BOUNDARYUNIDIRECTIONALPENALTYSPEC_H_

namespace PlasticityLab {

  template<typename Number = double>
  class BoundaryUnidirectionalPenaltySpec {
  public:
    BoundaryUnidirectionalPenaltySpec(
      unsigned int boundary_id,
      Number reference_displacement_increment,
      Number residual_force,
      Number quadratic_spring_factor) :
        boundary_id(boundary_id),
        reference_displacement_increment(reference_displacement_increment),
        residual_force(residual_force),
        quadratic_spring_factor(quadratic_spring_factor) {}

    unsigned int get_boundary_id() const { return boundary_id; }
    Number get_reference_displacement_increment() const { return reference_displacement_increment; }
    Number get_residual_force() const { return residual_force; }
    Number get_quadratic_spring_factor() const { return quadratic_spring_factor; }

  private:
    const unsigned int boundary_id;
    const Number quadratic_spring_factor;
    const Number reference_displacement_increment;
    const Number residual_force;
  };

} /* namespace PlasticityLab */

#endif /* BOUNDARYUNIDIRECTIONALPENALTYSPEC_H_ */
