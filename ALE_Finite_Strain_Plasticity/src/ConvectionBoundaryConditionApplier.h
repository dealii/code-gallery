/*
 * ConvectionBoundaryConditionApplier.h
 *
 *  Created on: 07 Oct 2017
 *      Author: maien
 */

#ifndef CONVECTIONBOUNDARYCONDITIONAPPLIER_H_
#define CONVECTIONBOUNDARYCONDITIONAPPLIER_H_

namespace PlasticityLab {

  template <int dim, typename Number = double>
  class ConvectionBoundaryConditionApplier {
   public:
    ConvectionBoundaryConditionApplier();
    ConvectionBoundaryConditionApplier(
      int direction,
      Number convection_coefficient = 1.0,
      Number ambient_field_value = 0.0);
    virtual ~ConvectionBoundaryConditionApplier();
    inline Number apply(const unsigned int direction,
                        const Number test_function_value,
                        const Number field_value,
                        const Number JxW) const;
    inline Number apply_gradient(
      const unsigned int direction,
      const Number &test_gradient,
      const Number &field_gradient,
      const Number JxW) const;
   private:
    const unsigned int direction;
    const Number convection_coefficient;
    const Number ambient_field_value;
  };


  template <int dim, typename Number>
  ConvectionBoundaryConditionApplier<dim, Number>::
  ConvectionBoundaryConditionApplier(
    int direction,
    Number convection_coefficient,
    Number ambient_field_value)
    : direction(direction),
      convection_coefficient(convection_coefficient),
      ambient_field_value(ambient_field_value) {
  }


  template <int dim, typename Number>
  ConvectionBoundaryConditionApplier<dim, Number>::~ConvectionBoundaryConditionApplier() {
  }

  template <int dim, typename Number>
  Number ConvectionBoundaryConditionApplier<dim, Number>::
  apply(const unsigned int direction,
        const Number test_function_value,
        const Number field_value,
        const Number JxW) const {
    if (this->direction == direction)
      return test_function_value * convection_coefficient * (field_value - ambient_field_value) * JxW;
    return 0.0;
  }

  template <int dim, typename Number>
  Number ConvectionBoundaryConditionApplier<dim, Number>::apply_gradient(
    const unsigned int direction,
    const Number &test_gradient,
    const Number &field_gradient,
    const Number JxW) const {
    if (this->direction == direction) {
      return convection_coefficient * (test_gradient * field_gradient) * JxW;
    }
    return 0;
  }


} /* namespace PlasticityLab */

#endif /* CONVECTIONBOUNDARYCONDITIONAPPLIER_H_ */
