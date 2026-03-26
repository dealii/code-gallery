/*
 * RotationFunction.h
 *
 *  Created on: 06 Dec 2016
 *      Author: maien
 */

#ifndef ROTATIONFUNCTION_H_
#define ROTATIONFUNCTION_H_

#include <math.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

using namespace dealii;

namespace PlasticityLab {

  template<int dim, typename Number = double>
  class RotationFunction: public Function<dim, Number> {
   public:
    RotationFunction(
      const Number angular_frequency,
      const Number axial_velocity,
      const Number current_angle = 0.0);
    virtual ~RotationFunction();

    virtual void  vector_value (const Point< dim > &p, Vector< Number > &values) const;
    virtual void  set_time (const Number new_time);

   private:
    void update_rotation_matrix();
    Tensor<2, dim, Number> previousRotationMatrix, currentRotationMatrix;
    Number current_angle;
    Number previous_displacement, current_displacement;
    Number angular_frequency;
    Number axial_velocity;
  };

  template<int dim, typename Number = double>
  class AngularVelocityFunction: public Function<dim, Number> {
   public:
    AngularVelocityFunction(const Number angular_frequency) : Function<dim, Number>(dim) {
        this->angular_frequency = angular_frequency;
    }
    virtual ~AngularVelocityFunction() {}

    virtual void  vector_value (const Point< dim > &p, Vector< Number > &values) const {
      values[0] = - 2 * 3.14159 * angular_frequency * p[1];
      values[1] =   2 * 3.14159 * angular_frequency * p[0];
      if(3==dim) {
        values[2] = 0;
      }
    }

   private:
    Number angular_frequency;
  };

  template<int dim, typename Number>
  RotationFunction<dim, Number>::~RotationFunction() {}

  template<int dim, typename Number>
  RotationFunction<dim, Number>::RotationFunction(
    const Number angular_frequency,
    const Number axial_velocity,
    const Number current_angle) : Function<dim, Number>(dim) {
    this->angular_frequency = angular_frequency;
    this->axial_velocity = axial_velocity;
    this->current_angle = current_angle;
    this->set_time(0.0);
    current_displacement = 0.0;
  }

  template<int dim, typename Number>
  void RotationFunction<dim, Number>::vector_value (const Point< dim > &p, Vector< Number > &values) const {
    const auto p_prev = previousRotationMatrix * p;
    const auto res = currentRotationMatrix * p;
    for (unsigned int i = 0; i < dim; ++i)
      values[i] = res[i] - p_prev[i] + (i == 2 ? current_displacement - previous_displacement : 0.0);
  }

  template<int dim, typename Number>
  void RotationFunction<dim, Number>::update_rotation_matrix () {
    previousRotationMatrix = currentRotationMatrix;
    const Number c_theta = std::cos(current_angle);
    const Number s_theta = std::sin(current_angle);
    currentRotationMatrix[0][0] = c_theta;
    currentRotationMatrix[0][1] = -s_theta;
    currentRotationMatrix[1][0] = s_theta;
    currentRotationMatrix[1][1] = c_theta;
    if (dim == 3)
      currentRotationMatrix[2][2] = 1.0;
  }

  template<int dim, typename Number>
  void  RotationFunction<dim, Number>::set_time (const Number new_time) {
    Function<dim, Number>::set_time(new_time);
    current_angle = angular_frequency * Function<dim, Number>::get_time();
    update_rotation_matrix();
    previous_displacement = current_displacement;
    current_displacement = axial_velocity * Function<dim, Number>::get_time();
  }

} /* namespace PlasticityLab */

#endif /* ROTATIONFUNCTION_H_ */
