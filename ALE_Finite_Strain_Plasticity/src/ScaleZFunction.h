/*
 * ScaleZFunction.h
 *
 *  Created on: 06 Dec 2016
 *      Author: maien
 */

#ifndef SCALEZFUNCTION_H_
#define SCALEZFUNCTION_H_

#include <math.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

using namespace dealii;

namespace PlasticityLab {

  template<int dim, typename Number = double, int components=dim>
  class ScaleZFunction: public Function<dim, Number> {
   public:
    ScaleZFunction(const Number scale_factor, const unsigned int component=2);
    virtual ~ScaleZFunction();
    virtual void vector_value(const Point<dim> &p, Vector<Number> &values) const;

   private:
    Number scale_factor;
    const unsigned int component;
  };

  template<int dim, typename Number, int components>
  ScaleZFunction<dim, Number, components>::ScaleZFunction(
      const Number scale_factor,
      const unsigned int component) :
  Function<dim, Number>(components),
  component(component) {
    this->scale_factor = scale_factor;
  }

  template<int dim, typename Number, int components>
  ScaleZFunction<dim, Number, components>::~ScaleZFunction() {}

  template<int dim, typename Number, int components>
  void ScaleZFunction<dim, Number, components>::vector_value(const Point<dim> &p, Vector<Number> &values) const {
    values *= 0.0;
    values[component] = scale_factor * p[component];
  }

} /* namespace PlasticityLab */

#endif /* SCALEZFUNCTION_H_ */
