/*
 * ScaleComponentFunction.h
 *
 *  Created on: 28 Dec 2019
 *      Author: maien
 */

#ifndef SCALECOMPONENTFUNCTION_H_
#define SCALECOMPONENTFUNCTION_H_

#include <math.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

using namespace dealii;

namespace PlasticityLab {

  template<int dim, typename Number = double, int components=dim>
  class ScaleComponentFunction: public Function<dim, Number> {
   public:
    ScaleComponentFunction(
      const Number scale_factor,
      const unsigned int in_component=2,
      const unsigned int out_component=2);
    virtual ~ScaleComponentFunction();
    virtual void vector_value(const Point<dim> &p, Vector<Number> &values) const;

   private:
    Number scale_factor;
    const unsigned int in_component;
    const unsigned int out_component;
  };

  template<int dim, typename Number, int components>
  ScaleComponentFunction<dim, Number, components>::ScaleComponentFunction(
      const Number scale_factor,
      const unsigned int in_component,
      const unsigned int out_component) :
  Function<dim, Number>(components),
  in_component(in_component),
  out_component(out_component),
  scale_factor(scale_factor) { }

  template<int dim, typename Number, int components>
  ScaleComponentFunction<dim, Number, components>::~ScaleComponentFunction() {}

  template<int dim, typename Number, int components>
  void ScaleComponentFunction<dim, Number, components>::vector_value(const Point<dim> &p, Vector<Number> &values) const {
    values *= 0.0;
    values[out_component] = scale_factor * p[in_component];
  }

} /* namespace PlasticityLab */

#endif /* SCALECOMPONENTFUNCTION_H_ */
