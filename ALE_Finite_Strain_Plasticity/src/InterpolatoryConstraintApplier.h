/*
 * InterpolatoryConstraintApplier.h
 *
 *  Created on: 15 Jan 2015
 *      Author: maien
 */

#ifndef INTERPOLATORYCONSTRAINTAPPLIER_H_
#define INTERPOLATORYCONSTRAINTAPPLIER_H_

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace PlasticityLab {

  template <int dim, typename Number = double>
  class InterpolatoryConstraintApplier {
   public:
    InterpolatoryConstraintApplier();
    InterpolatoryConstraintApplier(const std::map< dealii::types::boundary_id, const dealii::Function< dim, Number > * > &constraintFunctionMap,
                                   dealii::ComponentMask componentMask);
    virtual ~InterpolatoryConstraintApplier();

    void configure(std::map< dealii::types::boundary_id, const dealii::Function< dim, Number > * > constraintFunctionMap,
                   dealii::ComponentMask componentMask);

    void apply(const dealii::Mapping<dim> &mapping,
               dealii::DoFHandler<dim> &doFHandler,
               dealii::AffineConstraints<Number> &constraintMatrix,
               bool useComponentMask = true) const;

   private:
    std::map< dealii::types::boundary_id, const dealii::Function< dim, Number > * > constraintFunctionMap;
    dealii::ComponentMask componentMask;
  };

  template <int dim, typename Number>
  InterpolatoryConstraintApplier<dim, Number>::InterpolatoryConstraintApplier() {
  }

  template <int dim, typename Number>
  InterpolatoryConstraintApplier<dim, Number>::InterpolatoryConstraintApplier
  (const std::map< dealii::types::boundary_id, const dealii::Function< dim, Number > * > &constraintFunctionMap,
   dealii::ComponentMask componentMask):
    constraintFunctionMap(constraintFunctionMap),
    componentMask(componentMask) {
  }

  template <int dim, typename Number>
  InterpolatoryConstraintApplier<dim, Number>::~InterpolatoryConstraintApplier() {
  }

  template <int dim, typename Number>
  void InterpolatoryConstraintApplier<dim, Number>::configure
  (std::map< dealii::types::boundary_id, const dealii::Function< dim, Number > * > constraintFunctionMap,
   dealii::ComponentMask componentMask) {
    this->constraintFunctionMap = std::map< dealii::types::boundary_id, const dealii::Function< dim, Number > * >(constraintFunctionMap);
    this->componentMask = dealii::ComponentMask(componentMask);
  }

  template <int dim, typename Number>
  void InterpolatoryConstraintApplier<dim, Number>::apply(const dealii::Mapping<dim> &mapping,
                                                          dealii::DoFHandler<dim> &doFHandler,
                                                          dealii::AffineConstraints<Number> &constraintMatrix,
                                                          bool useComponentMask) const {
    if (useComponentMask)
      dealii::VectorTools::interpolate_boundary_values(mapping,
                                                       doFHandler,
                                                       constraintFunctionMap,
                                                       constraintMatrix,
                                                       componentMask);
    else
      dealii::VectorTools::interpolate_boundary_values(mapping,
                                                       doFHandler,
                                                       constraintFunctionMap,
                                                       constraintMatrix);
  }

} /* namespace PlasticityLab */

#endif /* INTERPOLATORYCONSTRAINTAPPLIER_H_ */
