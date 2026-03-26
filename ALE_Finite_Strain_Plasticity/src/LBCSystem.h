/*
 * LBCSystem.h
 *
 *  Created on: 08 Oct 2019
 *      Author: maien
 */

#ifndef LBCSYSTEM_H_
#define LBCSYSTEM_H_

#include <deal.II/base/function.h>

#include "InterpolatoryConstraintApplier.h"
#include "BodyForceApplier.h"
#include "ConvectionBoundaryConditionApplier.h"
#include "IncrementInterpolationHandler.h"
#include "utilities.h"
#include "DoFSystem.h"
#include "BoundaryUnidirectionalPenaltySpec.h"

using namespace dealii;
using namespace Functions;

namespace PlasticityLab {

template <int dim, typename Number=double, int components=dim>
class LBCSystem {
public:
    LBCSystem(): zero_function(components){}
    ~LBCSystem() {
        for(auto increment_interpolation_handler: increment_interpolation_handlers) {
            delete increment_interpolation_handler;
        }
        for(auto initial_velocity_interpolation_handler: initial_velocity_interpolation_handlers) {
            delete initial_velocity_interpolation_handler;
        }
        for(auto initial_deformation_interpolation_handler: initial_deformation_interpolation_handlers) {
            delete initial_deformation_interpolation_handler;
        }
        for(auto boundary_unidirectional_penalty_spec: boundary_unidirectional_penalty_specs) {
            delete boundary_unidirectional_penalty_spec;
        }
    }

    void apply_constraints(DoFSystem<dim, Number> &dof_system) const;

    void clear();

    std::vector< BodyForceApplier<dim, Number> > bodyLoadAppliers;
    std::vector< std::pair<int, BodyForceApplier<dim, Number> > > boundaryLoadAppliers;
    std::vector< std::pair<int, ConvectionBoundaryConditionApplier<dim, Number> > > convection_BC_appliers;
    std::vector< InterpolatoryConstraintApplier<dim, Number> > interpolatoryConstraintAppliers;
    std::vector<std::pair<unsigned int, std::set<types::boundary_id>>> no_normal_flux_constraints;
    std::vector<IncrementInterpolationHandler<dim, Number, components>*> increment_interpolation_handlers;
    std::vector<IncrementInterpolationHandler<dim, Number, components>*> initial_velocity_interpolation_handlers;
    std::vector<IncrementInterpolationHandler<dim, Number, components>*> initial_deformation_interpolation_handlers;
    std::vector<BoundaryUnidirectionalPenaltySpec<Number>*> boundary_unidirectional_penalty_specs;

    ZeroFunction<dim, Number>     zero_function;
};


template<int dim, typename Number, int components>
void LBCSystem<dim, Number, components>::apply_constraints(DoFSystem<dim, Number> &dof_system) const {
    for (auto constraintApplier = interpolatoryConstraintAppliers.cbegin();
          constraintApplier != interpolatoryConstraintAppliers.end();
          ++constraintApplier) {
      constraintApplier->apply(dof_system.mapping, dof_system.dof_handler, dof_system.nodal_constraints);
    }

    for (auto no_normal_flux_constraint : no_normal_flux_constraints) {
      VectorTools::compute_no_normal_flux_constraints(
        dof_system.dof_handler,
        no_normal_flux_constraint.first,
        no_normal_flux_constraint.second,
        dof_system.nodal_constraints,
        dof_system.mapping);
    }

    dof_system.nodal_constraints.close();
}


template<int dim, typename Number, int components>
void LBCSystem<dim, Number, components>::clear() {
    for(auto increment_interpolation_handler: increment_interpolation_handlers) {
        delete increment_interpolation_handler;
    }
    for(auto initial_velocity_interpolation_handler: initial_velocity_interpolation_handlers) {
        delete initial_velocity_interpolation_handler;
    }
    for(auto initial_deformation_interpolation_handler: initial_deformation_interpolation_handlers) {
        delete initial_deformation_interpolation_handler;
    }
    for(auto boundary_unidirectional_penalty_spec: boundary_unidirectional_penalty_specs) {
        delete boundary_unidirectional_penalty_spec;
    }

    bodyLoadAppliers.clear();
    boundaryLoadAppliers.clear();
    convection_BC_appliers.clear();
    interpolatoryConstraintAppliers.clear();
    no_normal_flux_constraints.clear();
    increment_interpolation_handlers.clear();
    initial_velocity_interpolation_handlers.clear();
    initial_deformation_interpolation_handlers.clear();
    boundary_unidirectional_penalty_specs.clear();

}


} /* namespace PlasticityLab */

#endif /* LBCSYSTEM_H_ */
