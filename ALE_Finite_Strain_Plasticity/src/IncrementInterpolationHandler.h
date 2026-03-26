/*
 * IncrementInterpolationHandler.h
 *
 *  Created on: 25 Oct 2019
 *      Author: maien
 */

#ifndef INCREMENTINTERPOLATIONHANDLER_H_
#define INCREMENTINTERPOLATIONHANDLER_H_


using namespace dealii;

namespace PlasticityLab {
template <int dim, typename Number=double, int components=dim>
class IncrementInterpolationHandler {
public:
    IncrementInterpolationHandler(
            Function<dim, Number> *increment_interpolation_function,
            bool do_interpolate,
            ComponentMask interpolation_component_mask,
            bool do_constrain,
            ComponentMask constrain_component_mask,
            types::boundary_id constrain_boundary_id,
            Mapping<dim> &mapping)
    : increment_interpolation_function(increment_interpolation_function),
      do_interpolate(do_interpolate),
      interpolation_component_mask(interpolation_component_mask),
      do_constrain(do_constrain),
      constrain_component_mask(constrain_component_mask),
      constrain_boundary_id(constrain_boundary_id),
      mapping(mapping) { }

    ~IncrementInterpolationHandler(){
        delete increment_interpolation_function;
    }

    void advance_time(const Number delta_t);

    template<typename VectorType>
    void distribute_step_constraints(VectorType &increment) const {
        if(do_constrain) {
            function_constraint.distribute(increment);
        }
    }

    template<typename VectorType>
    void interpolate(VectorType &increment, const DoFSystem<dim, Number> &dof_system) const {
        if(do_interpolate) {
            VectorTools::interpolate(
              mapping,
              dof_system.dof_handler,
              *increment_interpolation_function,
              increment,
              interpolation_component_mask);
        }
    }

    void reinit_constraint_matrix(const DoFSystem<dim, Number> &dof_system) {
        if(do_constrain) {
            function_constraint.reinit(dof_system.locally_relevant_dofs);
            DoFTools::make_hanging_node_constraints(dof_system.dof_handler, function_constraint);
            std::map< types::boundary_id, const Function< dim, Number > * > constraint_function_map;
            constraint_function_map.insert(std::make_pair(constrain_boundary_id, increment_interpolation_function));
            dealii::VectorTools::interpolate_boundary_values(dof_system.mapping, dof_system.dof_handler, constraint_function_map, function_constraint, constrain_component_mask);
            function_constraint.close();
        }
    }

private:
    Function<dim, Number>* increment_interpolation_function;
    const bool do_interpolate;
    ComponentMask interpolation_component_mask;
    const bool do_constrain;
    ComponentMask constrain_component_mask;
    types::boundary_id constrain_boundary_id;
    AffineConstraints<Number> function_constraint;
    const Mapping<dim> &mapping;
};


template<int dim, typename Number, int components>
void IncrementInterpolationHandler<dim, Number, components>::advance_time(const Number delta_t) {
    increment_interpolation_function->advance_time(delta_t);
}

} /* namespace PlasticityLab */

#endif /* INCREMENTINTERPOLATIONHANDLER_H_ */
