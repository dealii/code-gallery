/*
 * DoFSystem.h
 *
 *  Created on: 05 May 2015
 *      Author: maien
 */

#ifndef DOFSYSTEM_H_
#define DOFSYSTEM_H_

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/conditional_ostream.h>

#include "InterpolatoryConstraintApplier.h"
#include "BodyForceApplier.h"
#include "ConvectionBoundaryConditionApplier.h"
#include "mpi.h"
#include "utilities.h"

using namespace dealii;

namespace PlasticityLab {

  template <int dim, typename Number=double>
  class DoFSystem {
   public:
    DoFSystem (const dealii::Triangulation<dim> &triangulation,
               const dealii::Mapping<dim> &mapping);

    DoFHandler<dim>     dof_handler;

    AffineConstraints<Number>    nodal_constraints;

    IndexSet           locally_owned_dofs;
    IndexSet           locally_relevant_dofs;

    void setup_dof_system (const FiniteElement<dim> &fe);
    const dealii::Mapping<dim> &mapping;

  };

  template<int dim, typename Number>
  DoFSystem <dim, Number> :: DoFSystem(const dealii::Triangulation<dim> &triangulation,
                                       const dealii::Mapping<dim> &mapping) :
    dof_handler(triangulation),
    mapping(mapping) {
  }

  template <int dim, typename Number>
  void DoFSystem<dim, Number>::setup_dof_system (const FiniteElement<dim> &fe) {
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    nodal_constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, nodal_constraints);
  }

} /* namespace PlasticityLab */

#endif /* DOFSYSTEM_H_ */
