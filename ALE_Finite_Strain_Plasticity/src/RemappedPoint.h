/*
 * RemappedPoint.h
 *
 *  Created on: 29 Nov 2020
 *      Author: maien
 */

#ifndef REMAPPEDPOINT_H_
#define REMAPPEDPOINT_H_

#include <memory>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/point.h>

using namespace dealii;

namespace PlasticityLab {
  template <int dim, typename Number>
  class RemappedPoint {
   public:
    RemappedPoint();
    virtual ~RemappedPoint();

    typename DoFHandler<dim>::active_cell_iterator mesh_motion_cell;
    typename DoFHandler<dim>::active_cell_iterator field_cell;
    typename DoFHandler<dim>::active_cell_iterator mixed_fe_cell;
    Point<dim, Number> unit_cell_point;
    Point<dim, Number> remapped_point;
  };

  template <int dim, typename Number>
  RemappedPoint<dim, Number>::RemappedPoint() {
  }

  template <int dim, typename Number>
  RemappedPoint<dim, Number>::~RemappedPoint() {}

} /* namespace PlasticityLab */

#endif /* REMAPPEDPOINT_H_ */
