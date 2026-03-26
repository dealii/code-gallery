/*
 * ReferencePoint.h
 *
 *  Created on: 29 Nov 2020
 *      Author: maien
 */

#ifndef REFERENCEPOINT_H_
#define REFERENCEPOINT_H_

#include <memory>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/point.h>

using namespace dealii;

namespace PlasticityLab {
  template <int dim, typename Number>
  class ReferencePoint {
   public:
    ReferencePoint();
    virtual ~ReferencePoint();

    typename DoFHandler<dim>::active_cell_iterator mesh_motion_cell;
    typename DoFHandler<dim>::active_cell_iterator field_cell;
    unsigned int q_point;
    Point<dim, Number>   reference_point;
    Point<dim, Number>   remapped_point;
  };

  template <int dim, typename Number>
  ReferencePoint<dim, Number>::ReferencePoint() {
  }

  template <int dim, typename Number>
  ReferencePoint<dim, Number>::~ReferencePoint() {}

} /* namespace PlasticityLab */

#endif /* REFERENCEPOINT_H_ */
