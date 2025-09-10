#ifndef usrcodes_utilities_h
#define usrcodes_utilities_h
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

#include <fstream>
#include <iostream>


namespace usr_utilities
{
  using namespace dealii;

  template <int dim>
  std::vector<types::global_dof_index> get_vertex_dofs(
    const typename Triangulation<dim>::active_vertex_iterator &vertex,
    const DoFHandler<dim> &dof_handler)
  {
    DoFAccessor<0, dim, dim, false> vertex_dofs(
        &(dof_handler.get_triangulation()),
        vertex->level(),
        vertex->index(),
        &dof_handler);
    const unsigned int n_dofs = dof_handler.get_fe().dofs_per_vertex;
    std::vector<types::global_dof_index> dofs(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      dofs[i] = vertex_dofs.vertex_dof_index(0, i);
    }
    return dofs;
  }
}
#endif
