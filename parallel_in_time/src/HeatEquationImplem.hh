#include <iomanip>
#include <math.h>
#include "Utilities.hh"

// Calculates the forcing function for the RightHandSide. See the
// documentation for the math.
template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int component) const
{
  (void) component;
  Assert (component == 0, ExcIndexRange(component, 0, 1));
  Assert (dim == 2, ExcNotImplemented());

  double time = this->get_time();

  if ((p[0] > 0.5) && (p[1] > -0.5))
    {
      return std::exp(-0.5*(time-0.125)*(time-0.125)/(0.005));
    }
  else if ((p[0] > -0.5) && (p[1] > 0.5))
    {
      return std::exp(-0.5*(time-0.375)*(time-0.375)/(0.005));
    }
  else
    {
      return 0;
    }

  return 0; // No forcing function
}

// Calculates the forcing function for the method of manufactured
// solutions. See the documentation for the math.
template <int dim>
double RightHandSideMFG<dim>::value (const Point<dim> &p,
                                     const unsigned int component) const
{
  (void) component;
  Assert (component == 0, ExcIndexRange(component, 0, 1));
  Assert (dim == 2, ExcNotImplemented());

  double time = this->get_time();

  double pi = numbers::PI;
  return 4*pi*pi*std::exp(-4*pi*pi*time)*std::cos(2*pi*p[0])*std::cos(2*pi*p[1]);
}

// Calculates the boundary conditions, essentially zero everywhere.
template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                            const unsigned int component) const
{
  UNUSED(p);
  (void) component;
  Assert (component == 0, ExcIndexRange(component, 0, 1));
  return 0;
}

// Calculates the exact solution (and thus also boundary conditions)
// for the method of manufactured solutions.
template <int dim>
double ExactValuesMFG<dim>::value (const Point<dim> &p,
                                   const unsigned int component) const
{
  (void) component;
  Assert (component == 0, ExcIndexRange(component, 0, 1));

  double time = this->get_time();
  const double pi = numbers::PI;

  return std::exp(-4*pi*pi*time)*std::cos(2*pi*p[0])*std::cos(2*pi*p[1]);
}

// Calculates the gradient of the exact solution for the method of manufactured
// solutions. See the documentation for the math.
template <int dim>
Tensor<1,dim> ExactValuesMFG<dim>::gradient (const Point<dim>   &p,
                                             const unsigned int) const
{
  Assert (dim == 2, ExcNotImplemented());

  Tensor<1,dim> return_value;
  const double pi = numbers::PI;
  double time = this->get_time();
  return_value[0] = -2*pi*std::exp(-4*pi*pi*time)*std::cos(2*pi*p[1])*std::sin(2*pi*p[0]);
  return_value[1] = -2*pi*std::exp(-4*pi*pi*time)*std::cos(2*pi*p[0])*std::sin(2*pi*p[1]);
  return return_value;
}

// Calculates the initial values for the method of manufactured solutions.
// See the documentation for the math.
template <int dim>
double InitialValuesMFG<dim>::value (const Point<dim> &p,
                                     const unsigned int component) const
{
  (void) component;
  Assert (component == 0, ExcIndexRange(component, 0, 1));
  const double pi = numbers::PI;

  return std::cos(2*pi*p[0])*std::cos(2*pi*p[1]);
}

template <int dim>
HeatEquation<dim>::HeatEquation ()
  :
  fe(1),
  dof_handler(triangulation),
  theta(0.5)
{
}

template <int dim>
void HeatEquation<dim>::initialize(double a_time,
                                   Vector<double>& a_vector) const
{
#if DO_MFG
  // We only initialize values in the manufactured solution case
  InitialValuesMFG<dim> iv_function;
  iv_function.set_time(a_time);
  VectorTools::project (dof_handler, constraints,
                        QGauss<dim>(fe.degree+1), iv_function,
                        a_vector);
#else
  UNUSED(a_time);
  UNUSED(a_vector);
#endif // DO_MFG
  // If not the MFG solution case, a_vector is already zero'd so do nothing
}

template <int dim>
void HeatEquation<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ true);
  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree+1),
                                    mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe.degree+1),
                                       laplace_matrix);

  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void HeatEquation<dim>::solve_time_step(Vector<double>& a_solution)
{
  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<> cg(solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.0);

  cg.solve(system_matrix, a_solution, system_rhs,
           preconditioner);

  constraints.distribute(a_solution);
}



template <int dim>
void HeatEquation<dim>::output_results(int a_time_idx,
                                       double a_time,
                                       Vector<double>& a_solution) const
{

  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.time = a_time;
  vtk_flags.cycle = a_time_idx;

  DataOut<dim> data_out;
  data_out.set_flags(vtk_flags);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(a_solution, "U");

  data_out.build_patches();

  const std::string filename = "solution-"
    + Utilities::int_to_string(a_time_idx, 3) +
    ".vtk";
  std::ofstream output(filename.c_str());
  data_out.write_vtk(output);
}

// We define the geometry here, this is called on each processor
// and doesn't change in time. Once doing AMR, this won't need
// to exist anymore.
template <int dim>
void HeatEquation<dim>::define()
{
  const unsigned int initial_global_refinement = 6;

  GridGenerator::hyper_L (triangulation);
  triangulation.refine_global (initial_global_refinement);

  setup_system();

  tmp.reinit (dof_handler.n_dofs());
  forcing_terms.reinit (dof_handler.n_dofs());
}

// Here we advance the solution forward in time. This is done
// the same way as in the loop in step-26's run function.
template<int dim>
void HeatEquation<dim>::step(Vector<double>& braid_data,
                             double deltaT,
                             double a_time,
                             int a_time_idx)
{
  a_time += deltaT;
  ++a_time_idx;

  mass_matrix.vmult(system_rhs, braid_data);

  laplace_matrix.vmult(tmp, braid_data);

  system_rhs.add(-(1 - theta) * deltaT, tmp);

#if DO_MFG
  RightHandSideMFG<dim> rhs_function;
#else
  RightHandSide<dim> rhs_function;
#endif
  rhs_function.set_time(a_time);
  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree+1),
                                      rhs_function,
                                      tmp);

  forcing_terms = tmp;
  forcing_terms *= deltaT * theta;

  rhs_function.set_time(a_time - deltaT);
  VectorTools::create_right_hand_side(dof_handler,
                                      QGauss<dim>(fe.degree+1),
                                      rhs_function,
                                      tmp);

  forcing_terms.add(deltaT * (1 - theta), tmp);
  system_rhs += forcing_terms;

  system_matrix.copy_from(mass_matrix);
  system_matrix.add(theta * deltaT, laplace_matrix);

  constraints.condense (system_matrix, system_rhs);

  {
#if DO_MFG
    // If we are doing the method of manufactured solutions
    // then we set the boundary conditions to the exact solution.
    // Otherwise the boundary conditions are zero.
    ExactValuesMFG<dim> boundary_values_function;
#else
    BoundaryValues<dim> boundary_values_function;
#endif
    boundary_values_function.set_time(a_time);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             boundary_values_function,
                                             boundary_values);

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       braid_data,
                                       system_rhs);
  }

  solve_time_step(braid_data);
}

template<int dim>
int HeatEquation<dim>::size() const
{
  return dof_handler.n_dofs();
}

// This function computes the error for the time step when doing
// the method of manufactured solutions. First the exact values
// is calculated, then the difference per cell is computed for
// the various norms, and the error is computed. This is written
// out to a pretty table.
template<int dim> void
HeatEquation<dim>::process_solution(double a_time,
                                    int a_index,
                                    const Vector<double>& a_vector)
{
  // Compute the exact value for the manufactured solution case
  ExactValuesMFG<dim> exact_function;
  exact_function.set_time(a_time);

  Vector<double> difference_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    a_vector,
                                    exact_function,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree+1),
                                    VectorTools::L2_norm);

  const double L2_error = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::L2_norm);

  VectorTools::integrate_difference(dof_handler,
                                    a_vector,
                                    exact_function,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree+1),
                                    VectorTools::H1_seminorm);

  const double H1_error = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::H1_seminorm);

  const QTrapez<1> q_trapez;
  const QIterated<dim> q_iterated (q_trapez, 5);
  VectorTools::integrate_difference (dof_handler,
                                     a_vector,
                                     exact_function,
                                     difference_per_cell,
                                     q_iterated,
                                     VectorTools::Linfty_norm);
  const double Linfty_error = VectorTools::compute_global_error(triangulation,
                                                                difference_per_cell,
                                                                VectorTools::Linfty_norm);

  const unsigned int n_active_cells = triangulation.n_active_cells();
  const unsigned int n_dofs = dof_handler.n_dofs();

  pout() << "Cycle " << a_index << ':'
         << std::endl
         << "   Number of active cells:       "
         << n_active_cells
         << std::endl
         << "   Number of degrees of freedom: "
         << n_dofs
         << std::endl;

  convergence_table.add_value("cycle", a_index);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
  convergence_table.add_value("Linfty", Linfty_error);

  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_precision("Linfty", 3);

  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.set_scientific("Linfty", true);

  convergence_table.set_tex_caption("cells", "\\# cells");
  convergence_table.set_tex_caption("dofs", "\\# dofs");
  convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
  convergence_table.set_tex_caption("H1", "@f$H^1@f$-error");
  convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");

  convergence_table.set_tex_format("cells", "r");
  convergence_table.set_tex_format("dofs", "r");

  std::cout << std::endl;
  convergence_table.write_text(std::cout);

  std::ofstream error_table_file("tex-conv-table.tex");
  convergence_table.write_tex(error_table_file);
}
