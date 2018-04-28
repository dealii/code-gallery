#ifndef _HEATEQUATION_H_
#define _HEATEQUATION_H_

#include <fstream>

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/convergence_table.h>

using namespace dealii;

template <int dim>
class HeatEquation
{
public:
  HeatEquation();
  void define();
  void step(Vector<double>& braid_data,
            double deltaT,
            double a_time,
            int a_time_idx);

  int size() const; /// Returns the size of the solution vector

  void dump_vec(std::ofstream& file,
                const Vector<double>& vector) const;

  void output_results(int a_time_idx,
                      double a_time,
                      Vector<double>& a_solution) const;

  void initialize(double a_time,
                  Vector<double>& a_vector) const;

  void process_solution(double a_time,
                        int a_index,
                        const Vector<double>& a_vector);

private:
  void setup_system();
  void solve_time_step(Vector<double>& a_solution);

  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;

  ConstraintMatrix     constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;
  SparseMatrix<double> system_matrix;

  Vector<double>       system_rhs;

  std::ofstream myfile;

  const double         theta;

  // These were originally in the run() function but because
  // I am splitting the run() function up into define and step
  // they need to become member data
  Vector<double> tmp;
  Vector<double> forcing_terms;

  ConvergenceTable convergence_table;
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide ()
    :
    Function<dim>(),
    period (0.2)
  {}

  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;

private:
  const double period;
};

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value (const Point<dim>  &p,
                        const unsigned int component = 0) const;
};

template <int dim>
class RightHandSideMFG : public Function<dim>
{
public:
  virtual double value (const Point<dim>  &p,
                        const unsigned int component = 0) const;
};


template <int dim>
class InitialValuesMFG : public Function<dim>
{
public:
  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;
};

/**
 * Provides the exact value for the manufactured solution.
 * User must set the time using ExactValuesMFG::set_time(const Number new_time);
 * Calculates \f$u_h=e^(-4\pi\pi t)*\cos(2\pi x)\cos(2\pi y)\f$ for all points given to it.
 * In the context of manufactured solutions, this is used on the boundary.
 */
template <int dim>
class ExactValuesMFG : public Function<dim>
{
public:
  /**
   *  \brief Computes the value at the given point and member data time
   *
   *  Computes the exact value of the manufactured solution at point p and
   *  the member data time. See the class documentation and the design doc
   *  for details on what the exact solution is.
   *
   *  \param p The point that the exact solution is computed at
   *  \param component The component of the exact solution (always 0 for now)
   *  \return double The exact value that was computed
   */
  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;

  /**
   *  \brief Computes the gradient of the exact solution at the given point
   *
   *  Computes the gradient of the exact/manufactured solution value at
   *  point p and member data time. See the design doc for details on
   *  what the gradient of the exact solution is
   *
   *  \param p The point that the gradient is calculated at
   *  \param component The component of the system of equations this gradient is for
   *  \return Tensor<1,dim> A rank 1 tensor that contains the gradient
   *  in each spatial dimension
   */
  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
};


#include "HeatEquationImplem.hh"

#endif // _HEATEQUATION_H_
