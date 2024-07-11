#ifndef TRAVELING_WAVE_SOLVER
#define TRAVELING_WAVE_SOLVER

#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/sundials/kinsol.h>

#include "Parameters.h"
#include "Solution.h"
#include "AuxiliaryFunctions.h"

#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <set>

// Namespace of the program
namespace TravelingWave
{
  using namespace dealii;

  // The main class for construction of the traveling wave solutions.
  class TravelingWaveSolver
  {
  public:
    TravelingWaveSolver(const Parameters &parameters, const SolutionStruct &initial_guess_input);

    void set_triangulation(const Triangulation<1> &itriangulation);

    void run(const std::string filename="solution", const bool save_solution_to_file=true);
    void get_solution(SolutionStruct &solution) const;
    void get_triangulation(Triangulation<1> &otriangulation) const;

  private:
    void setup_system(const bool initial_step);
    void find_boundary_and_centering_dof_numbers();
    void set_boundary_and_centering_values();

    void set_initial_guess();

    double Heaviside_func(double x) const;

    void compute_and_factorize_jacobian(const Vector<double> &evaluation_point_extended);
    double compute_residual(const Vector<double> &evaluation_point_extended, Vector<double> &residual);
    void split_extended_solution_vector();

    void solve(const Vector<double> &rhs, Vector<double> &solution, const double /*tolerance*/);
    void refine_mesh();
    double run_newton_iterations(const double target_tolerance=1e-5);

    void output_with_double_precision(const Vector<double> &solution, const double wave_speed, const std::string filename="solution");

    // The dimension of the finite element solution increased by one to account for the value corresponding to the wave speed.
    unsigned int extended_solution_dim;
    std::map<std::string, unsigned int> boundary_and_centering_dof_numbers;

    // Parameters of the problem, taken from a .prm file.
    const Parameters  &params;
    const Problem     &problem;    // Reference variable, just for convenience.

    unsigned int number_of_quadrature_points;

    Triangulation<1> triangulation;
    // The flag indicating whether the triangulation was uploaded externally or created within the <code> run </code> member function.
    bool            triangulation_uploaded;
    FESystem<1>     fe;
    DoFHandler<1>   dof_handler;

    // Constraints for Dirichlet boundary conditions.
    AffineConstraints<double> zero_boundary_constraints;	

    SparsityPattern                       sparsity_pattern_extended;
    SparseMatrix<double>                  jacobian_matrix_extended;
    std::unique_ptr<SparseDirectUMFPACK>  jacobian_matrix_extended_factorization;

    // Finite element solution of the problem.
    Vector<double>  current_solution;

    // Value of the wave speed $c$.
    double          current_wave_speed;

    // Solution with an additional term, corresponding to the variable wave_speed.
    Vector<double>  current_solution_extended;

    // Initial guess for Newton's iterations.
    SolutionStruct  initial_guess;

    TimerOutput     computing_timer;
  };

} // namespace TravelingWave

#endif