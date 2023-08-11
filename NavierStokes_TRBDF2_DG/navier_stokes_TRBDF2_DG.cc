/* Author: Giuseppe Orlando, 2022. */

// We start by including all the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/fe/component_mask.h>

#include <deal.II/base/timer.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/meshworker/mesh_loop.h>

#include "runtime_parameters.h"
#include "equation_data.h"

namespace MatrixFreeTools {
  using namespace dealii;

  template<int dim, typename Number, typename VectorizedArrayType>
  void compute_diagonal(const MatrixFree<dim, Number, VectorizedArrayType>&                            matrix_free,
                        LinearAlgebra::distributed::Vector<Number>&                                    diagonal_global,
                        const std::function<void(const MatrixFree<dim, Number, VectorizedArrayType>&,
                                                 LinearAlgebra::distributed::Vector<Number>&,
                                                 const unsigned int&,
                                                 const std::pair<unsigned int, unsigned int>&)>& 	     cell_operation,
                        const std::function<void(const MatrixFree<dim, Number, VectorizedArrayType>&,
                                                 LinearAlgebra::distributed::Vector<Number>&,
                                                 const unsigned int&,
                                                 const std::pair<unsigned int, unsigned int>&)>& 	     face_operation,
                        const std::function<void(const MatrixFree<dim, Number, VectorizedArrayType>&,
                                                 LinearAlgebra::distributed::Vector<Number>&,
                                                 const unsigned int&,
                                                 const std::pair<unsigned int, unsigned int>&)>& 	     boundary_operation,
                        const unsigned int                                                             dof_no = 0) {
    // initialize vector
    matrix_free.initialize_dof_vector(diagonal_global, dof_no);

    const unsigned int dummy = 0;

    matrix_free.loop(cell_operation, face_operation, boundary_operation,
                     diagonal_global, dummy, false,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }
}

// We include the code in a suitable namespace:
//
namespace NS_TRBDF2 {
  using namespace dealii;

  // The following class is an auxiliary one for post-processing of the vorticity
  //
  template<int dim>
  class PostprocessorVorticity: public DataPostprocessor<dim> {
  public:
    virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
                                       std::vector<Vector<double>>&                computed_quantities) const override;

    virtual std::vector<std::string> get_names() const override;

    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    virtual UpdateFlags get_needed_update_flags() const override;
  };

  // This function evaluates the vorticty in both 2D and 3D cases
  //
  template <int dim>
  void PostprocessorVorticity<dim>::evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
                                                          std::vector<Vector<double>>&                computed_quantities) const {
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    /*--- Check the correctness of all data structres ---*/
    Assert(inputs.solution_gradients.size() == n_quadrature_points, ExcInternalError());
    Assert(computed_quantities.size() == n_quadrature_points, ExcInternalError());

    Assert(inputs.solution_values[0].size() == dim, ExcInternalError());

    if(dim == 2) {
      Assert(computed_quantities[0].size() == 1, ExcInternalError());
    }
    else {
      Assert(computed_quantities[0].size() == dim, ExcInternalError());
    }

    /*--- Compute the vorticty ---*/
    if(dim == 2) {
      for(unsigned int q = 0; q < n_quadrature_points; ++q)
        computed_quantities[q](0) = inputs.solution_gradients[q][1][0] - inputs.solution_gradients[q][0][1];
    }
    else {
      for(unsigned int q = 0; q < n_quadrature_points; ++q) {
        computed_quantities[q](0) = inputs.solution_gradients[q][2][1] - inputs.solution_gradients[q][1][2];
        computed_quantities[q](1) = inputs.solution_gradients[q][0][2] - inputs.solution_gradients[q][2][0];
        computed_quantities[q](2) = inputs.solution_gradients[q][1][0] - inputs.solution_gradients[q][0][1];
      }
    }
  }

  // This auxiliary function is required by the base class DataProcessor and simply
  // sets the name for the output file
  //
  template<int dim>
  std::vector<std::string> PostprocessorVorticity<dim>::get_names() const {
    std::vector<std::string> names;
    names.emplace_back("vorticity");
    if(dim == 3) {
      names.emplace_back("vorticity");
      names.emplace_back("vorticity");
    }

    return names;
  }

  // This auxiliary function is required by the base class DataProcessor and simply
  // specifies if the vorticity is a scalar (2D) or a vector (3D)
  //
  template<int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  PostprocessorVorticity<dim>::get_data_component_interpretation() const {
    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
    if(dim == 2)
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    else {
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    }

    return interpretation;
  }

  // This auxiliary function is required by the base class DataProcessor and simply
  // sets which variables have to updated (only the gradients)
  //
  template<int dim>
  UpdateFlags PostprocessorVorticity<dim>::get_needed_update_flags() const {
    return update_gradients;
  }


  // The following structs are auxiliary objects for mesh refinement. ScratchData simply sets
  // the FEValues object
  //
  template <int dim>
  struct ScratchData {
    ScratchData(const FiniteElement<dim>& fe,
                const unsigned int        quadrature_degree,
                const UpdateFlags         update_flags): fe_values(fe, QGauss<dim>(quadrature_degree), update_flags) {}

    ScratchData(const ScratchData<dim>& scratch_data): fe_values(scratch_data.fe_values.get_fe(),
                                                                 scratch_data.fe_values.get_quadrature(),
                                                                 scratch_data.fe_values.get_update_flags()) {}
    FEValues<dim> fe_values;
  };


  // CopyData simply sets the cell index
  //
  struct CopyData {
    CopyData() : cell_index(numbers::invalid_unsigned_int), value(0.0) {}

    CopyData(const CopyData &) = default;

    unsigned int cell_index;
    double       value;
  };


  // @sect{ <code>NavierStokesProjectionOperator::NavierStokesProjectionOperator</code> }

  // The following class sets effecively the weak formulation of the problems for the different stages
  // and for both velocity and pressure.
  // The template parameters are the dimnesion of the problem, the polynomial degree for the pressure,
  // the polynomial degree for the velocity, the number of quadrature points for integrals for the pressure step,
  // the number of quadrature points for integrals for the velocity step, the type of vector for storage and the type
  // of floating point data (in general double or float for preconditioners structures if desired).
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  class NavierStokesProjectionOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    NavierStokesProjectionOperator();

    NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data);

    void set_dt(const double time_step);

    void set_TR_BDF2_stage(const unsigned int stage);

    void set_NS_stage(const unsigned int stage);

    void set_u_extr(const Vec& src);

    void vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_grad_p_projection(Vec& dst, const Vec& src) const;

    virtual void compute_diagonal() override;

  protected:
    double       Re;
    double       dt;

    /*--- Parameters of time-marching scheme ---*/
    double       gamma;
    double       a31;
    double       a32;
    double       a33;

    unsigned int TR_BDF2_stage; /*--- Flag to denote at which stage of the TR-BDF2 are ---*/
    unsigned int NS_stage;      /*--- Flag to denote at which stage of NS solution inside each TR-BDF2 stage we are
                                      (solution of the velocity or of the pressure)---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override;

  private:
    /*--- Auxiliary variable for the TR stage
          (just to avoid to report a lot of 0.5 and for my personal choice to be coherent with the article) ---*/
    const double a21 = 0.5;
    const double a22 = 0.5;

    /*--- Penalty method parameters, theta = 1 means SIP, while C_p and C_u are the penalization coefficients ---*/
    const double theta_v = 1.0;
    const double theta_p = 1.0;
    const double C_p = 1.0*(fe_degree_p + 1)*(fe_degree_p + 1);
    const double C_u = 1.0*(fe_degree_v + 1)*(fe_degree_v + 1);

    Vec                          u_extr; /*--- Auxiliary variable to update the extrapolated velocity ---*/

    EquationData::Velocity<dim>  vel_boundary_inflow; /*--- Auxiliary variable to impose velocity boundary conditions ---*/

    /*--- The following functions basically assemble the linear and bilinear forms. Their syntax is due to
          the base class MatrixFreeOperators::Base ---*/
    void assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const Vec&                                   src,
                                                  const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const unsigned int&                          src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const unsigned int&                          src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const;
  };


  // We start with the default constructor. It is important for MultiGrid, so it is fundamental
  // to properly set the parameters of the time scheme.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  NavierStokesProjectionOperator():
    MatrixFreeOperators::Base<dim, Vec>(), Re(), dt(), gamma(2.0 - std::sqrt(2.0)), a31((1.0 - gamma)/(2.0*(2.0 - gamma))),
                                           a32(a31), a33(1.0/(2.0 - gamma)), TR_BDF2_stage(1), NS_stage(1), u_extr() {}


  // We focus now on the constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data):
    MatrixFreeOperators::Base<dim, Vec>(), Re(data.Reynolds), dt(data.dt),
                                           gamma(2.0 - std::sqrt(2.0)), a31((1.0 - gamma)/(2.0*(2.0 - gamma))),
                                           a32(a31), a33(1.0/(2.0 - gamma)), TR_BDF2_stage(1), NS_stage(1), u_extr(),
                                           vel_boundary_inflow(data.initial_time) {}


  // Setter of time-step (called by Multigrid and in case a smaller time-step towards the end is needed)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of TR-BDF2 stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  set_TR_BDF2_stage(const unsigned int stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());

    TR_BDF2_stage = stage;
  }


  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  set_NS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 4);
    Assert(stage > 0, ExcInternalError());

    NS_stage = stage;
  }


  // Setter of extrapolated velocity for different stages
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  set_u_extr(const Vec& src) {
    u_extr = src;
    u_extr.update_ghost_values();
  }


  // We are in a DG-MatrixFree framework, so it is convenient to compute separately cell contribution,
  // internal faces contributions and boundary faces contributions. We start by
  // assembling the rhs cell term for the velocity.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the old velocity, the
      extrapolated velocity and the old pressure. 'phi' will be used only to submit the result.
      The second argument specifies which dof handler has to be used (in this implementation 0 stands for
      velocity and 1 for pressure). ---*/
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_old(data, 0),
                                                                   phi_old_extr(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press(data, 1);

      /*--- We loop over the cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        /*--- Now we need to assign the current cell to each FEEvaluation object and then to specify which src vector
        it has to read (the proper order is clearly delegated to the user, which has to pay attention in the function
        call to be coherent). ---*/
        phi_old.reinit(cell);
        phi_old.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
                                                           /*--- The 'gather_evaluate' function reads data from the vector.
                                                           The second and third parameter specifies if you want to read
                                                           values and/or derivative related quantities ---*/
        phi_old_extr.reinit(cell);
        phi_old_extr.gather_evaluate(src[1], EvaluationFlags::values);
        phi_old_press.reinit(cell);
        phi_old_press.gather_evaluate(src[2], EvaluationFlags::values);
        phi.reinit(cell);

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_n                = phi_old.get_value(q);
          const auto& grad_u_n           = phi_old.get_gradient(q);
          const auto& u_n_gamma_ov_2     = phi_old_extr.get_value(q);
          const auto& tensor_product_u_n = outer_product(u_n, u_n_gamma_ov_2);
          const auto& p_n                = phi_old_press.get_value(q);
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = p_n;

          phi.submit_value(1.0/(gamma*dt)*u_n, q); /*--- 'submit_value' contains quantites that we want to test against the
                                                          test function ---*/
          phi.submit_gradient(-a21/Re*grad_u_n + a21*tensor_product_u_n + p_n_times_identity, q);
          /*--- 'submit_gradient' contains quantites that we want to test against the gradient of test function ---*/
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        /*--- 'integrate_scatter' is the responsible of distributing into dst.
              The flag parameter specifies if we are testing against the test function and/or its gradient ---*/
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_old(data, 0),
                                                                   phi_int(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press(data, 1);

      /*--- We loop over the cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old.reinit(cell);
        phi_old.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_int.reinit(cell);
        phi_int.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_press.reinit(cell);
        phi_old_press.gather_evaluate(src[2], EvaluationFlags::values);
        phi.reinit(cell);

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_n                      = phi_old.get_value(q);
          const auto& grad_u_n                 = phi_old.get_gradient(q);
          const auto& u_n_gamma                = phi_int.get_value(q);
          const auto& grad_u_n_gamma           = phi_int.get_gradient(q);
          const auto& tensor_product_u_n       = outer_product(u_n, u_n);
          const auto& tensor_product_u_n_gamma = outer_product(u_n_gamma, u_n_gamma);
          const auto& p_n                      = phi_old_press.get_value(q);
          auto p_n_times_identity              = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = p_n;

          phi.submit_value(1.0/((1.0 - gamma)*dt)*u_n_gamma, q);
          phi.submit_gradient(a32*tensor_product_u_n_gamma + a31*tensor_product_u_n -
                              a32/Re*grad_u_n_gamma - a31/Re*grad_u_n + p_n_times_identity, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // The followinf function assembles rhs face term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. In this case
      we are at the face between two elements and this is the reason of 'FEFaceEvaluation'. It contains an extra
      input argument, the second one, that specifies if it is from 'interior' or not---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_old_p(data, true, 0),
                                                                       phi_old_m(data, false, 0),
                                                                       phi_old_extr_p(data, true, 0),
                                                                       phi_old_extr_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press_p(data, true, 1),
                                                                       phi_old_press_m(data, false, 1);

      /*--- We loop over the faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old_p.reinit(face);
        phi_old_p.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_m.reinit(face);
        phi_old_m.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_extr_p.reinit(face);
        phi_old_extr_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_old_extr_m.reinit(face);
        phi_old_extr_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_old_press_p.reinit(face);
        phi_old_press_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_old_press_m.reinit(face);
        phi_old_press_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                 = phi_p.get_normal_vector(q); /*--- The normal vector is the same
                                                                                 for both phi_p and phi_m. If the face is interior,
                                                                                 it correspond to the outer normal ---*/

          const auto& avg_grad_u_old         = 0.5*(phi_old_p.get_gradient(q) + phi_old_m.get_gradient(q));
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(phi_old_p.get_value(q), phi_old_extr_p.get_value(q)) +
                                                    outer_product(phi_old_m.get_value(q), phi_old_extr_m.get_value(q)));
          const auto& avg_p_old              = 0.5*(phi_old_press_p.get_value(q) + phi_old_press_m.get_value(q));

          phi_p.submit_value((a21/Re*avg_grad_u_old - a21*avg_tensor_product_u_n)*n_plus - avg_p_old*n_plus, q);
          phi_m.submit_value(-(a21/Re*avg_grad_u_old - a21*avg_tensor_product_u_n)*n_plus + avg_p_old*n_plus, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_old_p(data, true, 0),
                                                                       phi_old_m(data, false, 0),
                                                                       phi_int_p(data, true, 0),
                                                                       phi_int_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press_p(data, true, 1),
                                                                       phi_old_press_m(data, false, 1);

      /*--- We loop over the faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++ face) {
        phi_old_p.reinit(face);
        phi_old_p.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_m.reinit(face);
        phi_old_m.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_int_p.reinit(face);
        phi_int_p.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_int_m.reinit(face);
        phi_int_m.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_press_p.reinit(face);
        phi_old_press_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_old_press_m.reinit(face);
        phi_old_press_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                       = phi_p.get_normal_vector(q);

          const auto& avg_grad_u_old               = 0.5*(phi_old_p.get_gradient(q) + phi_old_m.get_gradient(q));
          const auto& avg_grad_u_int               = 0.5*(phi_int_p.get_gradient(q) + phi_int_m.get_gradient(q));
          const auto& avg_tensor_product_u_n       = 0.5*(outer_product(phi_old_p.get_value(q), phi_old_p.get_value(q)) +
                                                          outer_product(phi_old_m.get_value(q), phi_old_m.get_value(q)));
          const auto& avg_tensor_product_u_n_gamma = 0.5*(outer_product(phi_int_p.get_value(q), phi_int_p.get_value(q)) +
                                                          outer_product(phi_int_m.get_value(q), phi_int_m.get_value(q)));
          const auto& avg_p_old                    = 0.5*(phi_old_press_p.get_value(q) + phi_old_press_m.get_value(q));

          phi_p.submit_value((a31/Re*avg_grad_u_old + a32/Re*avg_grad_u_int -
                              a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus - avg_p_old*n_plus, q);
          phi_m.submit_value(-(a31/Re*avg_grad_u_old + a32/Re*avg_grad_u_int -
                               a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus + avg_p_old*n_plus, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // The followinf function assembles rhs boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. Clearly on the boundary
      the second argument has to be true. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_old(data, true, 0),
                                                                       phi_old_extr(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press(data, true, 1);

      /*--- We loop over the faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old.reinit(face);
        phi_old.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(src[1], EvaluationFlags::values);
        phi_old_press.reinit(face);
        phi_old_press.gather_evaluate(src[2], EvaluationFlags::values);
        phi.reinit(face);

        const auto boundary_id = data.get_boundary_id(face); /*--- Get the id in order to impose the proper boundary condition ---*/
        const auto coef_jump   = (boundary_id == 1) ? 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double aux_coeff = (boundary_id == 1) ? 0.0 : 1.0;

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus             = phi.get_normal_vector(q);

          const auto& grad_u_old         = phi_old.get_gradient(q);
          const auto& tensor_product_u_n = outer_product(phi_old.get_value(q), phi_old_extr.get_value(q));
          const auto& p_old              = phi_old_press.get_value(q);
          const auto& point_vectorized   = phi.quadrature_point(q);
          auto u_int_m                   = Tensor<1, dim, VectorizedArray<Number>>();
          if(boundary_id == 0) {
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point; /*--- The point returned by the 'quadrature_point' function is not an instance of Point
                                      and so it is not ready to be directly used. We need to pay attention to the
                                      vectorization ---*/
              for(unsigned int d = 0; d < dim; ++d)
                point[d] = point_vectorized[d][v];
              for(unsigned int d = 0; d < dim; ++d)
                u_int_m[d][v] = vel_boundary_inflow.value(point, d);
            }
          }
          const auto tensor_product_u_int_m = outer_product(u_int_m, phi_old_extr.get_value(q));
          const auto lambda                 = (boundary_id == 1) ? 0.0 : std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));

          phi.submit_value((a21/Re*grad_u_old - a21*tensor_product_u_n)*n_plus - p_old*n_plus +
                           a22/Re*2.0*coef_jump*u_int_m -
                           aux_coeff*a22*tensor_product_u_int_m*n_plus + a22*lambda*u_int_m, q);
          phi.submit_normal_derivative(-aux_coeff*theta_v*a22/Re*u_int_m, q); /*--- This is equivalent to multiply to the gradient
                                                                                    with outer product and use 'submit_gradient' ---*/
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_old(data, true, 0),
                                                                       phi_int(data, true, 0),
                                                                       phi_int_extr(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_old_press(data, true, 1);

      /*--- We loop over the faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++ face) {
        phi_old.reinit(face);
        phi_old.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_int.reinit(face);
        phi_int.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_press.reinit(face);
        phi_old_press.gather_evaluate(src[2], EvaluationFlags::values);
        phi_int_extr.reinit(face);
        phi_int_extr.gather_evaluate(src[3], EvaluationFlags::values);
        phi.reinit(face);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = (boundary_id == 1) ? 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double aux_coeff = (boundary_id == 1) ? 0.0 : 1.0;

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                   = phi.get_normal_vector(q);

          const auto& grad_u_old               = phi_old.get_gradient(q);
          const auto& grad_u_int               = phi_int.get_gradient(q);
          const auto& tensor_product_u_n       = outer_product(phi_old.get_value(q), phi_old.get_value(q));
          const auto& tensor_product_u_n_gamma = outer_product(phi_int.get_value(q), phi_int.get_value(q));
          const auto& p_old                    = phi_old_press.get_value(q);
          const auto& point_vectorized         = phi.quadrature_point(q);
          auto u_m                             = Tensor<1, dim, VectorizedArray<Number>>();
          if(boundary_id == 0) {
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point;
              for(unsigned int d = 0; d < dim; ++d)
                point[d] = point_vectorized[d][v];
              for(unsigned int d = 0; d < dim; ++d)
                u_m[d][v] = vel_boundary_inflow.value(point, d);
            }
          }
          const auto tensor_product_u_m = outer_product(u_m, phi_int_extr.get_value(q));
          const auto lambda             = (boundary_id == 1) ? 0.0 : std::abs(scalar_product(phi_int_extr.get_value(q), n_plus));

          phi.submit_value((a31/Re*grad_u_old + a32/Re*grad_u_int -
                           a31*tensor_product_u_n - a32*tensor_product_u_n_gamma)*n_plus - p_old*n_plus +
                           a33/Re*2.0*coef_jump*u_m -
                           aux_coeff*a33*tensor_product_u_m*n_plus + a33*lambda*u_m, q);
          phi.submit_normal_derivative(-aux_coeff*theta_v*a33/Re*u_m, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Put together all the previous steps for velocity. This is done automatically by the loop function of 'MatrixFree' class
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const {
    for(auto& vec : src)
      vec.update_ghost_values();

    this->data->loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_velocity,
                     &NavierStokesProjectionOperator::assemble_rhs_face_term_velocity,
                     &NavierStokesProjectionOperator::assemble_rhs_boundary_term_velocity,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Now we focus on computing the rhs for the projection step for the pressure with the same ratio.
  // The following function assembles rhs cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities.
          The third parameter specifies that we want to use the second quadrature formula stored. ---*/
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number>   phi(data, 1, 1),
                                                                 phi_old(data, 1, 1);
    FEEvaluation<dim, fe_degree_v, n_q_points_1d_p, dim, Number> phi_proj(data, 0, 1);

    const double coeff   = (TR_BDF2_stage == 1) ? 1.0e6*gamma*dt*gamma*dt : 1.0e6*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    const double coeff_2 = (TR_BDF2_stage == 1) ? gamma*dt : (1.0 - gamma)*dt;

    /*--- We loop over cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_proj.reinit(cell);
      phi_proj.gather_evaluate(src[0], EvaluationFlags::values);
      phi_old.reinit(cell);
      phi_old.gather_evaluate(src[1], EvaluationFlags::values);
      phi.reinit(cell);

      /*--- Now we loop over all the quadrature points to compute the integrals ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& u_star_star = phi_proj.get_value(q);
        const auto& p_old       = phi_old.get_value(q);

        phi.submit_value(1.0/coeff*p_old, q);
        phi.submit_gradient(1.0/coeff_2*u_star_star, q);
      }
      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // The following function assembles rhs face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number>   phi_p(data, true, 1, 1),
                                                                     phi_m(data, false, 1, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_p, dim, Number> phi_proj_p(data, true, 0, 1),
                                                                     phi_proj_m(data, false, 0, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1.0/(gamma*dt) : 1.0/((1.0 - gamma)*dt);

    /*--- We loop over faces in the range ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_proj_p.reinit(face);
      phi_proj_p.gather_evaluate(src[0], EvaluationFlags::values);
      phi_proj_m.reinit(face);
      phi_proj_m.gather_evaluate(src[0], EvaluationFlags::values);
      phi_p.reinit(face);
      phi_m.reinit(face);

      /*--- Now we loop over all the quadrature points to compute the integrals ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus           = phi_p.get_normal_vector(q);
        const auto& avg_u_star_star  = 0.5*(phi_proj_p.get_value(q) + phi_proj_m.get_value(q));

        phi_p.submit_value(-coeff*scalar_product(avg_u_star_star, n_plus), q);
        phi_m.submit_value(coeff*scalar_product(avg_u_star_star, n_plus), q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // The following function assembles rhs boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number>   phi(data, true, 1, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_p, dim, Number> phi_proj(data, true, 0, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1.0/(gamma*dt) : 1.0/((1.0 - gamma)*dt);

    /*--- We loop over faces in the range ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_proj.reinit(face);
      phi_proj.gather_evaluate(src[0], EvaluationFlags::values);
      phi.reinit(face);

      /*--- Now we loop over all the quadrature points to compute the integrals ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus = phi.get_normal_vector(q);

        phi.submit_value(-coeff*scalar_product(phi_proj.get_value(q), n_plus), q);
      }
      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const {
    for(auto& vec : src)
      vec.update_ghost_values();

    this->data->loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_pressure,
                     &NavierStokesProjectionOperator::assemble_rhs_face_term_pressure,
                     &NavierStokesProjectionOperator::assemble_rhs_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Now we need to build the 'matrices', i.e. the bilinear forms. We start by
  // assembling the cell term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. Moreover 'phi' in
      this case serves for a bilinear form and so it will not used only to submit but also to read the src ---*/
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_old_extr(data, 0);

      /*--- We loop over all cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_extr.reinit(cell);
        phi_old_extr.gather_evaluate(u_extr, EvaluationFlags::values);

        /*--- Now we loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_int                = phi.get_value(q);
          const auto& grad_u_int           = phi.get_gradient(q);
          const auto& u_n_gamma_ov_2       = phi_old_extr.get_value(q);
          const auto& tensor_product_u_int = outer_product(u_int, u_n_gamma_ov_2);

          phi.submit_value(1.0/(gamma*dt)*u_int, q);
          phi.submit_gradient(-a22*tensor_product_u_int + a22/Re*grad_u_int, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_int_extr(data, 0);

      /*--- We loop over all cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_int_extr.reinit(cell);
        phi_int_extr.gather_evaluate(u_extr, EvaluationFlags::values);

        /*--- Now we loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_curr                   = phi.get_value(q);
          const auto& grad_u_curr              = phi.get_gradient(q);
          const auto& u_n1_int                 = phi_int_extr.get_value(q);
          const auto& tensor_product_u_curr    = outer_product(u_curr, u_n1_int);

          phi.submit_value(1.0/((1.0 - gamma)*dt)*u_curr, q);
          phi.submit_gradient(-a33*tensor_product_u_curr + a33/Re*grad_u_curr, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // The following function assembles face term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_face_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_old_extr_p(data, true, 0),
                                                                       phi_old_extr_m(data, false, 0);

      /*--- We loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_extr_p.reinit(face);
        phi_old_extr_p.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_old_extr_m.reinit(face);
        phi_old_extr_m.gather_evaluate(u_extr, EvaluationFlags::values);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Now we loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                   = phi_p.get_normal_vector(q);

          const auto& avg_grad_u_int           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_u_int               = phi_p.get_value(q) - phi_m.get_value(q);
          const auto& avg_tensor_product_u_int = 0.5*(outer_product(phi_p.get_value(q), phi_old_extr_p.get_value(q)) +
                                                      outer_product(phi_m.get_value(q), phi_old_extr_m.get_value(q)));
          const auto  lambda                   = std::max(std::abs(scalar_product(phi_old_extr_p.get_value(q), n_plus)),
                                                          std::abs(scalar_product(phi_old_extr_m.get_value(q), n_plus)));

          phi_p.submit_value(a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) +
                             a22*avg_tensor_product_u_int*n_plus + 0.5*a22*lambda*jump_u_int, q);
          phi_m.submit_value(-a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) -
                              a22*avg_tensor_product_u_int*n_plus - 0.5*a22*lambda*jump_u_int, q);
          phi_p.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u_int, q);
          phi_m.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u_int, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        phi_m.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_extr_p(data, true, 0),
                                                                       phi_extr_m(data, false, 0);

      /*--- We loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_extr_p.reinit(face);
        phi_extr_p.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_extr_m.reinit(face);
        phi_extr_m.gather_evaluate(u_extr, EvaluationFlags::values);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Now we loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus               = phi_p.get_normal_vector(q);

          const auto& avg_grad_u           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_u               = phi_p.get_value(q) - phi_m.get_value(q);
          const auto& avg_tensor_product_u = 0.5*(outer_product(phi_p.get_value(q), phi_extr_p.get_value(q)) +
                                                  outer_product(phi_m.get_value(q), phi_extr_m.get_value(q)));
          const auto  lambda               = std::max(std::abs(scalar_product(phi_extr_p.get_value(q), n_plus)),
                                                      std::abs(scalar_product(phi_extr_m.get_value(q), n_plus)));

          phi_p.submit_value(a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                             a33*avg_tensor_product_u*n_plus + 0.5*a33*lambda*jump_u, q);
          phi_m.submit_value(-a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                              a33*avg_tensor_product_u*n_plus - 0.5*a33*lambda*jump_u, q);
          phi_p.submit_normal_derivative(-theta_v*a33/Re*0.5*jump_u, q);
          phi_m.submit_normal_derivative(-theta_v*a33/Re*0.5*jump_u, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        phi_m.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // The following function assembles boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_old_extr(data, true, 0);

      /*--- We loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(u_extr, EvaluationFlags::values);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);

        /*--- The application of the mirror principle is not so trivial because we have a Dirichlet condition
              on a single component for the outflow; so we distinguish the two cases ---*/
        if(boundary_id != 1) {
          const double coef_trasp = 0.0;

          /*--- Now we loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus               = phi.get_normal_vector(q);
            const auto& grad_u_int           = phi.get_gradient(q);
            const auto& u_int                = phi.get_value(q);
            const auto& tensor_product_u_int = outer_product(phi.get_value(q), phi_old_extr.get_value(q));
            const auto& lambda               = std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));

            phi.submit_value(a22/Re*(-grad_u_int*n_plus + 2.0*coef_jump*u_int) +
                             a22*coef_trasp*tensor_product_u_int*n_plus + a22*lambda*u_int, q);
            phi.submit_normal_derivative(-theta_v*a22/Re*u_int, q);
          }
          phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
        else {
          /*--- Now we loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus               = phi.get_normal_vector(q);
            const auto& grad_u_int           = phi.get_gradient(q);
            const auto& u_int                = phi.get_value(q);
            const auto& lambda               = std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));

            const auto& point_vectorized     = phi.quadrature_point(q);
            auto u_int_m                     = u_int;
            auto grad_u_int_m                = grad_u_int;
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point;
              for(unsigned int d = 0; d < dim; ++d)
                point[d] = point_vectorized[d][v];

              u_int_m[1][v] = -u_int_m[1][v];

              grad_u_int_m[0][0][v] = -grad_u_int_m[0][0][v];
              grad_u_int_m[0][1][v] = -grad_u_int_m[0][1][v];
            }

            phi.submit_value(a22/Re*(-(0.5*(grad_u_int + grad_u_int_m))*n_plus + coef_jump*(u_int - u_int_m)) +
                             a22*outer_product(0.5*(u_int + u_int_m), phi_old_extr.get_value(q))*n_plus +
                             a22*0.5*lambda*(u_int - u_int_m), q);
            phi.submit_normal_derivative(-theta_v*a22/Re*(u_int - u_int_m), q);
          }
          phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_extr(data, true, 0);

      /*--- We loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_extr.reinit(face);
        phi_extr.gather_evaluate(u_extr, EvaluationFlags::values);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);

        if(boundary_id != 1) {
          const double coef_trasp = 0.0;

          /*--- Now we loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus           = phi.get_normal_vector(q);
            const auto& grad_u           = phi.get_gradient(q);
            const auto& u                = phi.get_value(q);
            const auto& tensor_product_u = outer_product(phi.get_value(q), phi_extr.get_value(q));
            const auto& lambda           = std::abs(scalar_product(phi_extr.get_value(q), n_plus));

            phi.submit_value(a33/Re*(-grad_u*n_plus + 2.0*coef_jump*u) +
                             a33*coef_trasp*tensor_product_u*n_plus + a33*lambda*u, q);
            phi.submit_normal_derivative(-theta_v*a33/Re*u, q);
          }
          phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
        else {
          /*--- Now we loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus           = phi.get_normal_vector(q);
            const auto& grad_u           = phi.get_gradient(q);
            const auto& u                = phi.get_value(q);
            const auto& lambda           = std::abs(scalar_product(phi_extr.get_value(q), n_plus));

            const auto& point_vectorized = phi.quadrature_point(q);
            auto u_m                     = u;
            auto grad_u_m                = grad_u;
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point;
              for(unsigned int d = 0; d < dim; ++d)
                point[d] = point_vectorized[d][v];

              u_m[1][v] = -u_m[1][v];

              grad_u_m[0][0][v] = -grad_u_m[0][0][v];
              grad_u_m[0][1][v] = -grad_u_m[0][1][v];
            }

            phi.submit_value(a33/Re*(-(0.5*(grad_u + grad_u_m))*n_plus + coef_jump*(u - u_m)) +
                             a33*outer_product(0.5*(u + u_m), phi_extr.get_value(q))*n_plus + a33*0.5*lambda*(u - u_m), q);
            phi.submit_normal_derivative(-theta_v*a33/Re*(u - u_m), q);
          }
          phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
      }
    }
  }


  // Next, we focus on 'matrices' to compute the pressure. We first assemble cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, 1, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1.0e6*gamma*dt*gamma*dt : 1.0e6*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    /*--- Loop over all cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      /*--- Now we loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_gradient(phi.get_gradient(q), q);
        phi.submit_value(1.0/coeff*phi.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // The following function assembles face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi_p(data, true, 1, 1),
                                                                   phi_m(data, false, 1, 1);

    /*--- Loop over all faces in the range ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      const auto coef_jump = C_p*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

      /*--- Loop over quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus        = phi_p.get_normal_vector(q);

        const auto& avg_grad_pres = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
        const auto& jump_pres     = phi_p.get_value(q) - phi_m.get_value(q);

        phi_p.submit_value(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres, q);
        phi_m.submit_value(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres, q);
        phi_p.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
        phi_m.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      phi_m.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // The following function assembles boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, true, 1, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      const auto boundary_id = data.get_boundary_id(face);

      if(boundary_id == 1) {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const auto coef_jump = C_p*std::abs((phi.get_normal_vector(0)*phi.inverse_jacobian(0))[dim - 1]);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus    = phi.get_normal_vector(q);

          const auto& grad_pres = phi.get_gradient(q);
          const auto& pres      = phi.get_value(q);

          phi.submit_value(-scalar_product(grad_pres, n_plus) + coef_jump*pres , q);
          phi.submit_normal_derivative(-theta_p*pres, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Before coding the 'apply_add' function, which is the one that will perform the loop, we focus on
  // the linear system that arises to project the gradient of the pressure into the velocity space.
  // The following function assembles rhs cell term for the projection of gradient of pressure. Since no
  // integration by parts is performed, only a cell term contribution is present.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_pres(data, 1);

    /*--- Loop over all cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres.reinit(cell);
      phi_pres.gather_evaluate(src, EvaluationFlags::gradients);
      phi.reinit(cell);

      /*--- Loop over quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi_pres.get_gradient(q), q);

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for porjection of pressure gradient. Here we loop only over cells
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  vmult_grad_p_projection(Vec& dst, const Vec& src) const {
    this->data->cell_loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_projection_grad_p,
                          this, dst, src, true);
  }


  // Assemble now cell term for the projection of gradient of pressure. This is nothing but a mass matrix
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0);

    /*--- Loop over all cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all previous steps. This is the overriden function that effectively performs the
  // matrix-vector multiplication.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    if(NS_stage == 1) {
      this->data->loop(&NavierStokesProjectionOperator::assemble_cell_term_velocity,
                       &NavierStokesProjectionOperator::assemble_face_term_velocity,
                       &NavierStokesProjectionOperator::assemble_boundary_term_velocity,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == 2) {
      this->data->loop(&NavierStokesProjectionOperator::assemble_cell_term_pressure,
                       &NavierStokesProjectionOperator::assemble_face_term_pressure,
                       &NavierStokesProjectionOperator::assemble_boundary_term_pressure,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == 3) {
      this->data->cell_loop(&NavierStokesProjectionOperator::assemble_cell_term_projection_grad_p,
                            this, dst, src, false); /*--- Since we have only a cell term contribution, we use cell_loop ---*/
    }
    else
      Assert(false, ExcNotImplemented());
  }


  // Finally, we focus on computing the diagonal for preconditioners and we start by assembling
  // the diagonal cell term for the velocity. Since we do not have access to the entries of the matrix,
  // in order to compute the element i, we test the matrix against a vector which is equal to 1 in position i and 0 elsewhere.
  // This is why 'src' will result as unused.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_old_extr(data, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      /*--- Build a vector of ones to be tested (here we will see the velocity as a whole vector, since
                                                 dof_handler_velocity is vectorial and so the dof values are vectors). ---*/
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      /*--- Loop over cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_old_extr.reinit(cell);
        phi_old_extr.gather_evaluate(u_extr, EvaluationFlags::values);
        phi.reinit(cell);

        /*--- Loop over dofs ---*/
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j); /*--- Set all dofs to zero ---*/
          phi.submit_dof_value(tmp, i); /*--- Set dof i equal to one ---*/
          phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          /*--- Loop over quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& u_int                = phi.get_value(q);
            const auto& grad_u_int           = phi.get_gradient(q);
            const auto& u_n_gamma_ov_2       = phi_old_extr.get_value(q);
            const auto& tensor_product_u_int = outer_product(u_int, u_n_gamma_ov_2);

            phi.submit_value(1.0/(gamma*dt)*u_int, q);
            phi.submit_gradient(-a22*tensor_product_u_int + a22/Re*grad_u_int, q);
          }
          phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_int_extr(data, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      /*--- Loop over cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_int_extr.reinit(cell);
        phi_int_extr.gather_evaluate(u_extr, EvaluationFlags::values);
        phi.reinit(cell);

        /*--- Loop over dofs ---*/
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi.submit_dof_value(tmp, i);
          phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          /*--- Loop over quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& u_curr                   = phi.get_value(q);
            const auto& grad_u_curr              = phi.get_gradient(q);
            const auto& u_n1_int                 = phi_int_extr.get_value(q);
            const auto& tensor_product_u_curr    = outer_product(u_curr, u_n1_int);

            phi.submit_value(1.0/((1.0 - gamma)*dt)*u_curr, q);
            phi.submit_gradient(-a33*tensor_product_u_curr + a33/Re*grad_u_curr, q);
          }
          phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
      }
    }
  }


  // The following function assembles diagonal face term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_old_extr_p(data, true, 0),
                                                                       phi_old_extr_m(data, false, 0);

      AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component); /*--- We just assert for safety that dimension match,
                                                                                in the sense that we have selected the proper
                                                                                space ---*/
      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal_p(phi_p.dofs_per_component),
                                                             diagonal_m(phi_m.dofs_per_component);

      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0); /*--- We build the usal vector of ones that we will use as dof value ---*/

      /*--- Now we loop over faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old_extr_p.reinit(face);
        phi_old_extr_p.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_old_extr_m.reinit(face);
        phi_old_extr_m.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_p.reinit(face);
        phi_m.reinit(face);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Loop over dofs. We will set all equal to zero apart from the current one ---*/
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
            phi_p.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi_m.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          }
          phi_p.submit_dof_value(tmp, i);
          phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
          phi_m.submit_dof_value(tmp, i);
          phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          /*--- Loop over quadrature points to compute the integral ---*/
          for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
            const auto& n_plus                   = phi_p.get_normal_vector(q);
            const auto& avg_grad_u_int           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
            const auto& jump_u_int               = phi_p.get_value(q) - phi_m.get_value(q);
            const auto& avg_tensor_product_u_int = 0.5*(outer_product(phi_p.get_value(q), phi_old_extr_p.get_value(q)) +
                                                        outer_product(phi_m.get_value(q), phi_old_extr_m.get_value(q)));
            const auto  lambda                   = std::max(std::abs(scalar_product(phi_old_extr_p.get_value(q), n_plus)),
                                                            std::abs(scalar_product(phi_old_extr_m.get_value(q), n_plus)));

            phi_p.submit_value(a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) +
                               a22*avg_tensor_product_u_int*n_plus + 0.5*a22*lambda*jump_u_int , q);
            phi_m.submit_value(-a22/Re*(-avg_grad_u_int*n_plus + coef_jump*jump_u_int) -
                               a22*avg_tensor_product_u_int*n_plus - 0.5*a22*lambda*jump_u_int, q);
            phi_p.submit_normal_derivative(-theta_v*0.5*a22/Re*jump_u_int, q);
            phi_m.submit_normal_derivative(-theta_v*0.5*a22/Re*jump_u_int, q);
          }
          phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal_p[i] = phi_p.get_dof_value(i);
          phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal_m[i] = phi_m.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          phi_p.submit_dof_value(diagonal_p[i], i);
          phi_m.submit_dof_value(diagonal_m[i], i);
        }
        phi_p.distribute_local_to_global(dst);
        phi_m.distribute_local_to_global(dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_extr_p(data, true, 0),
                                                                       phi_extr_m(data, false, 0);

      AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal_p(phi_p.dofs_per_component),
                                                             diagonal_m(phi_m.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      /*--- Now we loop over faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_extr_p.reinit(face);
        phi_extr_p.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_extr_m.reinit(face);
        phi_extr_m.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_p.reinit(face);
        phi_m.reinit(face);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Loop over dofs. We will set all equal to zero apart from the current one ---*/
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
            phi_p.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi_m.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          }
          phi_p.submit_dof_value(tmp, i);
          phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
          phi_m.submit_dof_value(tmp, i);
          phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          /*--- Loop over quadrature points to compute the integral ---*/
          for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
            const auto& n_plus               = phi_p.get_normal_vector(q);
            const auto& avg_grad_u           = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
            const auto& jump_u               = phi_p.get_value(q) - phi_m.get_value(q);
            const auto& avg_tensor_product_u = 0.5*(outer_product(phi_p.get_value(q), phi_extr_p.get_value(q)) +
                                                    outer_product(phi_m.get_value(q), phi_extr_m.get_value(q)));
            const auto  lambda               = std::max(std::abs(scalar_product(phi_extr_p.get_value(q), n_plus)),
                                                        std::abs(scalar_product(phi_extr_m.get_value(q), n_plus)));

            phi_p.submit_value(a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                               a33*avg_tensor_product_u*n_plus + 0.5*a33*lambda*jump_u, q);
            phi_m.submit_value(-a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                               a33*avg_tensor_product_u*n_plus - 0.5*a33*lambda*jump_u, q);
            phi_p.submit_normal_derivative(-theta_v*0.5*a33/Re*jump_u, q);
            phi_m.submit_normal_derivative(-theta_v*0.5*a33/Re*jump_u, q);
          }
          phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal_p[i] = phi_p.get_dof_value(i);
          phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal_m[i] = phi_m.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          phi_p.submit_dof_value(diagonal_p[i], i);
          phi_m.submit_dof_value(diagonal_m[i], i);
        }
        phi_p.distribute_local_to_global(dst);
        phi_m.distribute_local_to_global(dst);
      }
    }
  }


  // The following function assembles boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const unsigned int&                          ,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_old_extr(data, true, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      /*--- Loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_old_extr.reinit(face);
        phi_old_extr.gather_evaluate(u_extr, EvaluationFlags::values);
        phi.reinit(face);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);

        if(boundary_id != 1) {
          const double coef_trasp = 0.0;

          /*--- Loop over all dofs ---*/
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
              phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi.submit_dof_value(tmp, i);
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            /*--- Loop over quadrature points to compute the integral ---*/
            for(unsigned int q = 0; q < phi.n_q_points; ++q) {
              const auto& n_plus               = phi.get_normal_vector(q);
              const auto& grad_u_int           = phi.get_gradient(q);
              const auto& u_int                = phi.get_value(q);
              const auto& tensor_product_u_int = outer_product(phi.get_value(q), phi_old_extr.get_value(q));
              const auto& lambda               = std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));

              phi.submit_value(a22/Re*(-grad_u_int*n_plus + 2.0*coef_jump*u_int) +
                               a22*coef_trasp*tensor_product_u_int*n_plus + a22*lambda*u_int, q);
              phi.submit_normal_derivative(-theta_v*a22/Re*u_int, q);
            }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            diagonal[i] = phi.get_dof_value(i);
          }
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
            phi.submit_dof_value(diagonal[i], i);
          phi.distribute_local_to_global(dst);
        }
        else {
          /*--- Loop over all dofs ---*/
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
              phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi.submit_dof_value(tmp, i);
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            /*--- Loop over quadrature points to compute the integral ---*/
            for(unsigned int q = 0; q < phi.n_q_points; ++q) {
              const auto& n_plus               = phi.get_normal_vector(q);
              const auto& grad_u_int           = phi.get_gradient(q);
              const auto& u_int                = phi.get_value(q);
              const auto& lambda               = std::abs(scalar_product(phi_old_extr.get_value(q), n_plus));

              const auto& point_vectorized     = phi.quadrature_point(q);
              auto u_int_m                     = u_int;
              auto grad_u_int_m                = grad_u_int;
              for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
                Point<dim> point;
                for(unsigned int d = 0; d < dim; ++d)
                  point[d] = point_vectorized[d][v];

                u_int_m[1][v] = -u_int_m[1][v];

                grad_u_int_m[0][0][v] = -grad_u_int_m[0][0][v];
                grad_u_int_m[0][1][v] = -grad_u_int_m[0][1][v];
              }

              phi.submit_value(a22/Re*(-(0.5*(grad_u_int + grad_u_int_m))*n_plus + coef_jump*(u_int - u_int_m)) +
                               a22*outer_product(0.5*(u_int + u_int_m), phi_old_extr.get_value(q))*n_plus +
                               a22*0.5*lambda*(u_int - u_int_m), q);
              phi.submit_normal_derivative(-theta_v*a22/Re*(u_int - u_int_m), q);
            }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            diagonal[i] = phi.get_dof_value(i);
          }
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
            phi.submit_dof_value(diagonal[i], i);
          phi.distribute_local_to_global(dst);
        }
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_extr(data, true, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d)
        tmp[d] = make_vectorized_array<Number>(1.0);

      /*--- Loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_extr.reinit(face);
        phi_extr.gather_evaluate(u_extr, EvaluationFlags::values);
        phi.reinit(face);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);

        if(boundary_id != 1) {
          const double coef_trasp = 0.0;

          /*--- Loop over all dofs ---*/
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
              phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi.submit_dof_value(tmp, i);
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            /*--- Loop over quadrature points to compute the integral ---*/
            for(unsigned int q = 0; q < phi.n_q_points; ++q) {
              const auto& n_plus           = phi.get_normal_vector(q);
              const auto& grad_u           = phi.get_gradient(q);
              const auto& u                = phi.get_value(q);
              const auto& tensor_product_u = outer_product(phi.get_value(q), phi_extr.get_value(q));
              const auto& lambda           = std::abs(scalar_product(phi_extr.get_value(q), n_plus));

              phi.submit_value(a33/Re*(-grad_u*n_plus + 2.0*coef_jump*u) +
                               a33*coef_trasp*tensor_product_u*n_plus + a33*lambda*u, q);
              phi.submit_normal_derivative(-theta_v*a33/Re*u, q);
            }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            diagonal[i] = phi.get_dof_value(i);
          }
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
            phi.submit_dof_value(diagonal[i], i);
          phi.distribute_local_to_global(dst);
        }
        else {
          /*--- Loop over all dofs ---*/
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
              phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi.submit_dof_value(tmp, i);
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            /*--- Loop over quadrature points to compute the integral ---*/
            for(unsigned int q = 0; q < phi.n_q_points; ++q) {
              const auto& n_plus           = phi.get_normal_vector(q);
              const auto& grad_u           = phi.get_gradient(q);
              const auto& u                = phi.get_value(q);
              const auto& lambda           = std::abs(scalar_product(phi_extr.get_value(q), n_plus));

              const auto& point_vectorized = phi.quadrature_point(q);
              auto u_m                     = u;
              auto grad_u_m                = grad_u;
              for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
                Point<dim> point;
                for(unsigned int d = 0; d < dim; ++d)
                  point[d] = point_vectorized[d][v];

                u_m[1][v] = -u_m[1][v];

                grad_u_m[0][0][v] = -grad_u_m[0][0][v];
                grad_u_m[0][1][v] = -grad_u_m[0][1][v];
              }

              phi.submit_value(a33/Re*(-(0.5*(grad_u + grad_u_m))*n_plus + coef_jump*(u - u_m)) +
                               a33*outer_product(0.5*(u + u_m), phi_extr.get_value(q))*n_plus +
                               a33*0.5*lambda*(u - u_m), q);
              phi.submit_normal_derivative(-theta_v*a33/Re*(u - u_m), q);
            }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            diagonal[i] = phi.get_dof_value(i);
          }
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
            phi.submit_dof_value(diagonal[i], i);
          phi.distribute_local_to_global(dst);
        }
      }
    }
  }


  // Now we consider the pressure related bilinear forms. We first assemble diagonal cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, 1, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component); /*--- Here we are using dofs_per_component but
                                                                                   it coincides with dofs_per_cell since it is
                                                                                   scalar finite element space ---*/

    const double coeff = (TR_BDF2_stage == 1) ? 1e6*gamma*dt*gamma*dt : 1e6*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    /*--- Loop over all cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
          phi.submit_dof_value(VectorizedArray<Number>(), j); /*--- We set all dofs to zero ---*/
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i); /*--- Now we set the current one to 1; since it is scalar,
                                                                           we can directly use 'make_vectorized_array' without
                                                                           relying on 'Tensor' ---*/
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(1.0/coeff*phi.get_value(q), q);
          phi.submit_gradient(phi.get_gradient(q), q);
        }
        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal[i] = phi.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
        phi.submit_dof_value(diagonal[i], i);

      phi.distribute_local_to_global(dst);
    }
  }


  // The following function assembles diagonal face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi_p(data, true, 1, 1),
                                                                   phi_m(data, false, 1, 1);

    AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
    AlignedVector<VectorizedArray<Number>> diagonal_p(phi_p.dofs_per_component),
                                           diagonal_m(phi_m.dofs_per_component); /*--- Again, we just assert for safety that dimension
                                                                                       match, in the sense that we have selected
                                                                                       the proper space ---*/

    /*--- Loop over all faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_m.reinit(face);

      const auto coef_jump = C_p*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
          phi_p.submit_dof_value(VectorizedArray<Number>(), j);
          phi_m.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi_p.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_m.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points to compute the integral ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus        = phi_p.get_normal_vector(q);

          const auto& avg_grad_pres = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_pres     = phi_p.get_value(q) - phi_m.get_value(q);

          phi_p.submit_value(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres, q);
          phi_m.submit_value(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres, q);
          phi_p.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
          phi_m.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
        }
        phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_p[i] = phi_p.get_dof_value(i);
        phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_m[i] = phi_m.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        phi_p.submit_dof_value(diagonal_p[i], i);
        phi_m.submit_dof_value(diagonal_m[i], i);
      }
      phi_p.distribute_local_to_global(dst);
      phi_m.distribute_local_to_global(dst);
    }
  }


  // Eventually, we assemble diagonal boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const unsigned int&                          ,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, true, 1, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      const auto boundary_id = data.get_boundary_id(face);

      if(boundary_id == 1) {
        phi.reinit(face);

        const auto coef_jump = C_p*std::abs((phi.get_normal_vector(0)*phi.inverse_jacobian(0))[dim - 1]);

        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
            phi.submit_dof_value(VectorizedArray<Number>(), j);
          phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
          phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus    = phi.get_normal_vector(q);

            const auto& grad_pres = phi.get_gradient(q);
            const auto& pres      = phi.get_value(q);

            phi.submit_value(-scalar_product(grad_pres, n_plus) + 2.0*coef_jump*pres , q);
            phi.submit_normal_derivative(-theta_p*pres, q);
          }
          phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
      }
    }
  }


  // Put together all previous steps. We create a dummy auxliary vector that serves for the src input argument in
  // the previous functions that as we have seen before is unused. Then everything is done by the 'loop' function
  // and it is saved in the field 'inverse_diagonal_entries' already present in the base class. Anyway since there is
  // only one field, we need to resize properly depending on whether we are considering the velocity or the pressure.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  compute_diagonal() {
    Assert(NS_stage == 1 || NS_stage == 2, ExcInternalError());

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    if(NS_stage == 1) {
      ::MatrixFreeTools::compute_diagonal<dim, Number, VectorizedArray<Number>>
      (*(this->data),
       inverse_diagonal,
       [&](const auto& data, auto& dst, const auto& src, const auto& cell_range) {
         (this->assemble_diagonal_cell_term_velocity)(data, dst, src, cell_range);
       },
       [&](const auto& data, auto& dst, const auto& src, const auto& face_range) {
         (this->assemble_diagonal_face_term_velocity)(data, dst, src, face_range);
       },
       [&](const auto& data, auto& dst, const auto& src, const auto& boundary_range) {
         (this->assemble_diagonal_boundary_term_velocity)(data, dst, src, boundary_range);
       },
       0);
    }
    else if(NS_stage == 2) {
        ::MatrixFreeTools::compute_diagonal<dim, Number, VectorizedArray<Number>>
      (*(this->data),
       inverse_diagonal,
       [&](const auto& data, auto& dst, const auto& src, const auto& cell_range) {
         (this->assemble_diagonal_cell_term_pressure)(data, dst, src, cell_range);
       },
       [&](const auto& data, auto& dst, const auto& src, const auto& face_range) {
         (this->assemble_diagonal_face_term_pressure)(data, dst, src, face_range);
       },
       [&](const auto& data, auto& dst, const auto& src, const auto& boundary_range) {
         (this->assemble_diagonal_boundary_term_pressure)(data, dst, src, boundary_range);
       },
       1);
    }

    for(unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != 0.0,
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
    }
  }


  // @sect{The <code>NavierStokesProjection</code> class}

  // Now we are ready for the main class of the program. It implements the calls to the various steps
  // of the projection method for Navier-Stokes equations.
  //
  template<int dim>
  class NavierStokesProjection {
  public:
    NavierStokesProjection(RunTimeParameters::Data_Storage& data);

    void run(const bool verbose = false, const unsigned int output_interval = 10);

  protected:
    const double t_0;
    const double T;
    const double gamma;         //--- TR-BDF2 parameter
    unsigned int TR_BDF2_stage; //--- Flag to check at which current stage of TR-BDF2 are
    const double Re;
    double       dt;

    EquationData::Velocity<dim> vel_init;
    EquationData::Pressure<dim> pres_init; /*--- Instance of 'Velocity' and 'Pressure' classes to initialize. ---*/

    parallel::distributed::Triangulation<dim> triangulation;

    /*--- Finite Element spaces ---*/
    FESystem<dim> fe_velocity;
    FESystem<dim> fe_pressure;

    /*--- Handler for dofs ---*/
    DoFHandler<dim> dof_handler_velocity;
    DoFHandler<dim> dof_handler_pressure;

    /*--- Quadrature formulas for velocity and pressure, respectively ---*/
    QGauss<dim> quadrature_pressure;
    QGauss<dim> quadrature_velocity;

    /*--- Now we define all the vectors for the solution. We start from the pressure
          with p^n, p^(n+gamma) and a vector for rhs ---*/
    LinearAlgebra::distributed::Vector<double> pres_n;
    LinearAlgebra::distributed::Vector<double> pres_int;
    LinearAlgebra::distributed::Vector<double> rhs_p;

    /*--- Next, we move to the velocity, with u^n, u^(n-1), u^(n+gamma/2),
          u^(n+gamma) and other two auxiliary vectors as well as the rhs ---*/
    LinearAlgebra::distributed::Vector<double> u_n;
    LinearAlgebra::distributed::Vector<double> u_n_minus_1;
    LinearAlgebra::distributed::Vector<double> u_extr;
    LinearAlgebra::distributed::Vector<double> u_n_gamma;
    LinearAlgebra::distributed::Vector<double> u_star;
    LinearAlgebra::distributed::Vector<double> u_tmp;
    LinearAlgebra::distributed::Vector<double> rhs_u;
    LinearAlgebra::distributed::Vector<double> grad_pres_int;

    Vector<double> Linfty_error_per_cell_vel;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void create_triangulation(const unsigned int n_refines);

    void setup_dofs();

    void initialize();

    void interpolate_velocity();

    void diffusion_step();

    void projection_step();

    void project_grad(const unsigned int flag);

    double get_maximal_velocity();

    double get_maximal_difference_velocity();

    void output_results(const unsigned int step);

    void refine_mesh();

    void interpolate_max_res(const unsigned int level);

    void save_max_res();

  private:
    void compute_lift_and_drag();

    /*--- Technical member to handle the various steps ---*/
    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

    /*--- Now we need an instance of the class implemented before with the weak form ---*/
    NavierStokesProjectionOperator<dim, EquationData::degree_p, EquationData::degree_p + 1,
                                   EquationData::degree_p + 1, EquationData::degree_p + 2,
                                   LinearAlgebra::distributed::Vector<double>> navier_stokes_matrix;

    /*--- This is an instance for geometric multigrid preconditioner ---*/
    MGLevelObject<NavierStokesProjectionOperator<dim, EquationData::degree_p, EquationData::degree_p + 1,
                                                 EquationData::degree_p + 1, EquationData::degree_p + 2,
                                                 LinearAlgebra::distributed::Vector<float>>> mg_matrices;

    /*--- Here we define two 'AffineConstraints' instance, one for each finite element space.
          This is just a technical issue, due to MatrixFree requirements. In general
          this class is used to impose boundary conditions (or any kind of constraints), but in this case, since
          we are using a weak imposition of bcs, everything is already in the weak forms and so these instances
          will be default constructed ---*/
    AffineConstraints<double> constraints_velocity,
                              constraints_pressure;

    /*--- Now a bunch of variables handled by 'ParamHandler' introduced at the beginning of the code ---*/
    unsigned int max_its;
    double       eps;

    unsigned int max_loc_refinements;
    unsigned int min_loc_refinements;
    unsigned int refinement_iterations;

    std::string saving_dir;

    /*--- Finally, some output related streams ---*/
    ConditionalOStream pcout;

    std::ofstream      time_out;
    ConditionalOStream ptime_out;
    TimerOutput        time_table;

    std::ofstream output_n_dofs_velocity;
    std::ofstream output_n_dofs_pressure;

    std::ofstream output_lift;
    std::ofstream output_drag;
  };


  // In the constructor, we just read all the data from the
  // <code>Data_Storage</code> object that is passed as an argument, verify that
  // the data we read are reasonable and, finally, create the triangulation and
  // load the initial data.
  //
  template<int dim>
  NavierStokesProjection<dim>::NavierStokesProjection(RunTimeParameters::Data_Storage& data):
    t_0(data.initial_time),
    T(data.final_time),
    gamma(2.0 - std::sqrt(2.0)),  //--- Save also in the NavierStokes class the TR-BDF2 parameter value
    TR_BDF2_stage(1),             //--- Initialize the flag for the TR_BDF2 stage
    Re(data.Reynolds),
    dt(data.dt),
    vel_init(data.initial_time),
    pres_init(data.initial_time),
    triangulation(MPI_COMM_WORLD, parallel::distributed::Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    fe_velocity(FE_DGQ<dim>(EquationData::degree_p + 1), dim),
    fe_pressure(FE_DGQ<dim>(EquationData::degree_p), 1),
    dof_handler_velocity(triangulation),
    dof_handler_pressure(triangulation),
    quadrature_pressure(EquationData::degree_p + 1),
    quadrature_velocity(EquationData::degree_p + 2),
    navier_stokes_matrix(data),
    max_its(data.max_iterations),
    eps(data.eps),
    max_loc_refinements(data.max_loc_refinements),
    min_loc_refinements(data.min_loc_refinements),
    refinement_iterations(data.refinement_iterations),
    saving_dir(data.dir),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_out("./" + data.dir + "/time_analysis_" +
             Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    output_n_dofs_velocity("./" + data.dir + "/n_dofs_velocity.dat", std::ofstream::out),
    output_n_dofs_pressure("./" + data.dir + "/n_dofs_pressure.dat", std::ofstream::out),
    output_lift("./" + data.dir + "/lift.dat", std::ofstream::out),
    output_drag("./" + data.dir + "/drag.dat", std::ofstream::out) {
      if(EquationData::degree_p < 1) {
        pcout
        << " WARNING: The chosen pair of finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;
      }

      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

      create_triangulation(data.n_refines);
      setup_dofs();
      initialize();
  }


  // The method that creates the triangulation and refines it the needed number
  // of times.
  //
  template<int dim>
  void NavierStokesProjection<dim>::create_triangulation(const unsigned int n_refines) {
    TimerOutput::Scope t(time_table, "Create triangulation");

    GridGenerator::plate_with_a_hole(triangulation, 0.5, 1.0, 1.0, 1.1, 1.0, 19.0, Point<2>(2.0, 2.0), 0, 1, 1.0, 2, true);
    /*--- We strongly advice to check the documentation to verify the meaning of all input parameters. ---*/

    pcout << "Number of refines = " << n_refines << std::endl;
    triangulation.refine_global(n_refines);
  }


  // After creating the triangulation, it creates the mesh dependent
  // data, i.e. it distributes degrees of freedom, and
  // initializes the vectors that we will use.
  //
  template<int dim>
  void NavierStokesProjection<dim>::setup_dofs() {
    pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
    pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

    /*--- Distribute dofs and prepare for multigrid ---*/
    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_pressure.distribute_dofs(fe_pressure);

    pcout << "dim (X_h) = " << dof_handler_velocity.n_dofs()
          << std::endl
          << "dim (M_h) = " << dof_handler_pressure.n_dofs()
          << std::endl
          << "Re        = " << Re << std::endl
          << std::endl;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_n_dofs_velocity << dof_handler_velocity.n_dofs() << std::endl;
      output_n_dofs_pressure << dof_handler_pressure.n_dofs() << std::endl;
    }

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags                = (update_gradients | update_JxW_values |
                                                           update_quadrature_points | update_values);
    additional_data.mapping_update_flags_inner_faces    = (update_gradients | update_JxW_values | update_quadrature_points |
                                                           update_normal_vectors | update_values);
    additional_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                           update_normal_vectors | update_values);
    additional_data.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;

    std::vector<const DoFHandler<dim>*> dof_handlers; /*--- Vector of dof_handlers to feed the 'MatrixFree'. Here the order
                                                            counts and enters into the game as parameter of FEEvaluation and
                                                            FEFaceEvaluation in the previous class ---*/
    dof_handlers.push_back(&dof_handler_velocity);
    dof_handlers.push_back(&dof_handler_pressure);

    constraints_velocity.clear();
    constraints_velocity.close();
    constraints_pressure.clear();
    constraints_pressure.close();
    std::vector<const AffineConstraints<double>*> constraints;
    constraints.push_back(&constraints_velocity);
    constraints.push_back(&constraints_pressure);

    std::vector<QGauss<1>> quadratures; /*--- We cannot directly use 'quadrature_velocity' and 'quadrature_pressure',
                                              because the 'MatrixFree' structure wants a quadrature formula for 1D
                                              (this is way the template parameter of the previous class was called 'n_q_points_1d_p'
                                               and 'n_q_points_1d_v' and the reason of '1' as QGauss template parameter). ---*/
    quadratures.push_back(QGauss<1>(EquationData::degree_p + 2));
    quadratures.push_back(QGauss<1>(EquationData::degree_p + 1));

    /*--- Initialize the matrix-free structure and size properly the vectors. Here again the
          second input argument of the 'initialize_dof_vector' method depends on the order of 'dof_handlers' ---*/
    matrix_free_storage->reinit(MappingQ1<dim>(),dof_handlers, constraints, quadratures, additional_data);
    matrix_free_storage->initialize_dof_vector(u_star, 0);
    matrix_free_storage->initialize_dof_vector(rhs_u, 0);
    matrix_free_storage->initialize_dof_vector(u_n, 0);
    matrix_free_storage->initialize_dof_vector(u_extr, 0);
    matrix_free_storage->initialize_dof_vector(u_n_minus_1, 0);
    matrix_free_storage->initialize_dof_vector(u_n_gamma, 0);
    matrix_free_storage->initialize_dof_vector(u_tmp, 0);
    matrix_free_storage->initialize_dof_vector(grad_pres_int, 0);

    matrix_free_storage->initialize_dof_vector(pres_int, 1);
    matrix_free_storage->initialize_dof_vector(pres_n, 1);
    matrix_free_storage->initialize_dof_vector(rhs_p, 1);

    /*--- Initialize the multigrid structure. We dedicate ad hoc 'dof_handlers_mg' and 'constraints_mg' because
          we use float as type. Moreover we can initialize already with the index of the finite element of the pressure;
          anyway we need by requirement to declare also structures for the velocity for coherence (basically because
          the index of finite element space has to be the same, so the pressure has to be the second).---*/
    mg_matrices.clear_elements();
    dof_handler_velocity.distribute_mg_dofs();
    dof_handler_pressure.distribute_mg_dofs();

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);
    for(unsigned int level = 0; level < nlevels; ++level) {
      typename MatrixFree<dim, float>::AdditionalData additional_data_mg;
      additional_data_mg.tasks_parallel_scheme               = MatrixFree<dim, float>::AdditionalData::none;
      additional_data_mg.mapping_update_flags                = (update_gradients | update_JxW_values);
      additional_data_mg.mapping_update_flags_inner_faces    = (update_gradients | update_JxW_values);
      additional_data_mg.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values);
      additional_data_mg.mg_level = level;

      std::vector<const DoFHandler<dim>*> dof_handlers_mg;
      dof_handlers_mg.push_back(&dof_handler_velocity);
      dof_handlers_mg.push_back(&dof_handler_pressure);
      std::vector<const AffineConstraints<float>*> constraints_mg;
      AffineConstraints<float> constraints_velocity_mg;
      constraints_velocity_mg.clear();
      constraints_velocity_mg.close();
      constraints_mg.push_back(&constraints_velocity_mg);
      AffineConstraints<float> constraints_pressure_mg;
      constraints_pressure_mg.clear();
      constraints_pressure_mg.close();
      constraints_mg.push_back(&constraints_pressure_mg);

      std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
      mg_mf_storage_level->reinit(MappingQ1<dim>(),dof_handlers_mg, constraints_mg, quadratures, additional_data_mg);
      const std::vector<unsigned int> tmp = {1};
      mg_matrices[level].initialize(mg_mf_storage_level, tmp, tmp);
      mg_matrices[level].set_dt(dt);
      mg_matrices[level].set_NS_stage(2);
    }

    Linfty_error_per_cell_vel.reinit(triangulation.n_active_cells());
  }


  // This method loads the initial data. It simply uses the class <code>Pressure</code> instance for the pressure
  // and the class <code>Velocity</code> instance for the velocity.
  //
  template<int dim>
  void NavierStokesProjection<dim>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize pressure and velocity");

    VectorTools::interpolate(dof_handler_pressure, pres_init, pres_n);

    VectorTools::interpolate(dof_handler_velocity, vel_init, u_n_minus_1);
    VectorTools::interpolate(dof_handler_velocity, vel_init, u_n);
  }


  // This function computes the extrapolated velocity to be used in the momentum predictor
  //
  template<int dim>
  void NavierStokesProjection<dim>::interpolate_velocity() {
    TimerOutput::Scope t(time_table, "Interpolate velocity");

    //--- TR-BDF2 first step
    if(TR_BDF2_stage == 1) {
      u_extr.equ(1.0 + gamma/(2.0*(1.0 - gamma)), u_n);
      u_tmp.equ(gamma/(2.0*(1.0 - gamma)), u_n_minus_1);
      u_extr -= u_tmp;
    }
    //--- TR-BDF2 second step
    else {
      u_extr.equ(1.0 + (1.0 - gamma)/gamma, u_n_gamma);
      u_tmp.equ((1.0 - gamma)/gamma, u_n);
      u_extr -= u_tmp;
    }
  }


  // We are finally ready to solve the diffusion step.
  //
  template<int dim>
  void NavierStokesProjection<dim>::diffusion_step() {
    TimerOutput::Scope t(time_table, "Diffusion step");

    /*--- We first speicify that we want to deal with velocity dof_handler (index 0, since it is the first one
          in the 'dof_handlers' vector) ---*/
    const std::vector<unsigned int> tmp = {0};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

    /*--- Next, we specify at we are at stage 1, namely the diffusion step ---*/
    navier_stokes_matrix.set_NS_stage(1);

    /*--- Now, we compute the right-hand side and we set the convective velocity. The necessity of 'set_u_extr' is
          that this quantity is required in the bilinear forms and we can't use a vector of src like on the right-hand side,
          so it has to be available ---*/
    if(TR_BDF2_stage == 1) {
      navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_extr, pres_n});
      navier_stokes_matrix.set_u_extr(u_extr);
      u_star = u_extr;
    }
    else {
      navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_n_gamma, pres_int, u_extr});
      navier_stokes_matrix.set_u_extr(u_extr);
      u_star = u_extr;
    }

    /*--- Build the linear solver; in this case we specifiy the maximum number of iterations and residual ---*/
    SolverControl solver_control(max_its, eps*rhs_u.l2_norm());
    SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);

    /*--- Build a Jacobi preconditioner and solve ---*/
    PreconditionJacobi<NavierStokesProjectionOperator<dim,
                                                      EquationData::degree_p,
                                                      EquationData::degree_p + 1,
                                                      EquationData::degree_p + 1,
                                                      EquationData::degree_p + 2,
                                                      LinearAlgebra::distributed::Vector<double>>> preconditioner;
    navier_stokes_matrix.compute_diagonal();
    preconditioner.initialize(navier_stokes_matrix);

    gmres.solve(navier_stokes_matrix, u_star, rhs_u, preconditioner);
  }


  // Next, we solve the projection step.
  //
  template<int dim>
  void NavierStokesProjection<dim>::projection_step() {
    TimerOutput::Scope t(time_table, "Projection step pressure");

    /*--- We start in the same way of 'diffusion_step': we first reinitialize with the index of FE space,
          we specify that this is the second stage and we compute the right-hand side ---*/
    const std::vector<unsigned int> tmp = {1};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

    navier_stokes_matrix.set_NS_stage(2);

    if(TR_BDF2_stage == 1)
      navier_stokes_matrix.vmult_rhs_pressure(rhs_p, {u_star, pres_n});
    else
      navier_stokes_matrix.vmult_rhs_pressure(rhs_p, {u_star, pres_int});

    /*--- Build the linear solver (Conjugate Gradient in this case) ---*/
    SolverControl solver_control(max_its, eps*rhs_p.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    /*--- Build the preconditioner (as in step-37) ---*/
    MGTransferMatrixFree<dim, float> mg_transfer;
    mg_transfer.build(dof_handler_pressure);

    using SmootherType = PreconditionChebyshev<NavierStokesProjectionOperator<dim,
                                                                              EquationData::degree_p,
                                                                              EquationData::degree_p + 1,
                                                                              EquationData::degree_p + 1,
                                                                              EquationData::degree_p + 2,
                                                                              LinearAlgebra::distributed::Vector<float>>,
                                               LinearAlgebra::distributed::Vector<float>>;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      if(level > 0) {
        smoother_data[level].smoothing_range     = 15.0;
        smoother_data[level].degree              = 3;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else {
        smoother_data[0].smoothing_range     = 2e-2;
        smoother_data[0].degree              = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
      }
      mg_matrices[level].compute_diagonal();
      smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    PreconditionIdentity                                identity;
    SolverCG<LinearAlgebra::distributed::Vector<float>> cg_mg(solver_control);
    MGCoarseGridIterativeSolver<LinearAlgebra::distributed::Vector<float>,
                                SolverCG<LinearAlgebra::distributed::Vector<float>>,
                                NavierStokesProjectionOperator<dim,
                                                               EquationData::degree_p,
                                                               EquationData::degree_p + 1,
                                                               EquationData::degree_p + 1,
                                                               EquationData::degree_p + 2,
                                                               LinearAlgebra::distributed::Vector<float>>,
                                PreconditionIdentity> mg_coarse(cg_mg, mg_matrices[0], identity);

    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);

    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<float>,
                   MGTransferMatrixFree<dim, float>> preconditioner(dof_handler_pressure, mg, mg_transfer);

    /*--- Solve the linear system ---*/
    if(TR_BDF2_stage == 1) {
      pres_int = pres_n;
      cg.solve(navier_stokes_matrix, pres_int, rhs_p, preconditioner);
    }
    else {
      pres_n = pres_int;
      cg.solve(navier_stokes_matrix, pres_n, rhs_p, preconditioner);
    }
  }


  // This implements the projection step for the gradient of pressure
  //
  template<int dim>
  void NavierStokesProjection<dim>::project_grad(const unsigned int flag) {
    TimerOutput::Scope t(time_table, "Gradient of pressure projection");

    /*--- The input parameter flag is used just to specify where we want to save the result ---*/
    AssertIndexRange(flag, 3);
    Assert(flag > 0, ExcInternalError());

    /*--- We need to select the dof handler related to the velocity since the result lives there ---*/
    const std::vector<unsigned int> tmp = {0};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

    if(flag == 1)
      navier_stokes_matrix.vmult_grad_p_projection(rhs_u, pres_n);
    else if(flag == 2)
      navier_stokes_matrix.vmult_grad_p_projection(rhs_u, pres_int);

    /*--- We conventionally decide that the this corresponds to third stage ---*/
    navier_stokes_matrix.set_NS_stage(3);

    /*--- Solve the system ---*/
    SolverControl solver_control(max_its, 1e-12*rhs_u.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
    cg.solve(navier_stokes_matrix, u_tmp, rhs_u, PreconditionIdentity());
  }


  // The following function is used in determining the maximal velocity
  // in order to compute the Courant number.
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_maximal_velocity() {
    return u_n.linfty_norm();
  }


  // The following function is used in determining the maximal nodal difference
  // between old and current velocity value in order to see if we have reched steady-state.
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_maximal_difference_velocity() {
    u_tmp = u_n;
    u_tmp -= u_n_minus_1;

    return u_tmp.linfty_norm();
  }


  // This method plots the current solution. The main difficulty is that we want
  // to create a single output file that contains the data for all velocity
  // components and the pressure. On the other hand, velocities and the pressure
  // live on separate DoFHandler objects, so we need to pay attention when we use
  // 'add_data_vector' to select the proper space.
  //
  template<int dim>
  void NavierStokesProjection<dim>::output_results(const unsigned int step) {
    TimerOutput::Scope t(time_table, "Output results");

    DataOut<dim> data_out;

    std::vector<std::string> velocity_names(dim, "v");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
    u_n.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity, u_n, velocity_names, component_interpretation_velocity);
    pres_n.update_ghost_values();
    data_out.add_data_vector(dof_handler_pressure, pres_n, "p", {DataComponentInterpretation::component_is_scalar});

    std::vector<std::string> velocity_names_old(dim, "v_old");
    u_n_minus_1.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity, u_n_minus_1, velocity_names_old, component_interpretation_velocity);

    /*--- Here we rely on the postprocessor we have built ---*/
    PostprocessorVorticity<dim> postprocessor;
    data_out.add_data_vector(dof_handler_velocity, u_n, postprocessor);

    data_out.build_patches(MappingQ1<dim>(), 1, DataOut<dim>::curved_inner_cells);

    const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
  }


  // @sect{<code>NavierStokesProjection::compute_lift_and_drag</code>}

  // This routine computes the lift and the drag forces in a non-dimensional framework
  // (so basically for the classical coefficients, it is necessary to multiply by a factor 2).
  //
  template<int dim>
  void NavierStokesProjection<dim>::compute_lift_and_drag() {
    QGauss<dim - 1> face_quadrature_formula(EquationData::degree_p + 2);
    const int n_q_points = face_quadrature_formula.size();

    std::vector<double>                      pressure_values(n_q_points);
    std::vector<std::vector<Tensor<1, dim>>> velocity_gradients(n_q_points, std::vector<Tensor<1, dim>>(dim));

    Tensor<1, dim> normal_vector;
    Tensor<2, dim> fluid_stress;
    Tensor<2, dim> fluid_pressure;
    Tensor<1, dim> forces;

    /*--- We need to compute the integral over the cylinder boundary, so we need to use 'FEFaceValues' instances.
          For the velocity we need the gradients, for the pressure the values. ---*/
    FEFaceValues<dim> fe_face_values_velocity(fe_velocity, face_quadrature_formula,
                                              update_quadrature_points | update_gradients |
                                              update_JxW_values | update_normal_vectors);
    FEFaceValues<dim> fe_face_values_pressure(fe_pressure, face_quadrature_formula, update_values);

    double local_drag = 0.0;
    double local_lift = 0.0;

    /*--- We need to perform a unique loop because the whole stress tensor takes into account contributions of
          velocity and pressure obviously. However, the two dof_handlers are different, so we neede to create an ad-hoc
          iterator for the pressure that we update manually. It is guaranteed that the cells are visited in the same order
          (see the documentation) ---*/
    auto tmp_cell = dof_handler_pressure.begin_active();
    for(const auto& cell : dof_handler_velocity.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
          if(cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == 4) {
            fe_face_values_velocity.reinit(cell, face);
            fe_face_values_pressure.reinit(tmp_cell, face);

            fe_face_values_velocity.get_function_gradients(u_n, velocity_gradients); /*--- velocity gradients ---*/
            fe_face_values_pressure.get_function_values(pres_n, pressure_values); /*--- pressure values ---*/

            for(int q = 0; q < n_q_points; q++) {
              normal_vector = -fe_face_values_velocity.normal_vector(q);

              for(unsigned int d = 0; d < dim; ++ d) {
                fluid_pressure[d][d] = pressure_values[q];
                for(unsigned int k = 0; k < dim; ++k)
                  fluid_stress[d][k] = 1.0/Re*velocity_gradients[q][d][k];
              }
              fluid_stress = fluid_stress - fluid_pressure;

              forces = fluid_stress*normal_vector*fe_face_values_velocity.JxW(q);

              local_drag += forces[0];
              local_lift += forces[1];
            }
          }
        }
      }
      ++tmp_cell;
    }

    /*--- At the end, each processor has computed the contribution to the boundary cells it owns and, therefore,
          we need to sum up all the contributions. ---*/
    const double lift = Utilities::MPI::sum(local_lift, MPI_COMM_WORLD);
    const double drag = Utilities::MPI::sum(local_drag, MPI_COMM_WORLD);
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_lift << lift << std::endl;
      output_drag << drag << std::endl;
    }
  }


  // @sect{ <code>NavierStokesProjection::refine_mesh</code>}

  // After finding a good initial guess on the coarse mesh, we hope to
  // decrease the error through refining the mesh. We also need to transfer the current solution to the
  // next mesh using the SolutionTransfer class.
  //
  template <int dim>
  void NavierStokesProjection<dim>::refine_mesh() {
    TimerOutput::Scope t(time_table, "Refine mesh");

    /*--- We first create a proper vector for computing estimator ---*/
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler_velocity, locally_relevant_dofs);
    LinearAlgebra::distributed::Vector<double> tmp_velocity;
    tmp_velocity.reinit(dof_handler_velocity.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
    tmp_velocity = u_n;
    tmp_velocity.update_ghost_values();

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    /*--- This is basically the indicator per cell computation (see step-50). Since it is not so complciated
          we implement it through a lambda expression ---*/
    const auto cell_worker = [&](const Iterator&   cell,
                                 ScratchData<dim>& scratch_data,
                                 CopyData&         copy_data) {
      FEValues<dim>& fe_values = scratch_data.fe_values; /*--- Here we finally use the 'FEValues' inside ScratchData ---*/
      fe_values.reinit(cell);

      /*--- Compute the gradients for all quadrature points ---*/
      std::vector<std::vector<Tensor<1, dim>>> gradients(fe_values.n_quadrature_points, std::vector<Tensor<1, dim>>(dim));
      fe_values.get_function_gradients(tmp_velocity, gradients);
      copy_data.cell_index = cell->active_cell_index();
      double vorticity_norm_square = 0.0;
      /*--- Loop over quadrature points and evaluate the integral multiplying the vorticty
            by the weights and the determinant of the Jacobian (which are included in 'JxW') ---*/
      for(unsigned k = 0; k < fe_values.n_quadrature_points; ++k) {
        const double vorticity = gradients[k][1][0] - gradients[k][0][1];
        vorticity_norm_square += vorticity*vorticity*fe_values.JxW(k);
      }
      copy_data.value = cell->diameter()*cell->diameter()*vorticity_norm_square;
    };

    const UpdateFlags cell_flags = update_gradients | update_quadrature_points | update_JxW_values;

    const auto copier = [&](const CopyData &copy_data) {
      if(copy_data.cell_index != numbers::invalid_unsigned_int)
        estimated_error_per_cell[copy_data.cell_index] += copy_data.value;
    };

    /*--- Now everything is 'automagically' handled by 'mesh_loop' ---*/
    ScratchData<dim> scratch_data(fe_velocity, EquationData::degree_p + 2, cell_flags);
    CopyData copy_data;
    MeshWorker::mesh_loop(dof_handler_velocity.begin_active(),
                          dof_handler_velocity.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);

    /*--- Refine grid. In case the refinement level is above a certain value (or the coarsening level is below)
          we clear the flags. ---*/
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.01, 0.3);
    for(const auto& cell: triangulation.active_cell_iterators()) {
      if(cell->refine_flag_set() && static_cast<unsigned int>(cell->level()) == max_loc_refinements)
        cell->clear_refine_flag();
      if(cell->coarsen_flag_set() && static_cast<unsigned int>(cell->level()) == min_loc_refinements)
        cell->clear_coarsen_flag();
    }
    triangulation.prepare_coarsening_and_refinement();

    /*--- Now we prepare the object for transfering, basically saving the old quantities using SolutionTransfer.
          Since the 'prepare_for_coarsening_and_refinement' method can be called only once, but we have two vectors
          for dof_handler_velocity, we need to put them in an auxiliary vector. ---*/
    std::vector<const LinearAlgebra::distributed::Vector<double>*> velocities;
    velocities.push_back(&u_n);
    velocities.push_back(&u_n_minus_1);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_velocity(dof_handler_velocity);
    solution_transfer_velocity.prepare_for_coarsening_and_refinement(velocities);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_pressure(dof_handler_pressure);
    solution_transfer_pressure.prepare_for_coarsening_and_refinement(pres_n);

    triangulation.execute_coarsening_and_refinement(); /*--- Effectively perform the remeshing ---*/

    /*--- First DoFHandler objects are set up within the new grid ----*/
    setup_dofs();

    /*--- Interpolate current solutions to new mesh. This is done using auxliary vectors just for safety,
          but the new u_n or pres_n could be used. Again, the only point is that the function 'interpolate'
          can be called once and so the vectors related to 'dof_handler_velocity' have to collected in an auxiliary vector. ---*/
    LinearAlgebra::distributed::Vector<double> transfer_velocity,
                                               transfer_velocity_minus_1,
                                               transfer_pressure;
    transfer_velocity.reinit(u_n);
    transfer_velocity.zero_out_ghost_values();
    transfer_velocity_minus_1.reinit(u_n_minus_1);
    transfer_velocity_minus_1.zero_out_ghost_values();
    transfer_pressure.reinit(pres_n);
    transfer_pressure.zero_out_ghost_values();

    std::vector<LinearAlgebra::distributed::Vector<double>*> transfer_velocities;
    transfer_velocities.push_back(&transfer_velocity);
    transfer_velocities.push_back(&transfer_velocity_minus_1);
    solution_transfer_velocity.interpolate(transfer_velocities);
    transfer_velocity.update_ghost_values();
    transfer_velocity_minus_1.update_ghost_values();
    solution_transfer_pressure.interpolate(transfer_pressure);
    transfer_pressure.update_ghost_values();

    u_n         = transfer_velocity;
    u_n_minus_1 = transfer_velocity_minus_1;
    pres_n      = transfer_pressure;
  }


  // Interpolate the locally refined solution to a mesh with maximal resolution
  // and transfer velocity and pressure.
  //
  template<int dim>
  void NavierStokesProjection<dim>::interpolate_max_res(const unsigned int level) {
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_velocity(dof_handler_velocity);
    std::vector<const LinearAlgebra::distributed::Vector<double>*> velocities;
    velocities.push_back(&u_n);
    velocities.push_back(&u_n_minus_1);
    solution_transfer_velocity.prepare_for_coarsening_and_refinement(velocities);

    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_pressure(dof_handler_pressure);
    solution_transfer_pressure.prepare_for_coarsening_and_refinement(pres_n);

    for(const auto& cell: triangulation.active_cell_iterators_on_level(level)) {
      if(cell->is_locally_owned())
        cell->set_refine_flag();
    }
    triangulation.execute_coarsening_and_refinement();

    setup_dofs();

    LinearAlgebra::distributed::Vector<double> transfer_velocity, transfer_velocity_minus_1,
                                               transfer_pressure;

    transfer_velocity.reinit(u_n);
    transfer_velocity.zero_out_ghost_values();
    transfer_velocity_minus_1.reinit(u_n_minus_1);
    transfer_velocity_minus_1.zero_out_ghost_values();

    transfer_pressure.reinit(pres_n);
    transfer_pressure.zero_out_ghost_values();

    std::vector<LinearAlgebra::distributed::Vector<double>*> transfer_velocities;

    transfer_velocities.push_back(&transfer_velocity);
    transfer_velocities.push_back(&transfer_velocity_minus_1);
    solution_transfer_velocity.interpolate(transfer_velocities);
    transfer_velocity.update_ghost_values();
    transfer_velocity_minus_1.update_ghost_values();

    solution_transfer_pressure.interpolate(transfer_pressure);
    transfer_pressure.update_ghost_values();

    u_n         = transfer_velocity;
    u_n_minus_1 = transfer_velocity_minus_1;
    pres_n      = transfer_pressure;
  }


  // Save maximum resolution to a mesh adapted.
  //
  template<int dim>
  void NavierStokesProjection<dim>::save_max_res() {
    parallel::distributed::Triangulation<dim> triangulation_tmp(MPI_COMM_WORLD);
    GridGenerator::plate_with_a_hole(triangulation_tmp, 0.5, 1.0, 1.0, 1.1, 1.0, 19.0, Point<2>(2.0, 2.0), 0, 1, 1.0, 2, true);
    triangulation_tmp.refine_global(triangulation.n_global_levels() - 1);

    DoFHandler<dim> dof_handler_velocity_tmp(triangulation_tmp);
    DoFHandler<dim> dof_handler_pressure_tmp(triangulation_tmp);
    dof_handler_velocity_tmp.distribute_dofs(fe_velocity);
    dof_handler_pressure_tmp.distribute_dofs(fe_pressure);

    LinearAlgebra::distributed::Vector<double> u_n_tmp,
                                               pres_n_tmp;
    u_n_tmp.reinit(dof_handler_velocity_tmp.n_dofs());
    pres_n_tmp.reinit(dof_handler_pressure_tmp.n_dofs());

    DataOut<dim> data_out;
    std::vector<std::string> velocity_names(dim, "v");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
    VectorTools::interpolate_to_different_mesh(dof_handler_velocity, u_n, dof_handler_velocity_tmp, u_n_tmp);
    u_n_tmp.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity_tmp, u_n_tmp, velocity_names, component_interpretation_velocity);
    VectorTools::interpolate_to_different_mesh(dof_handler_pressure, pres_n, dof_handler_pressure_tmp, pres_n_tmp);
    pres_n_tmp.update_ghost_values();
    data_out.add_data_vector(dof_handler_pressure_tmp, pres_n_tmp, "p", {DataComponentInterpretation::component_is_scalar});
    PostprocessorVorticity<dim> postprocessor;
    data_out.add_data_vector(dof_handler_velocity_tmp, u_n_tmp, postprocessor);

    data_out.build_patches(MappingQ1<dim>(), 1, DataOut<dim>::curved_inner_cells);
    const std::string output = "./" + saving_dir + "/solution_max_res_end.vtu";
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
  }


  // @sect{ <code>NavierStokesProjection::run</code> }

  // This is the time marching function, which starting at <code>t_0</code>
  // advances in time using the projection method with time step <code>dt</code>
  // until <code>T</code>.
  //
  // Its second parameter, <code>verbose</code> indicates whether the function
  // should output information what it is doing at any given moment:
  // we use the ConditionalOStream class to do that for us.
  //
  template<int dim>
  void NavierStokesProjection<dim>::run(const bool verbose, const unsigned int output_interval) {
    ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    output_results(1);
    double time = t_0 + dt;
    unsigned int n = 1;
    while(std::abs(T - time) > 1e-10) {
      time += dt;
      n++;
      pcout << "Step = " << n << " Time = " << time << std::endl;

      /*--- First stage of TR-BDF2 and we start by setting the proper flag ---*/
      TR_BDF2_stage = 1;
      navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);
      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_matrices[level].set_TR_BDF2_stage(TR_BDF2_stage);

      verbose_cout << "  Interpolating the velocity stage 1" << std::endl;
      interpolate_velocity();

      verbose_cout << "  Diffusion Step stage 1 " << std::endl;
      diffusion_step();

      verbose_cout << "  Projection Step stage 1" << std::endl;
      project_grad(1);
      u_tmp.equ(gamma*dt, u_tmp);
      u_star += u_tmp; /*--- In the rhs of the projection step we need u_star + gamma*dt*grad(pres_n) and we save it into u_star ---*/
      projection_step();

      verbose_cout << "  Updating the Velocity stage 1" << std::endl;
      u_n_gamma.equ(1.0, u_star);
      project_grad(2);
      grad_pres_int.equ(1.0, u_tmp); /*--- We save grad(pres_int), because we will need it soon ---*/
      u_tmp.equ(-gamma*dt, u_tmp);
      u_n_gamma += u_tmp; /*--- u_n_gamma = u_star - gamma*dt*grad(pres_int) ---*/
      u_n_minus_1 = u_n;

      /*--- Second stage of TR-BDF2 ---*/
      TR_BDF2_stage = 2;
      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_matrices[level].set_TR_BDF2_stage(TR_BDF2_stage);
      navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);

      verbose_cout << "  Interpolating the velocity stage 2" << std::endl;
      interpolate_velocity();

      verbose_cout << "  Diffusion Step stage 2 " << std::endl;
      diffusion_step();

      verbose_cout << "  Projection Step stage 2" << std::endl;
      u_tmp.equ((1.0 - gamma)*dt, grad_pres_int);
      u_star += u_tmp;  /*--- In the rhs of the projection step we need u_star + (1 - gamma)*dt*grad(pres_int) ---*/
      projection_step();

      verbose_cout << "  Updating the Velocity stage 2" << std::endl;
      u_n.equ(1.0, u_star);
      project_grad(1);
      u_tmp.equ((gamma - 1.0)*dt, u_tmp);
      u_n += u_tmp;  /*--- u_n = u_star - (1 - gamma)*dt*grad(pres_n) ---*/

      const double max_vel = get_maximal_velocity();
      pcout<< "Maximal velocity = " << max_vel << std::endl;
      /*--- The Courant number is computed taking into account the polynomial degree for the velocity ---*/
      pcout << "CFL = " << dt*max_vel*(EquationData::degree_p + 1)*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;
      compute_lift_and_drag();
      if(n % output_interval == 0) {
        verbose_cout << "Plotting Solution final" << std::endl;
        output_results(n);
      }
      /*--- In case dt is not a multiple of T, we reduce dt in order to end up at T ---*/
      if(T - time < dt && T - time > 1e-10) {
        dt = T - time;
        navier_stokes_matrix.set_dt(dt);
        for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
          mg_matrices[level].set_dt(dt);
      }
      /*--- Perform the refinement if desired ---*/
      if(refinement_iterations > 0 && n % refinement_iterations == 0) {
        verbose_cout << "Refining mesh" << std::endl;
        refine_mesh();
      }
    }
    if(n % output_interval != 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
    }
    if(refinement_iterations > 0) {
      for(unsigned int lev = 0; lev < triangulation.n_global_levels() - 1; ++ lev)
        interpolate_max_res(lev);
      save_max_res();
    }
  }

} // namespace NS_TRBDF2


// @sect{ The main function }

// The main function looks very much like in all the other tutorial programs. We first initialize MPI,
// we initialize the class 'NavierStokesProjection' with the dimension as template parameter and then
// let the method 'run' do the job.
//
int main(int argc, char *argv[]) {
  try {
    using namespace NS_TRBDF2;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    NavierStokesProjection<2> test(data);
    test.run(data.verbose, data.output_interval);

    if(curr_rank == 0)
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Apparently everything went fine!" << std::endl
                << "Don't forget to brush your teeth :-)" << std::endl
                << std::endl;

    return 0;
  }
  catch(std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

}
