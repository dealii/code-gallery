/**
 * @brief Implementation of the PDE-based density filter used by the
 *        matrix-free heat-conduction topology optimization solver.
 *
 * This translation unit implements the filter-related member functions of
 * Elasticity_Solver:
 *
 *  - setup_filter_system()          : assemble the fine-level Helmholtz
 *                                     operator used as a PDE density filter.
 *  - setup_multigrid_filter_system(): build the geometric multigrid
 *                                     hierarchy for the filter operator.
 *  - assemble_filter_rhs()          : assemble the right-hand side of the
 *                                     Helmholtz filter from a DGQ(0) density.
 *  - solve_filter_chebyshev()       : solve the filter equation with a
 *                                     Chebyshev-preconditioned Krylov solver
 *                                     on top of multigrid.
 *  - generate_average_vector()      : project the filtered (continuous)
 *                                     density back onto the DGQ(0) space.
 *
 * The Helmholtz filter corresponds to solving
 * @f[
 *   \bigl(1 - R_{min}^2 \Delta\bigr)\, \tilde{\rho} = \rho,
 * @f]
 * where @f$ \rho @f$ is the unfiltered density and @f$ \tilde{\rho} @f$ is
 * the filtered density. The filter radius @f$ R_{min} @f$ is a member of
 * Elasticity_Solver.
 * 
 * for density feild && toppology optimization sensivity, share the same
 * pde matrix, only change rhs vector, mesh change will change the pde matrix
 */

#include "../include/Elasticity_Matrix_Free.h"

using namespace Elasticity_MF;

/**
 * @brief Set up the fine-level Helmholtz PDE filter operator.
 *
 * This routine
 *  - distributes the DoFs of the continuous filter space (fe_filter),
 *  - builds the hanging-node constraints,
 *  - initializes the matrix-free Helmholtz operator on the fine grid,
 *  - sets the filter radius @p Rmin_Pow2 and computes the inverse diagonal
 *    (used later by the Chebyshev smoother / Jacobi preconditioner),
 *  - allocates the filter right-hand side, solution and filtered-density
 *    vectors.
 *
 * @note The Helmholtz operator built here is **shared** between the
 *       density filter and the sensitivity filter (see the class
 *       documentation of Helmholtz_Matrix_Free). This setup only needs to
 *       be called when the mesh changes; within one optimization loop
 *       the matrix is reused and only the RHS vector is rebuilt
 *       (assemble_filter_rhs). This makes the filter step essentially
 *       "one operator, many right-hand sides".
 */
template<unsigned int dim, unsigned int degree>
void Elasticity_Solver<dim, degree>::setup_filter_system()
{
    TimerOutput::Scope t(computing_timer, "setup_filter_system");

    // Rebuild the fine-level filter operator from scratch.
    filter_matrix.clear();
    dof_handler_filter.distribute_dofs(fe_filter);
    dof_handler_filter.distribute_mg_dofs();

    // Set up hanging-node constraints on the continuous filter space.
    constraints_filter.clear();
    constraints_filter.reinit(dof_handler_filter.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(dof_handler_filter));
    DoFTools::make_hanging_node_constraints(dof_handler_filter,constraints_filter);
    constraints_filter.close();

    // Configure the matrix-free data structure. We need values, gradients,
    // JxW and quadrature points for the Helmholtz operator.
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_gradients |
                                            update_JxW_values |
                                            update_quadrature_points |
                                            update_values);
    std::shared_ptr<MatrixFree<dim, double>> filter_mf_storage(new MatrixFree<dim, double>());
    filter_mf_storage->reinit(mapping_filter,
                              dof_handler_filter,
                              constraints_filter,
                              QGauss<1>(fe_filter.degree + 1),
                              additional_data);
    filter_matrix.initialize(filter_mf_storage);

    // Store Rmin^2 (the coefficient of the Laplacian part of the
    // Helmholtz operator) and build the diagonal preconditioner.
    filter_matrix.Rmin_Pow2 = (double)(Rmin * Rmin);
    filter_matrix.compute_diagonal();

    // Allocate the vectors owned by the filter system.
    filter_matrix.initialize_dof_vector(filter_rhs);
    filter_matrix.initialize_dof_vector(filter_solution);
}

/**
 * @brief Build the geometric multigrid hierarchy for the filter operator.
 *
 * A level-wise matrix-free Helmholtz operator is constructed on every
 * level of the triangulation and stored in @p mg_matrices_filter. The
 * single-precision (float) scalar type is used on all levels to reduce
 * memory traffic inside the multigrid preconditioner; only the fine-level
 * operator (see setup_filter_system) works in double precision.
 */
template<unsigned int dim, unsigned int degree>
void Elasticity_Solver<dim, degree>::setup_multigrid_filter_system()
{
    TimerOutput::Scope t(computing_timer, "setup_multigrid_filter_system");

    // Reset the level operators and the level constraints.
    mg_matrices_filter.clear_elements();
    mg_constrained_dofs_filter.clear();

    const unsigned int nlevels = triangulation.n_levels();
    mg_matrices_filter.resize(0, nlevels - 1);
    mg_constrained_dofs_filter.initialize(dof_handler_filter);

    for (unsigned int level = 0; level < nlevels; level++)
    {
        // Level-local constraints (no hanging nodes across levels here).
        AffineConstraints<float> level_constraints(dof_handler_filter.locally_owned_mg_dofs(level),
                                 DoFTools::extract_locally_relevant_level_dofs(dof_handler_filter, level));;
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients |
                                                update_JxW_values |
                                                update_quadrature_points |
                                                update_values);
        additional_data.mg_level = level;

        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level = std::make_shared<MatrixFree<dim, float>>();
        mg_mf_storage_level->reinit(mapping_filter,
                                    dof_handler_filter,
                                    level_constraints,
                                    QGauss<1>(fe_filter.degree + 1),
                                    additional_data);

        // Level operator with the same Rmin^2 coefficient as on the fine level.
        mg_matrices_filter[level].initialize(mg_mf_storage_level,
                                            mg_constrained_dofs_filter,
                                            level);
        mg_matrices_filter[level].Rmin_Pow2 = static_cast<float>(Rmin * Rmin);
        mg_matrices_filter[level].compute_diagonal();
    }
}

/**
 * @brief Assemble the right-hand side of the Helmholtz filter.
 *
 * For each locally owned continuous filter cell, the cell-local right-hand
 * side is
 * @f[
 *   b_i = \int_{\Omega_K} \phi_i \, \rho \, \mathrm{d}x,
 * @f]
 * where @f$ \rho @f$ is the (piecewise constant) design density taken from
 * @p input, which is a DGQ(0) vector. Hanging-node constraints are
 * distributed to the global vector.
 *
 * @note The operator matrix is unchanged; only the RHS is rebuilt here.
 *       The exact same routine is also used to assemble the RHS of the
 *       sensitivity filter, with @p input being the (DGQ(0)-projected)
 *       objective derivative instead of the density. This is the concrete
 *       realization of the "same PDE matrix, different RHS" idea described
 *       in the class documentation of Helmholtz_Matrix_Free.
 *
 * @param input DGQ(0) density field used as the filter source. For
 *              sensitivity filtering, pass the discrete objective
 *              derivative instead.
 */
template<unsigned int dim, unsigned int degree>
void Elasticity_Solver<dim, degree>::assemble_filter_rhs
    (const LinearAlgebra::distributed::Vector<double> &input)
{
    TimerOutput::Scope t(computing_timer, "assemble_filter_rhs");

    filter_rhs = 0;

    // FEValues for the continuous filter space.
    QGauss<dim> quadrature_formula_filter(fe_filter.degree + 1);
    FEValues<dim> fe_values_filter(fe_filter,
                                         quadrature_formula_filter,
                                         update_values |
                                         update_quadrature_points |
                                         update_JxW_values);

    unsigned int dofs_per_cell = fe_filter.n_dofs_per_cell();
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices_filter(dofs_per_cell);
    std::vector<types::global_dof_index> local_rho_dof_indices_filter(1);
    std::vector<double> rho_values_filter(1);

    // Iterate over the active filter cells and the corresponding DGQ(0)
    // cells in lock-step, so that each filter cell is paired with its
    // own constant density value.
    auto cell_begin_filter = dof_handler_filter.begin_active();
    auto cell_end_filter = dof_handler_filter.end();
    auto base_begin_filter = dof_handler_simp.begin_active();

    for (; cell_begin_filter != cell_end_filter; ++cell_begin_filter, ++base_begin_filter)
    {
        if (cell_begin_filter->is_locally_owned() && cell_begin_filter->level() == base_begin_filter->level())
        {
            cell_rhs = 0;
            fe_values_filter.reinit(cell_begin_filter);

            // Read the (constant) DGQ(0) density on this cell.
            base_begin_filter->get_dof_indices(local_rho_dof_indices_filter);
            rho_values_filter[0] = input[local_rho_dof_indices_filter[0]];

            // Integrate \int \phi_i * rho dV over the cell.
            for (unsigned int q_point : fe_values_filter.quadrature_point_indices())
            {
                for (unsigned int i : fe_values_filter.dof_indices())
                {
                    cell_rhs(i) += (fe_values_filter.shape_value(i, q_point) *
                                    rho_values_filter[0] *
                                    fe_values_filter.JxW(q_point));
                }
            }

            cell_begin_filter->get_dof_indices(local_dof_indices_filter);
            constraints_filter.distribute_local_to_global(cell_rhs,
                                                          local_dof_indices_filter,
                                                          filter_rhs);
        }
    }

    // Finalize the assembly and update ghost values for the solver.
    filter_rhs.compress(VectorOperation::add);
    filter_rhs.update_ghost_values();
}

/**
 * @brief Solve the Helmholtz filter equation.
 *
 * The linear system
 * @f[
 *   \bigl(1 - R_{min}^2 \Delta\bigr)\, \tilde{\rho} = b
 * @f]
 * is solved with FGMRES preconditioned by a geometric multigrid cycle.
 * On every level except the coarsest, a Chebyshev smoother is used; on
 * the coarsest level, a Chebyshev smoother with the exact number of
 * iterations is used as an approximate coarse-grid solve
 * (MGCoarseGridApplySmoother). Hanging-node constraints are distributed
 * into the solution after the solve, so the filtered density is
 * consistent.
 *
 * @note Because the operator (filter_matrix + mg_matrices_filter) is
 *       identical for density and sensitivity filtering, this solver can
 *       be invoked twice per optimization iteration — once with the
 *       density RHS and once with the sensitivity RHS — without any
 *       reassembly. Only `filter_rhs` and `filter_solution` need to be
 *       swapped between the two calls.
 */
template<unsigned int dim, unsigned int degree>
void Elasticity_Solver<dim, degree>::solve_filter_chebyshev()
{
    TimerOutput::Scope t(computing_timer, "solve_filter_chebyshev");

    // Set up the multigrid transfer between levels.
    MGTransferMatrixFree<dim, float> mg_transfer_filter(mg_constrained_dofs_filter);
    mg_transfer_filter.build(dof_handler_filter);

    // Chebyshev smoother configuration for every level.
    using SmootherType = PreconditionChebyshev<LeveLFilterMatrixType, LeveLVectorType>;
    mg::SmootherRelaxation<SmootherType, LeveLVectorType> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        if (level > 0)
        {
            // Smoother on the finer levels.
            smoother_data[level].smoothing_range = 15.;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            // Approximate coarse solve: iterate until the smoother has
            // effectively converged.
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices_filter[0].m();
        }
        // The inverse diagonal is used as the diagonal of the smoother.
        mg_matrices_filter[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices_filter[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices_filter, smoother_data);

    // Coarse solver: apply the smoother on the coarsest level.
    MGCoarseGridApplySmoother<LeveLVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    // Level operators and interface (edge) matrices.
    mg::Matrix<LeveLVectorType> mg_matrix(mg_matrices_filter);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LeveLFilterMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        mg_interface_matrices[level].initialize(mg_matrices_filter[level]);
    }

    mg::Matrix<LeveLVectorType> mg_interface(mg_interface_matrices);

    // Assemble the multigrid cycle. The smoother is used both as
    // pre- and post-smoother.
    Multigrid<LeveLVectorType> mg(mg_matrix, mg_coarse, mg_transfer_filter, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    // Wrap the multigrid cycle as a preconditioner for FGMRES.
    PreconditionMG<dim, LeveLVectorType, MGTransferMatrixFree<dim, float>>
        preconditioner(dof_handler_filter, mg, mg_transfer_filter);

    // Solve the filter equation.
    filter_solution = 0;
    SolverControl solver_control_filter(dof_handler_filter.n_dofs(),1e-12);
    SolverFGMRES<VectorType> solver_filter(solver_control_filter);
    solver_filter.solve(filter_matrix,
                        filter_solution,
                        filter_rhs,
                        preconditioner);

    // Distribute constraints and update ghost values so the solution can
    // be used in the averaging step.
    constraints_filter.distribute(filter_solution);
    filter_solution.update_ghost_values();
    pcout<<"                             "<< "Filter  Solver Last Steps : "
         << solver_control_filter.last_step() << std::endl;
}

/**
 * @brief Project the filtered (continuous) density onto the DGQ(0) space.
 *
 * For each locally owned filter cell, the average of the filtered density
 * over the cell is computed from the continuous DoF values and written
 * back to the DGQ(0) vector @p output. This produces the piecewise constant
 * filtered density that is subsequently fed into the SIMP interpolation
 * (and thus into the Laplace operator).
 *
 * @param output DGQ(0) vector that receives the cell-averaged filtered density.
 */
template<unsigned int dim, unsigned int degree>
void Elasticity_Solver<dim, degree>::generate_average_vector
    (LinearAlgebra::distributed::Vector<double> &output)
{
    TimerOutput::Scope t(computing_timer, "generate_average_vector");

    // Pair continuous filter cells with the corresponding DGQ(0) cells.
    auto cell_begin_aver = dof_handler_filter.begin_active();
    auto cell_end_aver = dof_handler_filter.end();
    auto rho_begin_aver = dof_handler_simp.begin_active();

    Vector<double> dofs_value(fe_filter.n_dofs_per_cell());
    Vector<double> average_values(1);
    for (; cell_begin_aver != cell_end_aver; ++cell_begin_aver, ++rho_begin_aver)
    {
        if (cell_begin_aver->is_locally_owned() && cell_begin_aver->level() == rho_begin_aver->level())
        {
            // Average all DoF values of the cell: for an FE_Q space this
            // is the cell mean of the filtered density.
            cell_begin_aver->get_dof_values(filter_solution, dofs_value);
            average_values[0] = dofs_value.mean_value();
            rho_begin_aver->set_dof_values(average_values, output);
        }
    }
    output.compress(VectorOperation::insert);
    output.update_ghost_values();
}