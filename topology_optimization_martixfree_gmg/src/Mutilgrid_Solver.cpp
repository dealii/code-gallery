/**
 * @brief Implementation of the primal (thermal conduction) problem, the
 *        SIMP coefficient transfer, and the objective sensitivity.
 *
 * This translation unit implements the primal-problem members of
 * Heat_Solver:
 *
 *  - setup_laplace_system()              : fine-level matrix-free Laplace
 *                                          operator for the temperature.
 *  - setup_multigrid_laplace_system()    : geometric multigrid hierarchy
 *                                          for the Laplace operator.
 *  - transfer_simp_vector()              : transfer the filtered DGQ(0)
 *                                          density into the continuous
 *                                          temperature vector.
 *  - interpolate_simp_vector()           : re-evaluate the SIMP coefficient
 *                                          on all levels and interpolate the
 *                                          density to the multigrid levels.
 *  - assemble_laplace_rhs()              : assemble the (constant) heat
 *                                          source right-hand side.
 *  - solve_chebyshev()                   : solve the primal problem with a
 *                                          Chebyshev-preconditioned
 *                                          BiCGStab on top of multigrid.
 *  - generate_object_discrete_derivative(): evaluate the discrete objective
 *                                          derivative (sensitivity) via the
 *                                          self-adjoint formulation.
 *
 * The primal problem is
 * @f[
 *   -\nabla\cdot\bigl(k(\rho)\,\nabla u\bigr) = f,
 * @f]
 * with homogeneous Dirichlet data on the boundary with id 1 and a constant
 * volumetric heat source @f$ f @f$.
 */

#include "../include/Heat_Matrix_Free.h"

using namespace Heat_Matrix_Free;

/**
 * @brief Set up the fine-level matrix-free Laplace operator.
 *
 * This routine
 *  - distributes the DoFs of the continuous temperature space (fe),
 *  - builds the hanging-node constraints and imposes homogeneous
 *    Dirichlet boundary values on boundary id 1,
 *  - initializes the matrix-free Laplace operator on the fine grid,
 *  - allocates the RHS, temperature and continuous-density vectors.
 *
 * @note The MatrixFree data structure (DoF numbering, constraints,
 *       quadrature) only depends on the mesh. Within one optimization loop
 *       this setup is therefore executed once and only repeated after a
 *       mesh change. The density dependence of the operator enters later
 *       through evaluate_coefficient() and compute_diagonal() in
 *       interpolate_simp_vector(), which are called every iteration.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::setup_laplace_system()
{
    TimerOutput::Scope t(computing_timer, "setup_laplace_system");

    // Rebuild the fine-level operator from scratch.
    laplace_matrix.clear();

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    // Hanging-node constraints plus zero Dirichlet values on boundary 1.
    constraints.clear();
    constraints.reinit(dof_handler.locally_owned_dofs(),
                   DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             1,
                                             Functions::ConstantFunction<dim,double>(0),
                                             constraints);
    constraints.close();

    // We need values and gradients for the Laplace operator.
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_gradients | 
                                            update_JxW_values | 
                                            update_quadrature_points | 
                                            update_values);
    std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());
    system_mf_storage->reinit(mapping,
                              dof_handler,
                              constraints,
                              QGauss<1>(fe.degree + 1),
                              additional_data);
    laplace_matrix.initialize(system_mf_storage);

    // Allocate the vectors owned by the primal system.
    laplace_matrix.initialize_dof_vector(laplace_rhs);
    laplace_matrix.initialize_dof_vector(laplace_temperature);
    laplace_matrix.initialize_dof_vector(Simp_Rho_Continuous);
}

/**
 * @brief Build the geometric multigrid hierarchy for the Laplace operator.
 *
 * A level-wise matrix-free Laplace operator is created on every level.
 * Zero Dirichlet constraints for the boundary with id 1 are passed to the
 * level operators through @p mg_constrained_dofs. Single precision is used
 * on all levels inside the multigrid preconditioner (the fine-level
 * operator in setup_laplace_system works in double precision).
 *
 * @note Like setup_laplace_system(), this only depends on the mesh. The
 *       per-cell SIMP conductivity is set on every level later, in
 *       interpolate_simp_vector(), so this routine is not part of the
 *       optimization loop.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::setup_multigrid_laplace_system()
{
    TimerOutput::Scope t(computing_timer, "setup_multigrid_laplace_system");

    // Reset the level operators and the level constraints.
    mg_matrices.clear_elements();
    mg_constrained_dofs.clear();

    const unsigned int nlevels = triangulation.n_levels();
    mg_matrices.resize(0, nlevels - 1);

    // Extract the boundary DoFs once; they are added to the level
    // constraints below.
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, {1});

    for (unsigned int level = 0; level < nlevels; level++)
    {
        // Level-local constraints, including the zero Dirichlet DoFs.
        AffineConstraints<float> level_constraints(dof_handler.locally_owned_mg_dofs(level),
                                 DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

        for (const types::global_dof_index dof_index : mg_constrained_dofs.get_boundary_indices(level))
        {
            level_constraints.constrain_dof_to_zero(dof_index);
        }
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | 
                                                update_JxW_values | 
                                                update_quadrature_points | 
                                                update_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(std::make_shared<MatrixFree<dim, float>>());
        mg_mf_storage_level->reinit(mapping,
                                    dof_handler,
                                    level_constraints,
                                    QGauss<1>(fe.degree + 1),
                                    additional_data);

        // Level operator. Its SIMP coefficient and diagonal are set in
        // interpolate_simp_vector().
        mg_matrices[level].initialize(mg_mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
    }
}

/**
 * @brief Transfer the filtered DGQ(0) density into the continuous vector.
 *
 * The design density lives in a DGQ(0) space (one value per cell). To make
 * it available inside the matrix-free Laplace operator — which operates on
 * a continuous FE_Q vector — the DGQ(0) value of every locally owned cell
 * is accumulated into the *first* local DoF of the corresponding
 * continuous cell. This is the value later picked up by
 * Laplace_Matrix_Free::evaluate_coefficient(), which reads only
 * `get_dof_value(0)` of each cell to build the per-cell SIMP conductivity.
 *
 * @note This is a lightweight carrier mechanism rather than a
 *       mathematical projection: only the first DoF per cell is used, so
 *       no interpolation error is introduced for the piecewise constant
 *       density. The actual smoothing of the density is done by the
 *       Helmholtz filter before this transfer (see the filter
 *       implementation).
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::transfer_simp_vector()
{
    TimerOutput::Scope t(computing_timer, "transfer_simp_vector");

    unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_rho_dof_indices(fe_simp.n_dofs_per_cell());

    // Walk over continuous temperature cells and DGQ(0) density cells in
    // lock-step.
    auto cell_iter_begin = dof_handler.begin_active();
    auto cell_iter_end = dof_handler.end();
    auto simp_iter = dof_handler_simp.begin_active();

    for (; cell_iter_begin != cell_iter_end; cell_iter_begin++, simp_iter++)
    {
        if (cell_iter_begin->is_locally_owned())
        {
            cell_iter_begin->get_dof_indices(local_dof_indices);
            simp_iter->get_dof_indices(local_rho_dof_indices);
            const double rho = Simp_Rho_Discrete_Filter[local_rho_dof_indices[0]];
            // Only the first local DoF per cell is used by the operator.
            for (unsigned int i = 0; i < n_dofs_per_cell; i ++)
                Simp_Rho_Continuous[local_dof_indices[i]] = rho;
        }
    }

    Simp_Rho_Continuous.compress(VectorOperation::insert);
    Simp_Rho_Continuous.update_ghost_values();
}

/**
 * @brief Evaluate the SIMP conductivity on all levels.
 *
 * Steps:
 *  1. Build the continuous density carrier (transfer_simp_vector).
 *  2. Evaluate the SIMP conductivity per cell of the fine-level operator
 *     and recompute its inverse diagonal.
 *  3. Interpolate the density to all multigrid levels.
 *  4. Repeat step 2 for every level operator.
 *
 * This routine is part of the optimization loop: the MatrixFree objects
 * are reused, only the per-cell coefficient and the diagonal change.
 *
 * @note This is the counterpart of the "same matrix, changing
 *       coefficient" idea: the SIMP coefficient makes the operator
 *       density-dependent, but the sparsity/DoF structure stays fixed, so
 *       only the coefficient table and the diagonal are refreshed here.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::interpolate_simp_vector()
{
    TimerOutput::Scope t(computing_timer, "interpolate_simp_vector");

    // Allocate the level-wise density vectors.
    const unsigned int nlevels = triangulation.n_levels();
    Simp_Rho_Continuous_Level.clear();
    Simp_Rho_Continuous_Level.resize(0, nlevels - 1);
    for (unsigned int level = 0; level < nlevels; level++)
    {
        mg_matrices[level].initialize_dof_vector(Simp_Rho_Continuous_Level[level]);
    }

    // Fine level: build the density carrier, evaluate the SIMP
    // coefficient and refresh the diagonal.
    transfer_simp_vector();
    laplace_matrix.evaluate_coefficient(Simp_Rho_Continuous,
        make_vectorized_array<double>(Material_Thermal_Conductivity));
    laplace_matrix.compute_diagonal();

    // Interpolate the density to all multigrid levels.
    MGTransferMatrixFree<dim, float> mg_vector_transfer;
    mg_vector_transfer.build(dof_handler);
    mg_vector_transfer.interpolate_to_mg(dof_handler,
                                         Simp_Rho_Continuous_Level,
                                         Simp_Rho_Continuous);

    // Level operators: same coefficient/diagonal refresh as on the fine level.
    for (unsigned int i = 0; i < nlevels; i++)
    {
        Simp_Rho_Continuous_Level[i].update_ghost_values();
        mg_matrices[i].evaluate_coefficient(Simp_Rho_Continuous_Level[i],
            make_vectorized_array<float>(static_cast<float>(Material_Thermal_Conductivity)));
        mg_matrices[i].compute_diagonal();
    }
}

/**
 * @brief Assemble the right-hand side of the primal problem.
 *
 * The heat source is a constant volumetric term of magnitude 10.0; the
 * cell-local right-hand side is
 * @f[
 *   b_i = \int_{\Omega_K} \phi_i \cdot 10 \, \mathrm{d}x .
 * @f]
 * The constraints apply the homogeneous Dirichlet data during the
 * assembly (distribute_local_to_global).
 *
 * @note The RHS is independent of the density; it only changes when the
 *       mesh changes.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::assemble_laplace_rhs()
{
    TimerOutput::Scope t(computing_timer, "assemble_laplace_rhs");

    laplace_rhs = 0;

    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Lock-step iteration over temperature cells and DGQ(0) cells.
    auto cell_begin = dof_handler.begin_active();
    auto cell_end = dof_handler.end();
    auto rho_begin = dof_handler_simp.begin_active();

    for (; cell_begin != cell_end; ++cell_begin, ++rho_begin)
    {
        if (cell_begin->is_locally_owned() && cell_begin->level() == rho_begin->level())
        {
            cell_rhs = 0;
            fe_values.reinit(cell_begin);
            cell_begin->get_dof_indices(local_dof_indices);

            // Constant heat source with magnitude 10.0.
            for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                for (const unsigned int i : fe_values.dof_indices())
                {
                    cell_rhs[i] += (fe_values.shape_value(i, q_index) *
                                    10.0 *
                                    fe_values.JxW(q_index));
                }
            }
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   laplace_rhs);
        }
    }

    laplace_rhs.compress(VectorOperation::add);
    laplace_rhs.update_ghost_values();
}

/**
 * @brief Solve the primal thermal conduction problem.
 *
 * The system
 * @f[
 *   -\nabla\cdot\bigl(k(\rho)\,\nabla u\bigr) = f
 * @f]
 * is solved with BiCGStab preconditioned by a geometric multigrid cycle.
 * The multigrid uses a Chebyshev smoother on every level; the coarsest
 * level applies the smoother until convergence as an approximate direct
 * solve. Since the SIMP operator is symmetric, CG would also be possible;
 * BiCGStab is chosen for robustness against the ill-conditioning caused
 * by very small conductivities (near-void cells).
 *
 * @note The multigrid objects (transfer, smoothers, interface matrices)
 *       are constructed here from the already prepared level operators,
 *       whose SIMP coefficients were refreshed in interpolate_simp_vector().
 *       No operator assembly happens in this routine.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::solve_chebyshev()
{
    TimerOutput::Scope t(computing_timer, "solve_chebyshev");

    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    // Chebyshev smoother configuration for every level.
    using SmootherType = PreconditionChebyshev<LeveLLaplaceMatrixType, LeveLVectorType>;
    mg::SmootherRelaxation<SmootherType, LeveLVectorType> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        if (level > 0)
        {
            // Smoother on the finer levels.
            smoother_data[level].smoothing_range = 10.;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            // Approximate coarse solve on the coarsest level.
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
        // Refresh the diagonal (the coefficient has changed since the
        // previous iteration) and use its inverse as smoother diagonal.
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    // Coarse solve = smoother on the coarsest level.
    MGCoarseGridApplySmoother<LeveLVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    // Level operators and interface (edge) matrices.
    mg::Matrix<LeveLVectorType> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LeveLLaplaceMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        mg_interface_matrices[level].initialize(mg_matrices[level]);
    }

    mg::Matrix<LeveLVectorType> mg_interface(mg_interface_matrices);

    // Multigrid cycle with the smoother as pre- and post-smoother.
    Multigrid<LeveLVectorType> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LeveLVectorType, MGTransferMatrixFree<dim, float>> preconditioner(dof_handler, mg, mg_transfer);

    // Solve with BiCGStab. The tolerance is relative to the RHS norm.
    SolverControl solve_control(dof_handler.n_dofs(), 1e-5 * laplace_rhs.l2_norm());
    laplace_temperature = 0;
    SolverBicgstab<VectorType> solver(solve_control);
    solver.solve(laplace_matrix,
                 laplace_temperature,
                 laplace_rhs,
                 preconditioner);

    // Apply constraints and refresh ghosts for the subsequent
    // sensitivity evaluation.
    constraints.distribute(laplace_temperature);
    laplace_temperature.update_ghost_values();
    pcout<<"                             "<< "Laplace Solver Last Steps : "<< solve_control.last_step() << std::endl;
}

/**
 * @brief Evaluate the discrete objective derivative (sensitivity).
 *
 * For the thermal compliance objective
 * @f[
 *   J = \int_\Omega k(\rho)\,|\nabla u|^2\,\mathrm{d}x,
 * @f]
 * the derivative with respect to the cell density is, using the
 * self-adjointness of the primal operator (adjoint solution = primal
 * solution),
 * @f[
 *   \frac{\partial J}{\partial \rho_K}
 *   = -\int_{\Omega_K} \frac{\partial k}{\partial \rho}(\rho)\,
 *     |\nabla u|^2\,\mathrm{d}x ,
 * @f]
 * with the SIMP derivative
 * @f[
 *   \frac{\partial k}{\partial \rho}(\rho)
 *   = p\,(1-\varepsilon_{min})\,k_{solid}\,\rho^{p-1}.
 * @f]
 *
 * The result is stored in Object_Discrete_Derivative, which is a DGQ(0)
 * vector (one entry per cell). Before being handed to the optimizer, the
 * derivative is filtered (using the same Helmholtz operator as the density
 * filter, only with a different right-hand side).
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::generate_object_discrete_derivative()
{
    TimerOutput::Scope t(computing_timer, "generate_object_discrete_derivative");

    Object_Discrete_Derivative = 0;

    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    std::vector<types::global_dof_index> local_simp_dof_indices(fe_simp.n_dofs_per_cell());
    std::vector<double> rho_values(1);

    std::vector<Tensor<1, dim, double>> solution_gradients(fe_values.n_quadrature_points);

    double Thermal_Conductivity_Derivative;

    // Lock-step iteration over temperature cells and DGQ(0) cells.
    auto cell_begin = dof_handler.begin_active();
    auto cell_end = dof_handler.end();
    auto base_begin = dof_handler_simp.begin_active();

    for (; cell_begin != cell_end; ++cell_begin, ++base_begin)
    {
        if (cell_begin->is_locally_owned() && cell_begin->level() == base_begin->level())
        {
            fe_values.reinit(cell_begin);

            // SIMP derivative dk/drho at the current (filtered) density.
            base_begin->get_dof_indices(local_simp_dof_indices);
            rho_values[0] = Simp_Rho_Discrete_Filter[local_simp_dof_indices[0]];
            Thermal_Conductivity_Derivative = penal * (1. - epsimin) * Material_Thermal_Conductivity
                                                     * pow(rho_values[0], penal - 1);

            // Primal solution gradient at the quadrature points. Because
            // the problem is self-adjoint, the adjoint solution equals
            // laplace_temperature.
            fe_values.get_function_gradients(laplace_temperature, solution_gradients);

            // dJ/drho = - dk/drho * |grad u|^2, integrated over the cell.
            for (unsigned int q_point : fe_values.quadrature_point_indices())
            {
                Object_Discrete_Derivative[local_simp_dof_indices[0]] +=
                                         -solution_gradients[q_point] *
                                          Thermal_Conductivity_Derivative *
                                          solution_gradients[q_point] *
                                          fe_values.JxW(q_point);

            }
        }
    }

    Object_Discrete_Derivative.compress(VectorOperation::add);
}