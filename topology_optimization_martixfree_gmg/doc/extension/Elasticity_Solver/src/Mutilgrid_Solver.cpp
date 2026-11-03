#include "../include/Elasticity_Matrix_Free.h"

using namespace Elasticity_MF;

template<unsigned int dim, unsigned degree>
void Elasticity_Solver<dim, degree>::setup_elasticity_system()
{
    TimerOutput::Scope t(computing_timer, "setup_elasticity_system");
    //
    elasticity_matrix.clear();
    //
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();
    //
    constraints.clear();
    constraints.reinit(dof_handler.locally_owned_dofs(),
                   DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(mapping, 
                                             dof_handler, 
                                             1, 
                                             Functions::ZeroFunction<dim,double>(dim), 
                                             constraints);
    constraints.close();
    //
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
    std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());
    system_mf_storage->reinit(mapping,
                              dof_handler,
                              constraints,
                              QGauss<1>(fe.degree + 1),
                              additional_data);
    elasticity_matrix.initialize(system_mf_storage);

    elasticity_matrix.initialize_dof_vector(elasticity_rhs);
    elasticity_matrix.initialize_dof_vector(elasticity_displacement);
    elasticity_matrix.initialize_dof_vector(Simp_Rho_Continuous);
}

template<unsigned int dim, unsigned degree>
void Elasticity_Solver<dim, degree>::setup_multigrid_elasticity_system()
{
    TimerOutput::Scope t(computing_timer, "setup_multigrid_elasticity_system");
    mg_matrices.clear_elements();
    mg_constrained_dofs.clear();
    //
    const unsigned int nlevels = triangulation.n_levels();
    mg_matrices.resize(0, nlevels - 1);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, {1});
    //
    for (unsigned int level = 0; level < nlevels; level++)
    {    
        AffineConstraints<float> level_constraints(dof_handler.locally_owned_mg_dofs(level),
                                 DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

        for (const types::global_dof_index dof_index : mg_constrained_dofs.get_boundary_indices(level))
        {
            level_constraints.constrain_dof_to_zero(dof_index);
        } 
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(std::make_shared<MatrixFree<dim, float>>());
        mg_mf_storage_level->reinit(mapping,
                                    dof_handler,
                                    level_constraints,
                                    QGauss<1>(fe.degree + 1),
                                    additional_data);

        mg_matrices[level].initialize(mg_mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
    }
}

template<unsigned int dim, unsigned int degree>
void Elasticity_Solver<dim, degree>::transfer_simp_vector()
{
    TimerOutput::Scope t(computing_timer, "transfer_simp_vector");
    
    Simp_Rho_Continuous = 0;

    unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_rho_dof_indices(fe_simp.n_dofs_per_cell());

    // Walk over continuous cells and DGQ(0) density cells in
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
            for (unsigned int i = 0; i < n_dofs_per_cell; i++)
                Simp_Rho_Continuous[local_dof_indices[i]] = rho;
        }
    }
    Simp_Rho_Continuous.compress(VectorOperation::insert);
    Simp_Rho_Continuous.update_ghost_values();
}

/**
 * @brief Evaluate the SIMP on all levels.
 *
 * Steps:
 *  1. Build the continuous density carrier (transfer_simp_vector).
 *  2. Evaluate the SIMP per cell of the fine-level operator
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
void Elasticity_Solver<dim, degree>::interpolate_simp_vector()
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
    elasticity_matrix.evaluate_coefficient(Simp_Rho_Continuous,
                                        Material_Lambda,
                                        Material_Mu);
    elasticity_matrix.compute_diagonal();

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
                                            static_cast<float>(Material_Lambda),
                                            static_cast<float>(Material_Mu));
        mg_matrices[i].compute_diagonal();
    }
}

template<unsigned int dim, unsigned degree>
void Elasticity_Solver<dim, degree>::assemble_elasticity_rhs()
{
    TimerOutput::Scope t(computing_timer, "assemble_elasticity_rhs");
    elasticity_rhs = 0;
    QGauss<dim> quadrature_formula(fe.degree + 1);
    QGauss<dim - 1> quadrature_formula_face(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                           quadrature_formula_face,
                                           update_values | update_gradients |
                                           update_quadrature_points | update_JxW_values);
    unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    Vector<double> load_force(dim);
    load_force[1] =-1.0;

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    //
    auto cell_begin = dof_handler.begin_active();
    auto cell_end = dof_handler.end();
    //
    for (; cell_begin != cell_end; ++cell_begin)
    {
        if (cell_begin->is_locally_owned())
        {
            cell_rhs = 0;
            fe_values.reinit(cell_begin);
            cell_begin->get_dof_indices(local_dof_indices);
            for (auto cellface : cell_begin->face_iterators())
            {
                if (cellface->at_boundary() && cellface->boundary_id() == 2)
                {
                    fe_face_values.reinit(cell_begin, cellface);
                    for (const unsigned int i : fe_values.dof_indices())
                    {
                        const unsigned int component_i = 
                            cell_begin->get_fe().system_to_component_index(i).first;
                        for (const unsigned int q_point : fe_face_values.quadrature_point_indices())
                        {
                        cell_rhs(i) += (fe_face_values.shape_value(i, q_point) *
                                        load_force[component_i] *
                                        fe_face_values.JxW(q_point));
                        }
                    }
                }
            }
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   elasticity_rhs);                
        }
    }
    elasticity_rhs.compress(VectorOperation::add);
    elasticity_rhs.update_ghost_values();
}

template<unsigned int dim, unsigned degree>
void Elasticity_Solver<dim, degree>::solve_chebyshev()
{
    TimerOutput::Scope t(computing_timer, "solve_chebyshev");
    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    using SmootherType = PreconditionChebyshev<LeveLElasticityMatrixType, LeveLVectorType>;
    mg::SmootherRelaxation<SmootherType, LeveLVectorType> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        if (level > 0)
        {
            smoother_data[level].smoothing_range = 10.;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            smoother_data[0].smoothing_range = 1;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LeveLVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LeveLVectorType> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LeveLElasticityMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        mg_interface_matrices[level].initialize(mg_matrices[level]);
    }

    mg::Matrix<LeveLVectorType> mg_interface(mg_interface_matrices);

    Multigrid<LeveLVectorType> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LeveLVectorType, MGTransferMatrixFree<dim, float>> preconditioner(dof_handler, mg, mg_transfer);
    
    if (MMA_Optimizer.Loop_Iter == 1)
    {
        elasticity_displacement = 0.0;
    }
    SolverControl solve_control(dof_handler.n_dofs(), 1e-5 * elasticity_rhs.l2_norm());
    SolverBicgstab<VectorType> solver(solve_control);
    solver.solve(elasticity_matrix,
                 elasticity_displacement,
                 elasticity_rhs,
                 preconditioner);
    constraints.distribute(elasticity_displacement);
    elasticity_displacement.update_ghost_values();
    pcout<<"                             "<< "Elasticity Solver Last Steps : "<< solve_control.last_step() << std::endl;
}

template<unsigned int dim, unsigned int degree>
void Elasticity_Solver<dim, degree>::generate_object_discrete_derivative()
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
    
    FEValuesExtractors::Vector displacement(0);
    std::vector<SymmetricTensor<2, dim, double>> 
        solution_gradients(fe_values.n_quadrature_points);

    double Lambda_Derivative, Mu_Derivative;

    // Lock-step iteration over cells and DGQ(0) cells.
    auto cell_begin = dof_handler.begin_active();
    auto cell_end = dof_handler.end();
    auto base_begin = dof_handler_simp.begin_active();

    for (; cell_begin != cell_end; ++cell_begin, ++base_begin)
    {
        if (cell_begin->is_locally_owned())
        {
            fe_values.reinit(cell_begin);

            // SIMP derivative dk/drho at the current (filtered) density.
            base_begin->get_dof_indices(local_simp_dof_indices);
            rho_values[0] = Simp_Rho_Discrete_Filter[local_simp_dof_indices[0]];
            
            // \frac{\partial \lambda}{\partial \rho} 
            Lambda_Derivative = penal * (1. - epsimin) * Material_Lambda
                                * pow(rho_values[0], penal - 1);
            // \frac{\partial \mu}{\partial \rho} 
            Mu_Derivative     = penal * (1. - epsimin) * Material_Mu
                                * pow(rho_values[0], penal - 1);
            // \frac{\partial C_{ijkl}}{\partial \rho} 
            auto C_ijkl_Derivative = get_stress_strain_tensor<dim>(Lambda_Derivative,Mu_Derivative);

            // Primal solution gradient at the quadrature points. Because
            // the problem is self-adjoint, the adjoint solution equals
            // solution.
            fe_values[displacement].get_function_symmetric_gradients(
                elasticity_displacement, solution_gradients);

            // dJ/drho = - dk/drho * |grad u|^2, integrated over the cell.
            for (unsigned int q_point : fe_values.quadrature_point_indices())
            {
                Object_Discrete_Derivative[local_simp_dof_indices[0]] +=
                                         -solution_gradients[q_point] *
                                          C_ijkl_Derivative *
                                          solution_gradients[q_point] *
                                          fe_values.JxW(q_point);

            }
        }
    }
    Object_Discrete_Derivative.compress(VectorOperation::add);
}