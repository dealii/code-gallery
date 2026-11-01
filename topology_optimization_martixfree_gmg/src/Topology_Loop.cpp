/**
 * @brief Driver for the matrix-free heat-conduction topology optimization:
 *        object construction, mesh generation, SIMP bookkeeping, output and
 *        the main optimization loop.
 *
 * This translation unit implements the "infrastructure" members of
 * Heat_Solver:
 *
 *  - Heat_Solver()      : construct the distributed triangulation, the
 *                         mappings, FE objects, DoFHandlers, output stream
 *                         and the timing object.
 *  - make_grid()        : create the hyper-cube mesh and tag the Dirichlet
 *                         boundary (a small patch at the bottom center).
 *  - setup_simp_system(): allocate the DGQ(0) density vectors and the
 *                         optimizer-related vectors.
 *  - get_cell_volume()  : store the per-cell measure in a DGQ(0) vector.
 *  - output_results()   : write VTU/PVTU output of the design and the
 *                         sensitivity for visualization.
 *  - run()              : run the full optimization loop (filter, primal,
 *                         sensitivity, sensitivity filter, MMA update).
 *  - time()             : print the accumulated TimerOutput summary.
 *
 * The overall workflow is:
 *   make_grid
 *     -> setup_simp_system / get_cell_volume
 *     -> setup_laplace_system / setup_multigrid_laplace_system   (mesh only)
 *     -> setup_filter_system  / setup_multigrid_filter_system    (mesh only)
 *     -> loop:
 *          density filter (assemble_filter_rhs + solve_filter_chebyshev)
 *          SIMP coefficient (interpolate_simp_vector)
 *          primal solve     (assemble_laplace_rhs + solve_chebyshev)
 *          sensitivity      (generate_object_discrete_derivative)
 *          sensitivity filter (assemble_filter_rhs + solve_filter_chebyshev
 *                              reusing the SAME Helmholtz operator)
 *          MMA update
 */

#include "../include/Heat_Matrix_Free.h"

using namespace Heat_Matrix_Free;

/**
 * @brief Construct the solver and its fixed configuration.
 *
 * Notes on the constructor initializer list:
 *
 *  - The triangulation is built with
 *      * @p limit_level_difference_at_vertices, which keeps at most one
 *        hanging level difference at each vertex and is required for a
 *        proper geometric multigrid hierarchy, and
 *      * @p construct_multigrid_hierarchy, which enables
 *        distribute_mg_dofs() / level DoF handling used by the multigrid
 *        operators.
 *  - The mapping for the temperature space matches the FE degree
 *    (MappingFE<dim> with FE_Q(degree)); the filter mapping uses FE_Q(1)
 *    because the Helmholtz filter is discretized with FE_Q(1) only.
 *  - fe_simp is a DGQ(0) (piecewise constant) element — one DoF per cell,
 *    used for the design density.
 *  - The TimerOutput is configured as MPI-collective, never auto-printed
 *    (printed explicitly via time()), and records wall-clock times.
 */
template<unsigned int dim, unsigned int degree>
Heat_Solver<dim, degree>::Heat_Solver()
    : mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      mapping(FE_Q<dim>(degree)),
      mapping_filter(FE_Q<dim>(1)),
      fe(degree),
      fe_simp(0),
      fe_filter(1),
      dof_handler(triangulation),
      dof_handler_simp(triangulation),
      dof_handler_filter(triangulation),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
      computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
{};

/**
 * @brief Generate the coarse mesh and tag the Dirichlet boundary.
 *
 * A hyper-cube [-1,1]^dim is created and globally refined
 * (11 - dim) times, so that the 2D and 3D runs have comparable resolution.
 *
 * The homogeneous Dirichlet condition is applied on a *point-like* patch
 * at the bottom center of the domain (a small square in 2D, a small
 * square patch in 3D), mimicking a heat sink. This is implemented by
 * re-assigning boundary id 1 to the faces whose center lies inside the
 * patch; every other face keeps the default boundary id 0 (natural /
 * homogeneous Neumann).
 *
 * Note that the boundary id chosen here (1) is the one used by
 * setup_laplace_system() (VectorTools::interpolate_boundary_values) and
 * setup_multigrid_laplace_system() (mg_constrained_dofs).
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::make_grid()
{
    TimerOutput::Scope t(computing_timer, "make_grid");

    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(10 - dim);

    for (auto &face : triangulation.active_face_iterators())
    {
        if constexpr (dim == 3)
        {
            // Bottom-center patch of size 0.2 x 0.2 in the z = -1 plane.
            if (std::fabs(face->center()(0) - 0.) <=  0.1 &&
                std::fabs(face->center()(1) - 0.) <=  0.1 &&
                std::fabs(face->center()(2) + 1.) <=  1e-10)
            {
                face->set_boundary_id(1);
            }
        }
        else if constexpr (dim == 2)
        {
            // Bottom-center segment of length 0.2 on y = -1.
            if (std::fabs(face->center()(0) - 0.) <=  0.1 &&
                std::fabs(face->center()(1) + 1.) <=  1e-10)
            {
                face->set_boundary_id(1);
            }
        }
    }
}

/**
 * @brief Set up the DGQ(0) density space and the optimizer vectors.
 *
 * All design and optimizer vectors share the same DoF layout (one DoF per
 * active cell on the locally owned subdomain):
 *
 *  - Simp_Rho_Discrete           : current design density,
 *  - Simp_Rho_Discrete_Filter    : filtered design density (DGQ(0)),
 *  - Simp_Rho_Discrete_Max/Min   : MMA move limits,
 *  - Cell_Volume_Discrete        : per-cell measure (filled by get_cell_volume),
 *  - Object_Discrete_Derivative  : objective sensitivity (DGQ(0)),
 *  - Constraint_Discrete_Derivative[0] : volume-constraint sensitivity.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::setup_simp_system()
{
    TimerOutput::Scope t(computing_timer, "setup_simp_system");

    dof_handler_simp.distribute_dofs(fe_simp);

    Simp_Rho_Discrete.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Simp_Rho_Discrete_Filter.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Simp_Rho_Discrete_Max.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Simp_Rho_Discrete_Min.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);

    Cell_Volume_Discrete.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Object_Discrete_Derivative.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);

    // Exactly one constraint (volume fraction).
    Constraint_Discrete_Derivative.resize(1);
    Constraint_Discrete_Derivative[0].reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
}

/**
 * @brief Store the measure of every locally owned cell in a DGQ(0) vector.
 *
 * Because fe_simp is DGQ(0), each cell has exactly one DoF, so the cell
 * measure can be written directly into Cell_Volume_Discrete.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::get_cell_volume()
{
    TimerOutput::Scope t(computing_timer, "get_cell_volume");

    Vector<double> cell_measure(1);
    for (auto &cell_iter : dof_handler_simp.active_cell_iterators())
    {
        if (cell_iter->is_locally_owned())
        {
            cell_measure[0] = cell_iter->measure();
            cell_iter->set_dof_values(cell_measure, Cell_Volume_Discrete);
        }
    }
    Cell_Volume_Discrete.compress(VectorOperation::insert);
    Cell_Volume_Discrete.update_ghost_values();
}

/**
 * @brief Write the current design, filtered density, cell volume and
 *        sensitivity to VTU/PVTU files.
 *
 * All fields are attached on the DGQ(0) DoFHandler, so one value per cell
 * is written. DataOut::build_patches() produces the VTU output; the
 * filename is suffixed with the current MMA iteration number, which makes
 * it easy to inspect the optimization history afterwards.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::output_results()
{
    TimerOutput::Scope t(computing_timer, "output_results");

    // Simp Output
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_simp);
    data_out.add_data_vector(dof_handler_simp, Simp_Rho_Discrete, "Simp_Rho_Discrete");
    data_out.add_data_vector(dof_handler_simp, Simp_Rho_Discrete_Filter, "Simp_Rho_Discrete_Filter");
    data_out.add_data_vector(dof_handler_simp, Cell_Volume_Discrete, "Cell_Volume_Discrete");
    data_out.add_data_vector(dof_handler_simp, Object_Discrete_Derivative, "Object_Discrete_Derivative");
    data_out.add_data_vector(dof_handler_simp, Constraint_Discrete_Derivative[0], "Constraint_Discrete_Derivative");
    data_out.build_patches();

    // Fast compression — this output is written every few iterations and
    // should not dominate the runtime.
    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record("./", "Simp_Optimization", MMA_Optimizer.Loop_Iter, mpi_communicator);
}

/**
 * @brief Run the full topology optimization loop.
 *
 * Structure of the driver:
 *
 *  1. Print the vectorization width and a banner.
 *  2. make_grid + setup_simp_system + get_cell_volume.
 *  3. Compute the filter radius from the cell size,
 *     @f$ R_{min} = 2 \cdot h / (2\sqrt{3}) @f$, where @f$ h @f$ is the cube
 *     root of the largest cell volume.
 *  4. Initialize the MMA optimizer (one constraint, zero lower bound,
 *     move limit 1e3).
 *  5. Set up the Laplace system + multigrid and the filter system +
 *     multigrid. **These two setups are mesh-dependent only** and are
 *     therefore executed once outside the loop.
 *  6. Initialize the volume-constraint sensitivity by filtering the cell
 *     volume and normalizing by volfrac * total_volume.
 *  7. Main loop (up to 1000 iterations or until the design change is
 *     below 1e-3 * n_cells):
 *       a. On the first iteration, start from a full (rho = 1) design.
 *       b. Density filter: RHS = density, solve Helmholtz filter,
 *          project the continuous result back to DGQ(0).
 *       c. Transfer the filtered density to the SIMP coefficient on all
 *          levels (interpolate_simp_vector), assemble the primal RHS,
 *          solve the primal problem, and evaluate the objective
 *          sensitivity.
 *       d. Sensitivity filter: RHS = objective sensitivity, solve the
 *          **same** Helmholtz filter, project back to DGQ(0).
 *       e. Evaluate the volume constraint and the objective value.
 *       f. MMA update of the design and termination check.
 *       g. Print the iteration status and, every 10 iterations, write
 *          VTU output.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::run()
{
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

    pcout << "Vectorization over " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;

    Vector<double> Object_Function(1);
    Vector<double> Constraint_Function(1);
    double change = 1e10;
    double volfrac = 0.3;

    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<"|                             INIT OPTIMIZATION                             |"<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<std::endl;

    // --- Mesh and DGQ(0) density bookkeeping -------------------------------
    make_grid();

    setup_simp_system();
    get_cell_volume();

    // Filter radius derived from the cell size: Rmin = 2 h / (2 sqrt(3)),
    // with h = (max cell volume)^(1/dim).
    Rmin = 2. * pow(Cell_Volume_Discrete.linfty_norm(), 1./(double)dim) / (2. * sqrt(3.));
    pcout<<"Rmin : "<<Rmin<<std::endl;
    pcout<<std::endl;

    // --- MMA optimizer -----------------------------------------------------
    // 1 constraint, 0 lower objective history, move limit 1e3, initial
    // design vector Simp_Rho_Discrete.
    MMA_Optimizer.Deal_II_MMA_Init(1,0,1e3,Simp_Rho_Discrete);
    MMA_Optimizer.Loop_Iter = 0;

    // --- Mesh-dependent operator setup (outside the loop) ------------------
    // The MatrixFree data structures and the multigrid hierarchies only
    // depend on the mesh and the (constant) filter radius. They are
    // therefore built exactly once here and reused throughout the loop.
    setup_laplace_system();
    setup_multigrid_laplace_system();

    setup_filter_system();
    setup_multigrid_filter_system();

    // --- Volume-constraint sensitivity ------------------------------------
    // Filter the cell volume with the SAME Helmholtz operator and
    // normalize by volfrac * total volume. This gives the derivative of
    // the normalized volume constraint with respect to the filtered
    // density.
    assemble_filter_rhs(Cell_Volume_Discrete);
    solve_filter_chebyshev();
    generate_average_vector(Constraint_Discrete_Derivative[0]);
    Constraint_Discrete_Derivative[0].operator/=(volfrac * Cell_Volume_Discrete.l1_norm());

    // --- Print DoF counts ---------------------------------------------------
    pcout<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<"|                             DEGREE OF FREEDOM                             |"<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<std::endl;

    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<"                             "<< "Simp    : "<< dof_handler_simp.n_dofs() << std::endl;
    pcout<<"                             "<< "Laplace : "<< dof_handler.n_dofs() << std::endl;
    pcout<<"                             "<< "Filter  : "<< dof_handler_filter.n_dofs() << std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<std::endl;

    // --- Main optimization loop --------------------------------------------
    pcout<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<"|                            START OPTIMIZATION                             |"<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    while (MMA_Optimizer.Loop_Iter < 1000 && change > 1e-3 * dof_handler_simp.n_dofs())
    {
        MMA_Optimizer.Loop_Iter++;

        // Start from a full design on the first iteration.
        if (MMA_Optimizer.Loop_Iter == 1)
        {
            Simp_Rho_Discrete = 1.;
            Simp_Rho_Discrete_Max = 1.;
            Simp_Rho_Discrete_Min = 1.e-3;
        }

        pcout<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<"|                             SOLVER INFOMATION                             |"<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<std::endl;

        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;

        // (1) Density filter: solve (1 - Rmin^2 Laplacian) rho_f = rho.
        assemble_filter_rhs(Simp_Rho_Discrete);
        solve_filter_chebyshev();
        generate_average_vector(Simp_Rho_Discrete_Filter);

        // (2) SIMP coefficient + primal solve + sensitivity.
        //     interpolate_simp_vector() refreshes the per-cell conductivity
        //     and the diagonal on every multigrid level; the MatrixFree
        //     structures themselves are reused.
        interpolate_simp_vector();
        assemble_laplace_rhs();
        solve_chebyshev();
        generate_object_discrete_derivative();

        // (3) Sensitivity filter: solve the SAME Helmholtz operator with
        //     the objective sensitivity as right-hand side. This is the
        //     "same matrix, different RHS" reuse discussed in the filter
        //     implementation.
        assemble_filter_rhs(Object_Discrete_Derivative);
        solve_filter_chebyshev();
        generate_average_vector(Object_Discrete_Derivative);

        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<std::endl;

        // --- Constraint function --------------------------------------------
        // g = (sum rho * V) / (volfrac * V_total) - 1
        Constraint_Function[0] = Simp_Rho_Discrete_Filter * Cell_Volume_Discrete;
        Constraint_Function[0] /= volfrac * Cell_Volume_Discrete.l1_norm();
        Constraint_Function[0] -= 1.;

        // --- Objective function ---------------------------------------------
        // Thermal compliance J = b^T u (self-adjoint, so u is both primal
        // and adjoint solution).
        Object_Function[0] = laplace_rhs * laplace_temperature;

        // --- MMA design update ----------------------------------------------
        MMA_Optimizer.Deal_II_MMA_Dual_Problem_Solve(Simp_Rho_Discrete,
                                                    Object_Discrete_Derivative,
                                                    Constraint_Function,
                                                    Constraint_Discrete_Derivative,
                                                    Simp_Rho_Discrete_Max,
                                                    Simp_Rho_Discrete_Min);
        change = MMA_Optimizer.Deal_II_MMA_Get_Change(Simp_Rho_Discrete);

        // --- Status output --------------------------------------------------
        pcout<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<"|                              OUTPUT  RESULTS                              |"<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;

        pcout<<std::endl;
        pcout<<"                             "<< "Iter    : "<<MMA_Optimizer.Loop_Iter<<std::endl
             <<"                             "<< "Object  : "<<Object_Function[0]<<std::endl
             <<"                             "<< "Change  : "<<change<<std::endl
             <<"                             "<< "Volume  : "<<(Constraint_Function[0] + 1) * volfrac<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<std::endl;

        // Write visualization output every 10 iterations.
        if (MMA_Optimizer.Loop_Iter % 10 == 0) { output_results(); };
    }

    // Print the accumulated timings after the loop.
    time();
}

/**
 * @brief Print the accumulated TimerOutput summary and reset it.
 */
template<unsigned int dim, unsigned int degree>
void Heat_Solver<dim, degree>::time()
{
    computing_timer.print_summary();
    computing_timer.reset();
}
