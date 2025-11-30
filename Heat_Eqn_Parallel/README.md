A parallel implementation of the transient heat equation

## Motivation
This program solves the transient heat equation in deal.II using MPI. A parallel solver for the transient heat equation is highly valuable in numerous fields where efficient, large-scale simulations of heat transfer are required. In engineering, it models heat dissipation in electronics, engines, and industrial systems, while in climate science, it simulates heat flow in the atmosphere and oceans. Geophysical applications use it to study heat transfer in the Earth's crust, and coupled systems like conjugate heat transfer in fluids or thermoelasticity in structures rely on it for accurate, high-resolution solutions. Parallel solvers are essential for handling fine meshes, small time steps, and real-time applications, leveraging modern high-performance computing hardware to reduce computation time.

With regards to MPI, the programs draws from step-40 of the deal.II tutorials where a parallel solver for Poisson's equation is described in detail. The heat equation is already solved in serial in step-26 of the deal.II tutorials. Here, we extend the same program and explore the use of parallel computing.

## Governing equations

The program implements the heat equation:
@f{align*}
    \frac{\partial u(\boldsymbol{x}, t)}{\partial t} - \Delta u(\boldsymbol{x}, t) &= f(\boldsymbol{x}, t), 
    && \forall \boldsymbol{x} \in \Omega, \ t \in (0, T), \\
    u(\boldsymbol{x}, 0) &= u_0(\boldsymbol{x}), 
    && \forall \boldsymbol{x} \in \Omega, \\
    u(\boldsymbol{x}, t) &= g(\boldsymbol{x}, t), 
    && \forall \boldsymbol{x} \in \partial \Omega, \ t \in (0, T).
@f}
Here, $u$ is the temperature and $t$ is the time.

## To run the Code
After running `cmake .`, run `make release` or `make debug` to switch between `release` and `debug` mode. Compile using `make`.
Run the executable by using `make run` on the command line.
Run the executable on 'n' processes using 'mpirun -np $n$ ./MPI_heat_eqn'. For example, 'mpirun -np 40 ./MPI_heat_eqn' runs the program on 40 processes.

## Validation and scaling
An animation comparing results of temperature evolution from the current code with results of step-26 can be found [here](./doc/comparison_MPI_vs_step_26.mp4) (left: serial code (step-26); right: current parallel implementation (80 procs)). We also perform a scaling study running the program for just one time step with around 50 M cells. The results can be found [here](./doc/scaling_study.png)
