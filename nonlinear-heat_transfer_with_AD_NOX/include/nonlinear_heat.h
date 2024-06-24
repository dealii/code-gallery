/*-----------------------------------------------------------------------------
 * Created by Narasimhan Swaminathan on 20 Jun 2024.
 *-----------------------------------------------------------------------------
 */
#ifndef __MAIN_ALL_HEADER_H_INCLUDED__
#define __MAIN_ALL_HEADER_H_INCLUDED__
#include "allheaders.h"
/**
 * This is the main header of the programme with the definitions of all variables and functions
 */
class nonlinear_heat
{
public:
    /**
     * The constructor. The values of all the variables are defined in the nonlinear_heat_dons_des.cc
     */
    nonlinear_heat();
    ~nonlinear_heat();/*!< Destructor */
    /**
     * This function will run the main application
     */
    void run(); /*!< This function will be called to run the application */
    /**
     * Time step
     */
    const double delta_t;
    /**
     * Cracnk-Nicholson scheme parameter which is 0.5 in the current case
     */
    const double alpha;
    /**
     * Total time
     */
    const double tot_time;

    /**
     * The following three variables a, b and c, define the thermal conductivity as \f$k = a + bT + cT^{2}\f$.
     */
    const double a;
    const double b;
    const double c;
    /**
     * Specific heat
     */
    const double Cp;
    /**
     * Density
     */
    const double rho;
    /**
     * Time variable
     */
    double time;
private:
    /**
     * This function computes the jacobian by differentiating the residual. It takes in the point where the jacobian is to be evaluated.
     * @param evaluation_point The point where the jacobian is to be evaluated.
     */
    void compute_jacobian(const Vector<double> &evaluation_point);
    /**
     * This function computes the residual in a form that allows automatic differentiation (for the calculation of the Jacobian).
     * Then it allows it to be evaluated at the variable #evaluation_point.
     * @param evaluation_point The point where the residual is to be evaluated
     * @param residual The residual vector
     */
    void compute_residual(const Vector<double> &evaluation_point, Vector<double> & residual);
    /**
     * Sets up the system and initializes variables, sparsity etc.
     * @param time_step Time step of the problem. The value can be changed in the file nonlinear_heat_cons_des.cc
     */
    void setup_system(unsigned int time_step/** [in]  the time step*/);
    /**
     * Solves a linear system of equations.
     * @param rhs Right hand side vector
     * @param solution Solution
     * @param tolerance Tolerance
     */
    void solve(const Vector<double> &rhs, Vector<double> & solution, const double tolerance);
    /**
     * Outputs the results in the <code>vtu<code> format. Takes in the frequency with which we need to print the output.
     * @param prn Step number to print.
     */
    void output_results(unsigned int prn/** [in]  print number */) const;
    /**
     * Sets the actual boundary conditions of the problem, which could depend in #time.
     * @param time Actual time
     */
    void set_boundary_conditions(double time);

    Triangulation<2> triangulation;/*!<Triangulation to create the mesh*/
    DoFHandler<2> 	dof_handler; /*!< Attributes degrees of freedom to the mesh*/
    FESystem<2>       fe;/*!< Defines the finite element to be used */
    SparsityPattern     sparsity_pattern;/*!< Sparsity pattern*/
    SparseMatrix<double> system_matrix; /*Matrix holding the global Jacobian*/
    /**
     * A unique pointer for solving the linear system using the UMFPACK. See
     * Step-77.
     */
    std::unique_ptr<SparseDirectUMFPACK> matrix_factorization;
    /**
     * This variable, #converged_solution, contains the solution in the previous time step.
     * That is, the one that converged in the previous <b>time step<b>
     */
    Vector<double> converged_solution;/* Converged solution in the previous time step */
    /**
     * This variable, #present_solution, contains the solution during the non-linear iteration in the
     * current time step. That is, the one we want to converged to in the current time step.
   */
    Vector<double> present_solution;/* Converged solution in the previous time step */
};

/** A class to apply the initial condition.
 * The initial condition is used to simply ensure that the values of the concentrations are set everywhere to some value.
 * So we need to define a function within this class.
 */
class Initialcondition : public Function<2>
{
public:
    Initialcondition(): Function<2>(1)
   {}
   // Returns the intitial  values.
    virtual double value(const Point<2> &p,
                              const unsigned int component =0) const override;
};

/** A class to apply the boundary (Dirichlet) condition at the left edge.
 * This problem, also has the Newmann boundary condition at the right edge. This is directly
 * implemented in the compute_residual() and compute_jacobian() functions.
 */
class Boundary_values_left:public Function<2>
{
public:
    Boundary_values_left(): Function<2>(1)
    {}
    virtual double value(const Point<2> & p,const unsigned int component = 0) const override;
}; 
#endif //__MAIN_ALL_HEADER_H_INCLUDED__
