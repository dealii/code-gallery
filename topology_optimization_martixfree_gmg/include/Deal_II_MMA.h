#ifndef DEAL_II_MMA_SOLVER
#define DEAL_II_MMA_SOLVER
#include <deal.II/lac/generic_linear_algebra.h>
#include <type_traits>

namespace deal_ii_mma_detail
{
    // PETSc && Trilinos Vector
    template <typename V>
    struct needs_compress : std::true_type {};

    // deal.II parallel vector
    template <typename Number>
    struct needs_compress<dealii::LinearAlgebra::distributed::Vector<Number>>
        : std::false_type {};

    template <typename V>
    inline void finalize_vector_write(V &x)
    {
        if constexpr (needs_compress<V>::value)
            x.compress(dealii::VectorOperation::insert);
    }
}

using dealii::Vector;
using dealii::IndexSet;
using dealii::FullMatrix;
using dealii::Utilities::MPI::sum;

template <typename MMAVectorType>
class Deal_II_MMA_Solver
{
public:
	Deal_II_MMA_Solver();
	// Initialize MMA
	void Deal_II_MMA_Init(unsigned int Constraints,
						  double A_i,
						  double C_i,
						  const MMAVectorType &MMA_Design_Variables);
	// Solve with the interior-point method
	void Deal_II_MMA_Dual_Problem_Solve(MMAVectorType &MMA_Design_Variables,
										const MMAVectorType  &MMA_Object_Function_Diff_Values,
										const Vector<double> &MMA_Constraints_Function_Values,
										const std::vector<MMAVectorType> &MMA_Constraints_Function_Diff_Values,
										const MMAVectorType &MMA_Xmax,
										const MMAVectorType &MMA_Xmin);
	// Reinitialize after the mesh has changed
	// Connect Optimization Loop In Mesh Adapt 
	void Deal_II_MMA_Reinit(const MMAVectorType &MMA_Design_Variables);
	// Set the asymptote update parameters
	void Deal_II_MMA_SetAsymptotes(double init, double decrease, double increase);
	// Set whether to use the more robust MMA asymptote update
	void Deal_II_MMA_SetRobustAsymptotesType(bool robust_type);
	// Set whether the constraint functions are also convex-approximated
	void Deal_II_MMA_SetConstraintConvexApproximationType(bool constraint_type);
	// Get the change between the current and the previous design vector
	double Deal_II_MMA_Get_Change(const MMAVectorType &MMA_Design_Variables);
	// Current iteration counter, supplied externally and not tracked internally
	int Loop_Iter;
	// Moving upper and lower bounds during MMA optimization
	// xmin<Low<beta<x<alpha<Upp<xmax
	MMAVectorType Moving_Low_Boundary, Moving_Upp_Boundary;
	// Design variables from the two previous MMA iterations
	std::vector<MMAVectorType> Old_Design_Variables;
private:
	// Build the approximate subproblem from the supplied
	// objective and constraint derivatives
	void Generate_SubConvexProblem(const MMAVectorType &MMA_Design_Variables,
								   const MMAVectorType &MMA_Object_Function_Diff_Values,
								   const Vector<double> &MMA_Constraints_Function_Values,
								   const std::vector<MMAVectorType> &MMA_Constraints_Function_Diff_Values,
								   const MMAVectorType &MMA_Xmax,
								   const MMAVectorType &MMA_Xmin);
	// Interior-point solver
	void Interior_Point_Solve(MMAVectorType &MMA_Design_Variables);
	// Compute the dual gradient and Hessian
	void Get_Dual_Gradient(const MMAVectorType &MMA_Design_Variables);
	void Get_Dual_Hess(const MMAVectorType &MMA_Design_Variables);
	// Compute X, Y and Z from the dual variable Lambda
	void Generate_XYZ_From_Lambda(MMAVectorType &MMA_Design_Variables);
	// Backtracking line search on the step length once the search direction is known
	void Dual_Linear_Search();
	// Solve the dual linear system
	void Solve_Linear_System();
	// Compute the dual residual
	double Get_Dual_Residual(const MMAVectorType &MMA_Design_Variables, double Epsi);
	// Number of constraints
	unsigned int Number_Of_Constraints;
	// Primal-dual residual
	double Primal_Dual_Epsi;
	// Minimum difference between the upper (xmax) and lower (xmin) bounds during iterations
	double Xmax_Xmin_Epsi;
	// Control parameter for the approximate convex subproblem
	// used to tune how closely the subproblem matches the original problem
	// constant in MMA, adaptive in GCMMA
	double Rho_Epsi;
	// Control parameter for the change in x per iteration
	// this is a fraction of (xmax - xmin), not an absolute step size
	double Design_Change_Move;
	// Parameter used to compute the upper bound alpha and lower bound beta
	double Alpha_Beta;
	// Parameters controlling asymptote relaxation and contraction during MMA iterations
	// mainly affecting iteration count and robustness
	double Asym_Init, Asym_Decrease, Asym_Increase;
	// Index set of DOFs owned by this rank, used to loop over the design variables
	IndexSet MMA_Dofs_IndexSet;
	// Subproblem construction parameters
	// in this code the objective-subproblem
	// parameters a0 and d0 are both 1
	Vector<double> SubProblem_A, SubProblem_B, SubProblem_C;
	// Subproblem variables
	Vector<double> SubProblem_Y;
	double SubProblem_Z;

	// Dual variables
	Vector<double> DualProblem_Lambda, DualProblem_Mu;
	// Slack variables introduced to convert
	// inequality constraints into equality constraints
	Vector<double> Slack_Variables;

	// Moving upper and lower bounds during MMA optimization
	// xmin<Low<beta<x<alpha<Upp<xmax
	MMAVectorType Moving_Alpha_Boundary, Moving_Beta_Boundary;

	// Gradient information of the objective function used to build the MMA subproblem
	MMAVectorType Object_Function_Positive, Object_Function_Negative;
	std::vector<MMAVectorType> Constraint_Function_Positive, Constraint_Function_Negative;

	// Gradient and Hessian required by the interior-point solver
	Vector<double> Object_Function_Gradient;
	FullMatrix<double> Object_Function_Hess;
	// Whether the constraint functions are also convex-approximated
	bool Constraint_Convex_Approximation;
	// Whether to use the strictly robust MMA asymptote update
	bool RobustAsymptotesType;
};

template <typename MMAVectorType>
Deal_II_MMA_Solver<MMAVectorType>::
    Deal_II_MMA_Solver()
:   Primal_Dual_Epsi(1e-9),
    Xmax_Xmin_Epsi(1e-5),
    Rho_Epsi(5e-5),
    Design_Change_Move(0.5),
    Alpha_Beta(0.1),
    Asym_Init(0.5),
    Asym_Decrease(0.7),
    Asym_Increase(1.2),
    Constraint_Convex_Approximation(false),
    RobustAsymptotesType(false)
{};
/////////////////////////////////////////////////////////////////////////////////////
template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Deal_II_MMA_SetAsymptotes(double init, double decrease, double increase)
{
    Asym_Init = init;
    Asym_Increase = increase;
    Asym_Decrease = decrease;
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Deal_II_MMA_SetRobustAsymptotesType(bool robust_type)
{
    RobustAsymptotesType = robust_type;
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Deal_II_MMA_SetConstraintConvexApproximationType(bool constraint_type)
{
    Constraint_Convex_Approximation = constraint_type;
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Deal_II_MMA_Init(unsigned int Constraints,
                     double A_i,
                     double C_i,
                     const MMAVectorType &MMA_Design_Variables)
{
    MMA_Dofs_IndexSet = MMA_Design_Variables.locally_owned_elements();
    //
    Number_Of_Constraints = Constraints;
    // 
    SubProblem_A.reinit(Constraints);
    SubProblem_A = A_i;
    SubProblem_B.reinit(Constraints);
    SubProblem_C.reinit(Constraints);
    SubProblem_C = C_i;
    // 
    SubProblem_Y.reinit(Constraints);
    // 
    DualProblem_Lambda.reinit(Constraints);
    DualProblem_Mu.reinit(Constraints);
    // 
    Slack_Variables.reinit(2 * Constraints);
    // Low,Upp,Alpha,Beta
    Moving_Low_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Upp_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Alpha_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Beta_Boundary.reinit(MMA_Design_Variables, true);
    //
    Object_Function_Positive.reinit(MMA_Design_Variables, true);
    Object_Function_Negative.reinit(MMA_Design_Variables, true);
    Constraint_Function_Positive.resize(Constraints);
    Constraint_Function_Negative.resize(Constraints);
    for (unsigned int i = 0; i < Constraints; i++)
    {
        Constraint_Function_Positive[i].reinit(MMA_Design_Variables, true);
        Constraint_Function_Negative[i].reinit(MMA_Design_Variables, true);
    }
    //
    Old_Design_Variables.resize(2);
    for (unsigned int i = 0; i < 2; i++)
    {
        Old_Design_Variables[i].reinit(MMA_Design_Variables, true);
    }
    //
    Object_Function_Gradient.reinit(Constraints);
    Object_Function_Hess.reinit(Constraints,Constraints);
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Deal_II_MMA_Reinit(const MMAVectorType &MMA_Design_Variables)
{
    //
    MMA_Dofs_IndexSet = MMA_Design_Variables.locally_owned_elements();
    // Low,Upp,Alpha,Beta
    Moving_Low_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Upp_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Alpha_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Beta_Boundary.reinit(MMA_Design_Variables, true);
    //
    Object_Function_Positive.reinit(MMA_Design_Variables, true);
    Object_Function_Negative.reinit(MMA_Design_Variables, true);
    Constraint_Function_Positive.resize(Number_Of_Constraints);
    Constraint_Function_Negative.resize(Number_Of_Constraints);
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        Constraint_Function_Positive[i].reinit(MMA_Design_Variables, true);
        Constraint_Function_Negative[i].reinit(MMA_Design_Variables, true);
    }
    //
    Old_Design_Variables.resize(2);
    for (unsigned int i = 0; i < 2; i++)
    {
        Old_Design_Variables[i].reinit(MMA_Design_Variables, true);
    }
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Deal_II_MMA_Dual_Problem_Solve(MMAVectorType &MMA_Design_Variables,
                                   const MMAVectorType  &MMA_Object_Function_Diff_Values,
                                   const Vector<double> &MMA_Constraints_Function_Values,
                                   const std::vector<MMAVectorType> &MMA_Constraints_Function_Diff_Values,
                                   const MMAVectorType &MMA_Xmax,
                                   const MMAVectorType &MMA_Xmin)
{
    Generate_SubConvexProblem(MMA_Design_Variables,
                              MMA_Object_Function_Diff_Values,
                              MMA_Constraints_Function_Values,
                              MMA_Constraints_Function_Diff_Values,
                              MMA_Xmax,
                              MMA_Xmin);
    std::swap(Old_Design_Variables[0], Old_Design_Variables[1]);
    Old_Design_Variables[0] = MMA_Design_Variables;
    Interior_Point_Solve(MMA_Design_Variables);
}

/////////////////////////////////////////////////////////////////////////////////////
template <typename MMAVectorType>
double Deal_II_MMA_Solver<MMAVectorType>::
    Deal_II_MMA_Get_Change(const MMAVectorType &MMA_Design_Variables)
{
    double change = 0.;
    for (unsigned int i : MMA_Dofs_IndexSet)
    {
        change += std::abs(MMA_Design_Variables[i] - Old_Design_Variables[0][i]);
    }
    change = sum(change, MMA_Design_Variables.get_mpi_communicator());
    return change;
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Generate_SubConvexProblem(const MMAVectorType &MMA_Design_Variables,
                              const MMAVectorType &MMA_Object_Function_Diff_Values,
                              const Vector<double> &MMA_Constraints_Function_Values,
                              const std::vector<MMAVectorType> &MMA_Constraints_Function_Diff_Values,
                              const MMAVectorType &MMA_Xmax,
                              const MMAVectorType &MMA_Xmin)
{
    if (Loop_Iter < 3)
    {
        for (unsigned int i : MMA_Dofs_IndexSet)
        {
            Moving_Low_Boundary[i] = MMA_Design_Variables[i] - Asym_Init * (MMA_Xmax[i] - MMA_Xmin[i]);
            Moving_Upp_Boundary[i] = MMA_Design_Variables[i] + Asym_Init * (MMA_Xmax[i] - MMA_Xmin[i]);
        }
    }
    else
    {
        double Loop_Factor = 0;
        double Asym_Factor = 0;
        double Xmax_Xmin_Epsi_Temp = 0;
        double Xmax_Temp = 0;
        double Xmin_Temp = 0;
        for (unsigned int i : MMA_Dofs_IndexSet)
        {
            // (x-x_old1)*(x_old1-x_old2)
            Loop_Factor = (MMA_Design_Variables[i] - Old_Design_Variables[0][i]) *
                          (Old_Design_Variables[0][i] - Old_Design_Variables[1][i]);
            if (Loop_Factor < 0.)
            {
                Asym_Factor = Asym_Decrease;
            }
            else if (Loop_Factor > 0.)
            {
                Asym_Factor = Asym_Increase;
            }
            else
            {
                Asym_Factor = 1.;
            }
            Moving_Low_Boundary[i] = MMA_Design_Variables[i] - Asym_Factor *
                                                                   (Old_Design_Variables[0][i] - Moving_Low_Boundary[i]);
            Moving_Upp_Boundary[i] = MMA_Design_Variables[i] + Asym_Factor *
                                                                   (Moving_Upp_Boundary[i] - Old_Design_Variables[0][i]);
            Xmax_Xmin_Epsi_Temp = std::max(Xmax_Xmin_Epsi, MMA_Xmax[i] - MMA_Xmin[i]);
            if (RobustAsymptotesType)
            {
                Moving_Low_Boundary[i] = std::max(Moving_Low_Boundary[i], MMA_Design_Variables[i] - 100. * Xmax_Xmin_Epsi_Temp);
                Moving_Low_Boundary[i] = std::min(Moving_Low_Boundary[i], MMA_Design_Variables[i] - 1.e-4 * Xmax_Xmin_Epsi_Temp);
                Moving_Upp_Boundary[i] = std::max(Moving_Upp_Boundary[i], MMA_Design_Variables[i] + 1.e-4 * Xmax_Xmin_Epsi_Temp);
                Moving_Upp_Boundary[i] = std::min(Moving_Upp_Boundary[i], MMA_Design_Variables[i] + 100. * Xmax_Xmin_Epsi_Temp);

                Xmax_Temp = MMA_Xmax[i] + 1.e-5;
                Xmin_Temp = MMA_Xmin[i] - 1.e-5;
                if (MMA_Design_Variables[i] < Xmin_Temp)
                {
                    Moving_Low_Boundary[i] = MMA_Design_Variables[i] - (Xmax_Temp - MMA_Design_Variables[i]) / 0.9;
                    Moving_Upp_Boundary[i] = MMA_Design_Variables[i] + (Xmax_Temp - MMA_Design_Variables[i]) / 0.9;
                }
                if (MMA_Design_Variables[i] > Xmax_Temp)
                {
                    Moving_Low_Boundary[i] = MMA_Design_Variables[i] - (MMA_Design_Variables[i] - Xmin_Temp) / 0.9;
                    Moving_Upp_Boundary[i] = MMA_Design_Variables[i] + (MMA_Design_Variables[i] - Xmin_Temp) / 0.9;
                }                
            }else
            {
                Moving_Low_Boundary[i] = std::max(Moving_Low_Boundary[i], MMA_Design_Variables[i] - 10. * Xmax_Xmin_Epsi_Temp);
                Moving_Low_Boundary[i] = std::min(Moving_Low_Boundary[i], MMA_Design_Variables[i] - 1.e-2 * Xmax_Xmin_Epsi_Temp);
                Moving_Upp_Boundary[i] = std::max(Moving_Upp_Boundary[i], MMA_Design_Variables[i] + 1.e-2 * Xmax_Xmin_Epsi_Temp);
                Moving_Upp_Boundary[i] = std::min(Moving_Upp_Boundary[i], MMA_Design_Variables[i] + 10. * Xmax_Xmin_Epsi_Temp);
            }
        }
    }
    double Object_Function_Diff_Value_Positive_Temp = 0;
    double Object_Function_Diff_Value_Negative_Temp = 0;
    double Constraint_Function_Diff_Value_Positive_Temp = 0;
    double Constraint_Function_Diff_Value_Negative_Temp = 0;
    // 
    double Convex_Change_Part_Temp = 0;
    double Xmax_Xmin_Epsi_Inv = 0;
    for (unsigned int i : MMA_Dofs_IndexSet)
    {
        // Alpha
        Moving_Alpha_Boundary[i] = std::max(MMA_Xmin[i], Moving_Low_Boundary[i] + Alpha_Beta * (MMA_Design_Variables[i] - Moving_Low_Boundary[i]));
        Moving_Alpha_Boundary[i] = std::max(Moving_Alpha_Boundary[i], MMA_Design_Variables[i] - Design_Change_Move * (MMA_Xmax[i] - MMA_Xmin[i]));
        Moving_Alpha_Boundary[i] = std::min(Moving_Alpha_Boundary[i], MMA_Xmax[i]);
        // Beta
        Moving_Beta_Boundary[i] = std::min(MMA_Xmax[i], Moving_Upp_Boundary[i] - Alpha_Beta * (Moving_Upp_Boundary[i] - MMA_Design_Variables[i]));
        Moving_Beta_Boundary[i] = std::min(Moving_Beta_Boundary[i], MMA_Design_Variables[i] + Design_Change_Move * (MMA_Xmax[i] - MMA_Xmin[i]));
        Moving_Beta_Boundary[i] = std::max(Moving_Beta_Boundary[i], MMA_Xmin[i]);
        // 
        Xmax_Xmin_Epsi_Inv = 1. / std::max(Xmax_Xmin_Epsi, MMA_Xmax[i] - MMA_Xmin[i]);
        // Object Function
        Object_Function_Diff_Value_Positive_Temp = std::max(0., MMA_Object_Function_Diff_Values[i]);
        Object_Function_Diff_Value_Negative_Temp = std::max(0., -1. * MMA_Object_Function_Diff_Values[i]);
        Convex_Change_Part_Temp = 1e-3 * std::abs(MMA_Object_Function_Diff_Values[i]) + Rho_Epsi * Xmax_Xmin_Epsi_Inv;
        // P0_ij
        Object_Function_Positive[i] = std::pow(Moving_Upp_Boundary[i] - MMA_Design_Variables[i], 2.) *
                                      (Object_Function_Diff_Value_Positive_Temp + Convex_Change_Part_Temp);
        // Q0_ij
        Object_Function_Negative[i] = std::pow(MMA_Design_Variables[i] - Moving_Low_Boundary[i], 2.) *
                                      (Object_Function_Diff_Value_Negative_Temp + Convex_Change_Part_Temp);
        // Constraint Function
        for (unsigned int j = 0; j < Number_Of_Constraints; j++)
        {
            Constraint_Function_Diff_Value_Positive_Temp = std::max(0., MMA_Constraints_Function_Diff_Values[j][i]);
            Constraint_Function_Diff_Value_Negative_Temp = std::max(0., -1. * MMA_Constraints_Function_Diff_Values[j][i]);
            Convex_Change_Part_Temp = 1e-3 * std::abs(MMA_Constraints_Function_Diff_Values[j][i]) + Rho_Epsi * Xmax_Xmin_Epsi_Inv;
            if (Constraint_Convex_Approximation)
            {
                // P_ij
                Constraint_Function_Positive[j][i] = std::pow(Moving_Upp_Boundary[i] - MMA_Design_Variables[i], 2.) *
                                                    (Constraint_Function_Diff_Value_Positive_Temp + Convex_Change_Part_Temp);
                // Q_ij
                Constraint_Function_Negative[j][i] = std::pow(MMA_Design_Variables[i] - Moving_Low_Boundary[i], 2.) *
                                                    (Constraint_Function_Diff_Value_Negative_Temp + Convex_Change_Part_Temp);             
            }else{
                // P_ij
                Constraint_Function_Positive[j][i] = std::pow(Moving_Upp_Boundary[i] - MMA_Design_Variables[i], 2.) *
                                                    (Constraint_Function_Diff_Value_Positive_Temp);
                // Q_ij
                Constraint_Function_Negative[j][i] = std::pow(MMA_Design_Variables[i] - Moving_Low_Boundary[i], 2.) *
                                                    (Constraint_Function_Diff_Value_Negative_Temp);
            }
        }
    }
    // 
    for (unsigned int j = 0; j < Number_Of_Constraints; j++)
    {
        SubProblem_B[j] = 0.;
        for (unsigned int i : MMA_Dofs_IndexSet)
        {
            SubProblem_B[j] += Constraint_Function_Positive[j][i] / (Moving_Upp_Boundary[i] - MMA_Design_Variables[i]) +
                               Constraint_Function_Negative[j][i] / (MMA_Design_Variables[i] - Moving_Low_Boundary[i]);
        }
        SubProblem_B[j] = sum(SubProblem_B[j], MMA_Design_Variables.get_mpi_communicator());
    }
    SubProblem_B.operator-=(MMA_Constraints_Function_Values);
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Generate_XYZ_From_Lambda(MMAVectorType &MMA_Design_Variables)
{
    double Lambda_Mutiply_A = 0;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        if (DualProblem_Lambda[i] < 0.)
        {
            DualProblem_Lambda[i] = 0.;
        }
        SubProblem_Y[i] = std::max(0., DualProblem_Lambda[i] - SubProblem_C[i]);
        Lambda_Mutiply_A += DualProblem_Lambda[i] * SubProblem_A[i];
    }
    SubProblem_Z = std::max(0., 10. * (Lambda_Mutiply_A - 1.));

    double Object_Function_Positive_Mutiply_Lambda = 0;
    double Object_Function_Negative_Mutiply_Lambda = 0;
    for (unsigned int i : MMA_Dofs_IndexSet)
    {
        Object_Function_Positive_Mutiply_Lambda = Object_Function_Positive[i];
        Object_Function_Negative_Mutiply_Lambda = Object_Function_Negative[i];
        for (unsigned int j = 0; j < Number_Of_Constraints; j++)
        {
            Object_Function_Positive_Mutiply_Lambda += Constraint_Function_Positive[j][i] * DualProblem_Lambda[j];
            Object_Function_Negative_Mutiply_Lambda += Constraint_Function_Negative[j][i] * DualProblem_Lambda[j];
        }
        MMA_Design_Variables[i] = (std::sqrt(Object_Function_Positive_Mutiply_Lambda) * Moving_Low_Boundary[i] +
                                   std::sqrt(Object_Function_Negative_Mutiply_Lambda) * Moving_Upp_Boundary[i]) /
                                  (std::sqrt(Object_Function_Positive_Mutiply_Lambda) + std::sqrt(Object_Function_Negative_Mutiply_Lambda));
        if (MMA_Design_Variables[i] < Moving_Alpha_Boundary[i])
        {
            MMA_Design_Variables[i] = Moving_Alpha_Boundary[i];
        }
        if (MMA_Design_Variables[i] > Moving_Beta_Boundary[i])
        {
            MMA_Design_Variables[i] = Moving_Beta_Boundary[i];
        }
    }
    deal_ii_mma_detail::finalize_vector_write(MMA_Design_Variables);
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Get_Dual_Gradient(const MMAVectorType &MMA_Design_Variables)
{
    for (unsigned int j = 0; j < Number_Of_Constraints; j++)
    {
        Object_Function_Gradient[j] = 0.;
        for (unsigned int i : MMA_Dofs_IndexSet)
        {
            Object_Function_Gradient[j] += Constraint_Function_Positive[j][i] / (Moving_Upp_Boundary[i] - MMA_Design_Variables[i]) +
                                           Constraint_Function_Negative[j][i] / (MMA_Design_Variables[i] - Moving_Low_Boundary[i]);
        }
        Object_Function_Gradient[j] = sum(Object_Function_Gradient[j], MMA_Design_Variables.get_mpi_communicator());
        Object_Function_Gradient[j] -= SubProblem_B[j] + SubProblem_A[j] * SubProblem_Z + SubProblem_Y[j];
    }
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Get_Dual_Hess(const MMAVectorType &MMA_Design_Variables)
{
    double Object_Function_Positive_Mutiply_Lambda = 0;
    double Object_Function_Negative_Mutiply_Lambda = 0;
    double Object_Function_Diff_2_Values = 0;
    double Zero_Point_Of_Diff_Function = 0;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        for (unsigned int j = 0; j < Number_Of_Constraints; j++)
        {
            Object_Function_Hess[i][j] = 0.;
            for (unsigned int Index_i : MMA_Dofs_IndexSet)
            {
                Object_Function_Positive_Mutiply_Lambda = Object_Function_Positive[Index_i];
                Object_Function_Negative_Mutiply_Lambda = Object_Function_Negative[Index_i];
                for (unsigned int Index_ii = 0; Index_ii < Number_Of_Constraints; Index_ii++)
                {
                    Object_Function_Positive_Mutiply_Lambda += Constraint_Function_Positive[Index_ii][Index_i] * DualProblem_Lambda[Index_ii];
                    Object_Function_Negative_Mutiply_Lambda += Constraint_Function_Negative[Index_ii][Index_i] * DualProblem_Lambda[Index_ii];
                }
                Zero_Point_Of_Diff_Function = (std::sqrt(Object_Function_Positive_Mutiply_Lambda) * Moving_Low_Boundary[Index_i] +
                                               std::sqrt(Object_Function_Negative_Mutiply_Lambda) * Moving_Upp_Boundary[Index_i]) /
                                              (std::sqrt(Object_Function_Positive_Mutiply_Lambda) + std::sqrt(Object_Function_Negative_Mutiply_Lambda));
                Object_Function_Diff_2_Values = -1. / (2. * Object_Function_Positive_Mutiply_Lambda /
                                                           std::pow(Moving_Upp_Boundary[Index_i] - MMA_Design_Variables[Index_i], 3.) +
                                                       2. * Object_Function_Negative_Mutiply_Lambda /
                                                           std::pow(MMA_Design_Variables[Index_i] - Moving_Low_Boundary[Index_i], 3.));
                if (Zero_Point_Of_Diff_Function < Moving_Alpha_Boundary[Index_i])
                {
                    Object_Function_Diff_2_Values = 0.;
                }
                if (Zero_Point_Of_Diff_Function > Moving_Beta_Boundary[Index_i])
                {
                    Object_Function_Diff_2_Values = 0.;
                }
                Object_Function_Hess[i][j] +=
                    (Constraint_Function_Positive[i][Index_i] / std::pow(Moving_Upp_Boundary[Index_i] - MMA_Design_Variables[Index_i], 2.) -
                     Constraint_Function_Negative[i][Index_i] / std::pow(MMA_Design_Variables[Index_i] - Moving_Low_Boundary[Index_i], 2.)) *
                    Object_Function_Diff_2_Values *
                    (Constraint_Function_Positive[j][Index_i] / std::pow(Moving_Upp_Boundary[Index_i] - MMA_Design_Variables[Index_i], 2.) -
                     Constraint_Function_Negative[j][Index_i] / std::pow(MMA_Design_Variables[Index_i] - Moving_Low_Boundary[Index_i], 2.));
            }
            Object_Function_Hess[i][j] = sum(Object_Function_Hess[i][j], MMA_Design_Variables.get_mpi_communicator());
        }
    }
    
    double Lambda_Mutiply_A = 0.;
    for (unsigned int j = 0; j < Number_Of_Constraints; j++)
    {
        if (DualProblem_Lambda[j] < 0.)
        {
            DualProblem_Lambda[j] = 0.;
        }
        Lambda_Mutiply_A += DualProblem_Lambda[j] * SubProblem_A[j];
        if (DualProblem_Lambda[j] > SubProblem_C[j])
        {
            Object_Function_Hess[j][j] += -1.;
        }
        Object_Function_Hess[j][j] += -DualProblem_Mu[j] / DualProblem_Lambda[j];
    }

    if (Lambda_Mutiply_A > 0.)
    {
        for (unsigned int i = 0; i < Number_Of_Constraints; i++)
        {
            for (unsigned int j = 0; j < Number_Of_Constraints; j++)
            {
                Object_Function_Hess[i][j] += -10. * SubProblem_A[i] * SubProblem_A[j];
            }
        }
    }

    double HessCorr = 1e-4 * Object_Function_Hess.trace() / Number_Of_Constraints;
    if (-1. * HessCorr < 1.e-7)
    {
        HessCorr = -1.e-7;
    }
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        Object_Function_Hess[i][i] += HessCorr;
    }
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Dual_Linear_Search()
{
    double Theta = 1.005;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        if (Theta < -1.01 * Slack_Variables[i] / DualProblem_Lambda[i])
        {
            Theta = -1.01 * Slack_Variables[i] / DualProblem_Lambda[i];
        }
        if (Theta < -1.01 * Slack_Variables[i + Number_Of_Constraints] / DualProblem_Mu[i])
        {
            Theta = -1.01 * Slack_Variables[i + Number_Of_Constraints] / DualProblem_Mu[i];
        }
    }
    Theta = 1. / Theta;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        DualProblem_Lambda[i] = DualProblem_Lambda[i] + Theta * Slack_Variables[i];
        DualProblem_Mu[i] = DualProblem_Mu[i] + Theta * Slack_Variables[i + Number_Of_Constraints];
    }
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Solve_Linear_System()
{
    if (Number_Of_Constraints > 1)
    {
        double  Max_Row_Number = 0;
        unsigned int Max_Row_Index = 0;
        double Swap_Temp = 0;
        for (unsigned int col_index = 0; col_index < Number_Of_Constraints; col_index++)
        {
            Max_Row_Number = 0;
            Max_Row_Index = 0;
            for (unsigned int row_index = col_index; row_index < Number_Of_Constraints; row_index++)
            {
                if (abs(Object_Function_Hess(row_index,col_index)) > Max_Row_Number)
                {
                    Max_Row_Number = abs(Object_Function_Hess(row_index,col_index));
                    Max_Row_Index = row_index;
                }                      
            }
            if (Max_Row_Index != col_index)
            {
                for (unsigned int row_index = 0; row_index < Number_Of_Constraints; row_index++)
                {
                    Swap_Temp = Object_Function_Hess(Max_Row_Index,row_index);
                    Object_Function_Hess(Max_Row_Index,row_index) = Object_Function_Hess(col_index,row_index);
                    Object_Function_Hess(col_index,row_index) = Swap_Temp;
                }
                Swap_Temp = Object_Function_Gradient[Max_Row_Index];
                Object_Function_Gradient[Max_Row_Index] = Object_Function_Gradient[col_index];
                Object_Function_Gradient[col_index] = Swap_Temp;
            }
            for (unsigned int row_index = col_index + 1; row_index < Number_Of_Constraints; row_index++)
            {
                Swap_Temp = -Object_Function_Hess(row_index,col_index)/Object_Function_Hess(col_index,col_index);
                for (unsigned int col_tmp = 0; col_tmp < Number_Of_Constraints; col_tmp++)
                {
                    Object_Function_Hess(row_index,col_tmp) += Object_Function_Hess(col_index,col_tmp) * Swap_Temp;
                }
                Object_Function_Gradient[row_index] += Object_Function_Gradient[col_index] * Swap_Temp;
            }   
        }
        for (int i = Number_Of_Constraints - 1; i >= 0; i --)
        {
            // 
            if (i == (int)Number_Of_Constraints - 1)
            {
                Slack_Variables[i] = Object_Function_Gradient[i]/Object_Function_Hess(i,i);                
            }
            // 
            Slack_Variables[i] = Object_Function_Gradient[i];
            for (unsigned int j = i + 1; j < Number_Of_Constraints; j++)
            {
                Slack_Variables[i] -= Slack_Variables[j] * Object_Function_Hess(i,j);
            }
            Slack_Variables[i] /= Object_Function_Hess(i,i);
        }
    }
    else if (Number_Of_Constraints > 0)
    {
        Slack_Variables[0] = Object_Function_Gradient[0] / Object_Function_Hess[0][0];
    }
}

template <typename MMAVectorType>
double Deal_II_MMA_Solver<MMAVectorType>::
    Get_Dual_Residual(const MMAVectorType &MMA_Design_Variables,
                      double Epsi)
{
    double Residual = 0.;
    double Error = 0.;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        Residual = 0.;
        for (unsigned int j : MMA_Dofs_IndexSet)
        {
            Residual += Constraint_Function_Positive[i][j] / (Moving_Upp_Boundary[j] - MMA_Design_Variables[j]) +
                        Constraint_Function_Negative[i][j] / (MMA_Design_Variables[j] - Moving_Low_Boundary[j]);
        }
        Residual = sum(Residual, MMA_Design_Variables.get_mpi_communicator());
        Residual -= SubProblem_B[i] + SubProblem_A[i] * SubProblem_Z + SubProblem_Y[i] - DualProblem_Mu[i];
        if (std::abs(Residual) > std::abs(Error))
        {
            Error = std::abs(Residual);
        }
    }
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        Residual = 0.;
        Residual += DualProblem_Mu[i] * DualProblem_Lambda[i] - Epsi;
        if (std::abs(Residual) > std::abs(Error))
        {
            Error = std::abs(Residual);
        }
    }
    return Error;
}

template <typename MMAVectorType>
void Deal_II_MMA_Solver<MMAVectorType>::
    Interior_Point_Solve(MMAVectorType &MMA_Design_Variables)
{
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        DualProblem_Lambda[i] = SubProblem_C[i] / 2.;
        DualProblem_Mu[i] = 1.;
    }
    double Tolerance = Primal_Dual_Epsi;
    double Epsi_Init = 1.;
    double Error_Init = 1.;
    unsigned int Loop_Couter = 0;
    while (Epsi_Init > Tolerance)
    {
        Loop_Couter = 0;
        while (Error_Init > 0.9 * Epsi_Init && Loop_Couter < 100)
        {
            Loop_Couter++;
            Generate_XYZ_From_Lambda(MMA_Design_Variables);
            Get_Dual_Gradient(MMA_Design_Variables);
            for (unsigned int i = 0; i < Number_Of_Constraints; i++)
            {
                Object_Function_Gradient[i] = -1. * Object_Function_Gradient[i] - Epsi_Init / DualProblem_Lambda[i];
            }
            Get_Dual_Hess(MMA_Design_Variables);
            Solve_Linear_System();
            for (unsigned int i = 0; i < Number_Of_Constraints; i++)
            {
                Slack_Variables[i + Number_Of_Constraints] = -DualProblem_Mu[i] + Epsi_Init / DualProblem_Lambda[i] - Slack_Variables[i] * DualProblem_Mu[i] / DualProblem_Lambda[i];
            }
            Dual_Linear_Search();
            Generate_XYZ_From_Lambda(MMA_Design_Variables);
            Error_Init = Get_Dual_Residual(MMA_Design_Variables, Epsi_Init);
        }
        Epsi_Init = Epsi_Init * 0.1;
    }
}
#endif