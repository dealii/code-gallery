/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2015 by the deal.II authors 
 *  							and Salvador Flores.
 *                      
 *
 * 
 * This is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Salvador Flores, 
 *         Center for Mathematical Modelling,
 *		   Universidad de Chile, 2015.
 */
 
 
 /* 
  This piece of software solves the elliptic p-laplacian 
  boundary-value problems:
  
    Min {∫ 1/2 W(|Du|²)+ 1/p |Du|^p -fu : u=g on ∂S }   (1)
     u
 
  for large values of p, which approximates (see Alvarez & Flores 2015)
  
    Min {∫ 1/2 W(|Du|²) -fu : |Du|<1 a.s. on S, u=g on ∂S }   
     u

  By default W(t)=t and S=unit disk.
  
  Large portions of this code are borrowed from the deal.ii tutorials 
  
		     step-15, step-29. 
 	
  For further details see the technical report available at
  the documentation andmof	 at http://www.dim.uchile.cl/~sflores.
   
 */

// Include files 

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <typeinfo>
#include <fstream>
#include <iostream>

#include <deal.II/numerics/solution_transfer.h>

// Open a namespace for this program and import everything from the
// dealii namespace into it.
namespace nsp
{
  using namespace dealii;

// ********************************************************//
 class ParameterReader : public Subscriptor
{
public:
ParameterReader(ParameterHandler &);
void read_parameters(const std::string);
private:
void declare_parameters();
ParameterHandler &prm;
};
// Constructor
ParameterReader::ParameterReader(ParameterHandler &paramhandler):
prm(paramhandler)
{}

 void ParameterReader::declare_parameters()
{

prm.enter_subsection ("Global Parameters");
{
	prm.declare_entry("p", "100",Patterns::Double(2.1),
		"Penalization parameter");
	prm.declare_entry("known_solution", "true",Patterns::Bool(),
		"Whether the exact solution is known");
}
prm.leave_subsection ();

prm.enter_subsection ("Mesh & Refinement Parameters");
{
	prm.declare_entry("Code for the domain", "0",Patterns::Integer(0,2),
		"Number identifying the domain in which we solve the problem");
	prm.declare_entry("No of initial refinements", "4",Patterns::Integer(0),
		"Number of global mesh refinement steps applied to initial coarse grid");
	prm.declare_entry("No of adaptive refinements", "8",Patterns::Integer(0),
		"Number of global adaptive mesh refinements");
	prm.declare_entry("top_fraction_of_cells", "0.25",Patterns::Double(0),
		"refinement threshold");
	prm.declare_entry("bottom_fraction_of_cells", "0.05",Patterns::Double(0),
		"coarsening threshold");
}
prm.leave_subsection ();


prm.enter_subsection ("Algorithm Parameters");
{
	prm.declare_entry("Descent_direction", "0",Patterns::Integer(0,1),
		"0: Preconditioned descent, 1: Newton Method");
	prm.declare_entry("init_p", "10",Patterns::Double(2),
		"Initial p");
	prm.declare_entry("delta_p", "50",Patterns::Double(0),
		"increase of p");
	prm.declare_entry("Max_CG_it", "1500",Patterns::Integer(1),
		"Maximum Number of CG iterations");
	prm.declare_entry("CG_tol", "1e-10",Patterns::Double(0),
		"Tolerance for CG iterations");
	prm.declare_entry("max_LS_it", "45",Patterns::Integer(1),
		"Maximum Number of LS iterations");
	prm.declare_entry("line_search_tolerence", "1e-6",Patterns::Double(0),
		"line search tolerance constant (c1 in Nocedal-Wright)");
	prm.declare_entry("init_step_length", "1e-2",Patterns::Double(0),
		"initial step length in line-search");
	prm.declare_entry("Max_inner", "800",Patterns::Integer(1),
		"Maximum Number of inner iterations");
	prm.declare_entry("eps", "1.0e-8",Patterns::Double(0),
		"Threshold on norm of the derivative to declare optimality achieved");
	prm.declare_entry("hi_eps", "1.0e-9",Patterns::Double(0),
		"Threshold on norm of the derivative to declare optimality achieved in highly refined mesh");
	prm.declare_entry("hi_th", "8",Patterns::Integer(0),
		"Number of adaptive refinement before change convergence threshold");
}
prm.leave_subsection ();

}
void ParameterReader::read_parameters (const std::string parameter_file)
{
declare_parameters();
prm.read_input (parameter_file);
}

// ******************************************************************************************//
// The solution of the elastoplastic torsion problem on the unit disk with rhs=4.

template <int dim>
class Solution : public Function<dim>
{
public:
Solution () : Function<dim>() {}
virtual double value (const Point<dim> &pto, const unsigned int component = 0) const;
virtual Tensor<1,dim> gradient (const Point<dim> &pto, const unsigned int component = 0) const;
};

template <int dim>
double Solution<dim>::value (const Point<dim> &pto,const unsigned int) const
{
double r=sqrt(pto.square());
	if (r<0.5)
	  return -1.0*std::pow(r,2.0)+0.75;          
	else
          return 1.0-r;
}



template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim> &pto,const unsigned int) const
{
double r=sqrt(pto.square());
	if (r<0.5)
          return -2.0*pto;
	else
	  return  -1.0*pto/r;
}




// ****************************************************************************************** //
/*                 Compute the Lagrange multiplier (as a derived quantity)                   */


template <int dim>
class ComputeMultiplier : public DataPostprocessor<dim>
{
  private:
   	double p;
  public:
    ComputeMultiplier (double pe);

    virtual
    void compute_derived_quantities_scalar (
                const std::vector< double > &  ,
		const std::vector< Tensor< 1, dim > > & ,
		const std::vector< Tensor< 2, dim > > &  ,
		const std::vector< Point< dim > > &  ,
		const std::vector< Point< dim > > & ,
		std::vector< Vector< double > > &
    ) const;

    virtual std::vector<std::string> get_names () const;

    virtual
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation () const;
    virtual UpdateFlags get_needed_update_flags () const;
};


  template <int dim>
  ComputeMultiplier<dim>::ComputeMultiplier (double pe): p(pe)
     {}


template <int dim>
void ComputeMultiplier<dim>::compute_derived_quantities_scalar( 
        	const std::vector< double > &  	/*uh*/,
		const std::vector< Tensor< 1, dim > > &  duh,
		const std::vector< Tensor< 2, dim > > &  /*dduh*/,
		const std::vector< Point< dim > > &  /*	normals*/,
		const std::vector< Point< dim > > &  /*evaluation_points*/,
		std::vector< Vector< double > > &  	computed_quantities ) 	const
{
    const unsigned int n_quadrature_points = duh.size();

    for (unsigned int q=0; q<n_quadrature_points; ++q)
      { long  double sqrGrad=duh[q]* duh[q]; //squared norm of the gradient
	long double exponent=(p-2.0)/2*std::log(sqrGrad);
          computed_quantities[q](0) = std::sqrt(sqrGrad); // norm of the gradient
          computed_quantities[q](1)= std::exp(exponent); // multiplier      
}
}





template <int dim>
std::vector<std::string>
ComputeMultiplier<dim>::get_names() const
{
 std::vector<std::string> solution_names;
solution_names.push_back ("Gradient norm");
solution_names.push_back ("Lagrange multiplier"); 
  return solution_names;
}


template <int dim>
UpdateFlags
ComputeMultiplier<dim>::get_needed_update_flags () const
{
  return update_gradients;
}



  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
ComputeMultiplier<dim>:: get_data_component_interpretation () const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation;
    // norm of the gradient
    interpretation.push_back (DataComponentInterpretation::component_is_scalar); 
	// Lagrange multiplier
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    return interpretation;
}





// *************************************************************************************** //
  template <int dim>
  class ElastoplasticTorsion
  {
  public:
    ElastoplasticTorsion (ParameterHandler &);
    ~ElastoplasticTorsion ();
    void run ();

  private:
    void setup_system (const bool initial_step);
    void assemble_system ();
    bool solve (const int inner_it);
    void init_mesh ();
    void refine_mesh ();
    void set_boundary_values ();
    double phi (const double alpha) const;
    bool checkWolfe(double & alpha, double & phi_alpha) const;
    bool determine_step_length (const int inner_it);
    void print_it_message (const int counter, bool ks);
    void output_results (unsigned int refinement) const;
    void format_convergence_tables();
    void process_solution (const unsigned int cycle);
    void process_multiplier (const unsigned int cycle,const int iter,double time);
    double dual_error () const;
    double dual_infty_error () const;
    double W (double Du2) const;
    double Wp (double Du2) const;
    double G (double Du2) const;


        
	ParameterHandler &prm;
    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;
    ConstraintMatrix     hanging_node_constraints;
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    ConvergenceTable convergence_table;
    ConvergenceTable dual_convergence_table;
    Vector<double>       present_solution;
    Vector<double>       newton_update;
    Vector<double>       system_rhs;
    Vector<double>       grad_norm;
    Vector<double>	      lambda;


    double step_length,phi_zero,phi_alpha,phip,phip_zero;
    double old_step,old_phi_zero,old_phip;
	double L2_error;
   double H1_error;
   double Linfty_error; 
   double dual_L1_error; 	
   double dual_L_infty_error;
    FE_Q<dim> fe;
	double p;  
    double line_search_tolerence; // c_1 in Nocedal & Wright
    unsigned int dir_id;
	std::string elements;
	std::string Method;

};

/*******************************************************************************************/
//                              Boundary condition

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  double BoundaryValues<dim>::value (const Point<dim> &pto,
                                     const unsigned int /*component*/) const
  {	// could be anything else (theory works provided |Dg|_infty < 1/2)
	return 0.0;	
	
	/* A challenging BC leading to overdetermined problems
	   it is regulated by the parameter 0<eta<1.
	   eta closer to 1 leads to more difficult problems.	

	double pii=numbers::PI;     
	double theta=std::atan2(p[1],p[0])+pii;
	double eta=0.9;

	    if (theta <= 0.5)
			return eta*(theta*theta);
		else if ((theta >0.5) & (theta<= pii-0.5))
			return eta*(theta-0.25);
		else if ((theta>pii-0.5)&(theta<= pii+0.5))
			return eta*(pii-0.75-(theta-(pii-0.5))*(theta-(pii+0.5)));
		else if ((theta>pii+0.5)&(theta<= 2*pii-0.5))
			return eta*((2*pii-theta)-0.25);
		else
			return eta*((theta-2*pii)*(theta-2*pii) );*/
  }



/******************************************************************************/
//                       Right-Hand Side
template <int dim>
class RightHandSide : public Function<dim>
{
	public:
		RightHandSide () : Function<dim>() {}
		virtual double value (const Point<dim> &p, 
		const unsigned int component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
const unsigned int /*component*/) const
{ // set to constant = 4, for which explicit solution to compare exists
  // could be anything
	double return_value = 4.0;
	return return_value;
}



/*******************************************************************/
// The ElastoplasticTorsion class implementation

 // Constructor of the class 
  template <int dim>
ElastoplasticTorsion<dim>::ElastoplasticTorsion (ParameterHandler &param):
	prm(param),
    dof_handler (triangulation),
    L2_error(1.0),
    H1_error(1.0),
    Linfty_error(1.0),
    dual_L1_error(1.0), 	
    dual_L_infty_error(1.0),
	fe(2)
  {
    prm.enter_subsection ("Global Parameters");
	p=prm.get_double("p");
    prm.leave_subsection ();
    prm.enter_subsection ("Algorithm Parameters");
	line_search_tolerence=prm.get_double("line_search_tolerence");
	dir_id=prm.get_integer("Descent_direction");
    prm.leave_subsection ();
   	if (fe.degree==1)
		 elements="P1";
	else elements="P2";
			
	if (dir_id==0)
	  Method="Precond";
	else 
	  Method="Newton";
}



template <int dim>
ElastoplasticTorsion<dim>::~ElastoplasticTorsion ()
  {
    dof_handler.clear ();
  }

/*****************************************************************************************/
//  print iteration message

template <int dim>
void ElastoplasticTorsion<dim>::print_it_message (const int counter, bool ks)
{
	if(ks){
	        process_solution (counter);
   	        std::cout << "iteration="<< counter+1 << "  J(u_h)= "<< phi_zero << ", H1 error: "
        	<<  H1_error  <<", W0-1,infty error: "<< Linfty_error<< " J'(u_h)(w)= "<< phip
       		<< ", |J'(u_h)|= "<< system_rhs.l2_norm()<<std::endl;
			}
	else {
			std::cout << "iteration= " << counter+1 << " J(u_h)= " 
			<< phi_alpha << " J'(u_h)= "<< phip<<std::endl; 
		  } 
}


/*****************************************************************************************/
//                            Convergence Tables


/*************************************************************/
// formating

template <int dim>
void ElastoplasticTorsion<dim>::format_convergence_tables()
{
			convergence_table.set_precision("L2", 3);
			convergence_table.set_precision("H1", 3);
			convergence_table.set_precision("Linfty", 3);
			convergence_table.set_precision("function value", 3);
			convergence_table.set_precision("derivative", 3);
			dual_convergence_table.set_precision("dual_L1", 3);
			dual_convergence_table.set_precision("dual_Linfty", 3);
			dual_convergence_table.set_precision("L2", 3);
			dual_convergence_table.set_precision("H1", 3);
			dual_convergence_table.set_precision("Linfty", 3);
			convergence_table.set_scientific("L2", true);
			convergence_table.set_scientific("H1", true);
			convergence_table.set_scientific("Linfty", true);
			convergence_table.set_scientific("function value", true);
			convergence_table.set_scientific("derivative", true);
			dual_convergence_table.set_scientific("dual_L1", true);
			dual_convergence_table.set_scientific("dual_Linfty", true);
			dual_convergence_table.set_scientific("L2", true);
			dual_convergence_table.set_scientific("H1", true);
			dual_convergence_table.set_scientific("Linfty", true);

}

/****************************************/
// fill-in entry for the solution
template <int dim>
void ElastoplasticTorsion<dim>::process_solution (const unsigned int it)
{
	Vector<float> difference_per_cell (triangulation.n_active_cells());
	
	// compute L2 error (save to difference_per_cell)
	VectorTools::integrate_difference (dof_handler,present_solution,
		Solution<dim>(),difference_per_cell,QGauss<dim>(3),VectorTools::L2_norm);
	L2_error = difference_per_cell.l2_norm();

	// compute H1 error (save to difference_per_cell)
	VectorTools::integrate_difference (dof_handler,present_solution,Solution<dim>(),
		difference_per_cell,QGauss<dim>(3),VectorTools::H1_seminorm);
	H1_error = difference_per_cell.l2_norm();

	// compute W1infty error (save to difference_per_cell)
	const QTrapez<1> q_trapez;
	const QIterated<dim> q_iterated (q_trapez, 5);
	VectorTools::integrate_difference (dof_handler,present_solution,Solution<dim>(),
	difference_per_cell,q_iterated,VectorTools::W1infty_seminorm);
	Linfty_error = difference_per_cell.linfty_norm();


	convergence_table.add_value("cycle", it);
	convergence_table.add_value("p", p);
	convergence_table.add_value("L2", L2_error);
	convergence_table.add_value("H1", H1_error);
	convergence_table.add_value("Linfty", Linfty_error);
	convergence_table.add_value("function value", phi_alpha);
	convergence_table.add_value("derivative", phip);
}


/***************************************/
// fill-in entry  for the multiplier	
template <int dim>
void ElastoplasticTorsion<dim>::process_multiplier (const unsigned int cycle, const int iter,double time)
{
	const unsigned int n_active_cells=triangulation.n_active_cells();
	const unsigned int n_dofs=dof_handler.n_dofs();
	dual_L1_error=dual_error();
	dual_L_infty_error=dual_infty_error();


	dual_convergence_table.add_value("cycle", cycle);
	dual_convergence_table.add_value("p", p);
	dual_convergence_table.add_value("iteration_number", iter);
	dual_convergence_table.add_value("cpu_time", time);
	dual_convergence_table.add_value("cells", n_active_cells);
	dual_convergence_table.add_value("dofs", n_dofs);
	dual_convergence_table.add_value("L2", L2_error);
	dual_convergence_table.add_value("H1", H1_error);
	dual_convergence_table.add_value("Linfty", Linfty_error);
	dual_convergence_table.add_value("dual_L1", dual_L1_error);
	dual_convergence_table.add_value("dual_Linfty", dual_L_infty_error);

}




/****************************************************************************************/
// ElastoplasticTorsion::setup_system
// unchanged from step-15

  template <int dim>
  void ElastoplasticTorsion<dim>::setup_system (const bool initial_step)
  {
    if (initial_step)
      {
        dof_handler.distribute_dofs (fe);
        present_solution.reinit (dof_handler.n_dofs());
        grad_norm.reinit (dof_handler.n_dofs());
        lambda.reinit (dof_handler.n_dofs());

        hanging_node_constraints.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler,
        hanging_node_constraints);
        hanging_node_constraints.close ();
      }


    // The remaining parts of the function

    newton_update.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
    hanging_node_constraints.condense (c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit (sparsity_pattern);
  }

/***************************************************************************************/
  /* the coeffcients W, W' and G defining the problem.
   
     Min_u \int W(|Du|^2) dx
     
  They must be consistent as G(s)=W'(s)+2s W''(s) for any s>0.
  recall that they receive the SQUARED gradient. */

  template <int dim>
  double ElastoplasticTorsion<dim>::W (double Du2) const
  {
	return Du2;
  }

  template <int dim>
  double ElastoplasticTorsion<dim>::Wp (double Du2) const
  {
	return 1.0;
  }

  template <int dim>
  double ElastoplasticTorsion<dim>::G (double Du2) const
  {
	return 1.0;
  }
/***************************************************************************************/

  template <int dim>
  void ElastoplasticTorsion<dim>::assemble_system ()
  {


    const QGauss<dim>  quadrature_formula(3);
	const RightHandSide<dim> right_hand_side;
    system_matrix = 0;
    system_rhs = 0;

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_gradients         |
						      update_values           |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    const unsigned int           n_q_points    = quadrature_formula.size();

    FullMatrix<double>           cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>               cell_rhs (dofs_per_cell);

    std::vector<Tensor<1, dim> > old_solution_gradients(n_q_points);
    std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);


    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);
        fe_values.get_function_gradients(present_solution,
                                         old_solution_gradients);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
             long double coeff=0.0;
	     	 long double a=old_solution_gradients[q_point] * old_solution_gradients[q_point];	
	     	 long double exponent=(p-2.0)/2*std::log(a);
             coeff= std::exp( exponent);  
             for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  { 
                  	if (dir_id==1){
                    	cell_matrix(i, j) +=  fe_values.shape_grad(i, q_point) *  fe_values.shape_grad(j, q_point)
							* (G(a)+(p-1.0)*coeff)    * fe_values.JxW(q_point);
					}
					else {
					    cell_matrix(i, j) +=  fe_values.shape_grad(i, q_point) *  fe_values.shape_grad(j, q_point)
							* (Wp(a)+coeff)                                                  
							   * fe_values.JxW(q_point);
					}
                  }

                cell_rhs(i) -= (  fe_values.shape_grad(i, q_point)
                                         * old_solution_gradients[q_point]
											* (Wp(a)+coeff)
                                 -right_hand_side.value(fe_values.quadrature_point(q_point))
									*fe_values.shape_value(i, q_point)
                                      ) 
							* fe_values.JxW(q_point);
              }
           }

		cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              system_matrix.add (local_dof_indices[i],
                                 local_dof_indices[j],
                                 cell_matrix(i,j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
    }

    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);

    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(),
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        newton_update,
                                        system_rhs);
 }




/**********************************      Refine Mesh      ****************************************/
// unchanged from step-15

  template <int dim>
  void ElastoplasticTorsion<dim>::refine_mesh ()
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(3),
                                        typename FunctionMap<dim>::type(),
                                        present_solution,
                                        estimated_error_per_cell);

	prm.enter_subsection ("Mesh & Refinement Parameters");
		const double top_fraction=prm.get_double("top_fraction_of_cells");
		const double bottom_fraction=prm.get_double("bottom_fraction_of_cells");
	prm.leave_subsection ();
    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     top_fraction, bottom_fraction);

    triangulation.prepare_coarsening_and_refinement ();
    SolutionTransfer<dim> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
    triangulation.execute_coarsening_and_refinement();
    dof_handler.distribute_dofs(fe);
    Vector<double> tmp(dof_handler.n_dofs());
    solution_transfer.interpolate(present_solution, tmp);
    present_solution = tmp;
    set_boundary_values ();
    hanging_node_constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler,
                                          hanging_node_constraints);
    hanging_node_constraints.close();
    hanging_node_constraints.distribute (present_solution);
    setup_system (false);
  }


/*******************************************************************************************/
// Dump the norm of the gradient and the lagrange multiplier in vtu format for visualization    
  template <int dim>
  void ElastoplasticTorsion<dim>::output_results (unsigned int counter) const
  {    
	// multiplier object contains both |Du| and lambda.
	ComputeMultiplier<dim> multiplier(p);	
	DataOut<dim> data_out;

    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (present_solution, "solution");
    data_out.add_data_vector (present_solution, multiplier);
    data_out.build_patches ();
    std::ostringstream p_str;
    p_str << p<<"-cycle-"<<counter;
    std::string str = p_str.str();
    const std::string filename = "solution-" + str+".vtu";
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);
}

/********************************************************************************************/
// unchanged from step-15
  template <int dim>
  void ElastoplasticTorsion<dim>::set_boundary_values ()
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              BoundaryValues<dim>(),
                                              boundary_values);
    for (std::map<types::global_dof_index, double>::const_iterator
         bp = boundary_values.begin();
         bp != boundary_values.end(); ++bp)
      present_solution(bp->first) = bp->second;
  }


/****************************************************************************************/
//  COMPUTE \phi(\alpha)=J_p(u_h+\alpha w) 
  template <int dim>
  double ElastoplasticTorsion<dim>::phi (const double alpha) const
  {
    double obj = 0.0;
	const RightHandSide<dim> right_hand_side;
    Vector<double> evaluation_point (dof_handler.n_dofs());
    evaluation_point = present_solution;  // copy of u_h
    evaluation_point.add (alpha, newton_update); // u_{n+1}=u_n+alpha w_n

    const QGauss<dim>  quadrature_formula(3);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_gradients         |
		                     update_values    |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    const unsigned int           n_q_points    = quadrature_formula.size();

    Vector<double>               cell_residual (dofs_per_cell);
    std::vector<Tensor<1, dim> > gradients(n_q_points);
    std::vector<double> values(n_q_points);


    std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_residual = 0;
        fe_values.reinit (cell);
        fe_values.get_function_gradients (evaluation_point, gradients);
		fe_values.get_function_values (evaluation_point, values);


        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
              double Du2=gradients[q_point] *  gradients[q_point]; // Du2=|Du|^2
              double penalty;
              if (Du2<1.0e-10)
                penalty=0.0;
              else
                penalty=std::pow(Du2,p/2.0); // penalty=|Du|^p

		      // obj+= 1/2 W(|Du|^2)+1/p |Du|^p -fu (see (1))
              obj+=(    
              	(0.5*W(Du2)+penalty/p)- right_hand_side.value(fe_values.quadrature_point(q_point))*values[q_point]
                   ) * fe_values.JxW(q_point);
            }

        }

    return obj;
  }


/***************************************************************************************************/
// Compute L^1 error norm of Lagrange Multiplier
// with respect to exact solution (cf. Alvarez & Flores, 2015)

  template <int dim>
  double ElastoplasticTorsion<dim>::dual_error () const
  {
    double obj = 0.0;

    const QGauss<dim>  quadrature_formula(3);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_gradients         |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    const unsigned int           n_q_points    = quadrature_formula.size();

    Vector<double>               cell_residual (dofs_per_cell);
    std::vector<Tensor<1, dim> > gradients(n_q_points);
  
    std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_residual = 0;
        fe_values.reinit (cell);
        fe_values.get_function_gradients (present_solution, gradients);

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
              double coeff=gradients[q_point] *  gradients[q_point] ;
              if (coeff<1.0e-15)
                  coeff=0.0;
              else
                    coeff=std::pow(coeff,(p-2.0)/2.0); // |Du_p|^(p-2)

              double r=std::sqrt(fe_values.quadrature_point(q_point).square());
	      double exact=0;	      
	      if (r>0.5)
	      	exact= 2*r-1;
		
              obj+=(     std::abs(coeff-exact) ) * fe_values.JxW(q_point);
            }

        }

    return obj;
  }

/*******************************************************************************************/
// Compute L^infinity error norm of Lagrange Multiplier
// with respect to exact solution (cf. Alvarez & Flores, 2015)

  template <int dim>
  double ElastoplasticTorsion<dim>::dual_infty_error () const
  {
    double obj = 0.0;
   const QTrapez<1> q_trapez;
   const QIterated<dim> quadrature_formula (q_trapez, 10);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_gradients         |
                             update_quadrature_points );

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    const unsigned int           n_q_points    = quadrature_formula.size();

    Vector<double>               cell_residual (dofs_per_cell);
    std::vector<Tensor<1, dim> > gradients(n_q_points);
  
    std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_residual = 0;
        fe_values.reinit (cell);
        fe_values.get_function_gradients (present_solution, gradients);

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
              long double sqdGrad=gradients[q_point] *  gradients[q_point] ;
              double r=std::sqrt(fe_values.quadrature_point(q_point).square());
	      double exact=0;	      
	      if (r>0.5)
	      	exact= 2*r-1.0;
		// compute |Du|^(p-2) as  exp(p-2/2*log(Du^2)) 
		long double exponent=(p-2.0)/2*std::log(sqdGrad);
		long double coeff=std::exp(exponent);
		
             if(std::abs(coeff-exact)>obj )
		obj=std::abs(coeff-exact);
            }

        }

    return obj;
  }

/*****************************************************************************************/
// check whether putative step-length satisfies sufficient decrease conditions
  template <int dim>
bool ElastoplasticTorsion<dim>::checkWolfe(double & alpha, double & phi_alpha) const
{ 
if (phi_alpha< phi_zero+line_search_tolerence*phip*alpha )
	return true;
   else
	return false;		
}


/*****************************************************************************************/
// Find a step-length satisfying sufficient decrease condition by line-search 
// uses quadratic interpolation

  template <int dim>
bool ElastoplasticTorsion<dim>::determine_step_length(const int inner_it) 
{
    unsigned int it=0;
	bool done;
	double alpha,nalpha;
	prm.enter_subsection ("Algorithm Parameters");
		const unsigned int max_LS_it=prm.get_integer("max_LS_it");
		double init_SL=prm.get_double("init_step_length");
	prm.leave_subsection ();
	if (inner_it==0)
		alpha=init_SL;
	else
	{
		alpha=std::min(1.45*old_step*old_phip/phip,1.0);
	}
	phi_alpha=phi(alpha);
	std::cerr << "Step length=" << alpha << ", Value= " << phi_alpha;
	// check if step-size satisfies sufficient decrease condition
    done=checkWolfe(alpha,phi_alpha);
    if (done)
    		std::cerr << " accepted" << std::endl;
	else 
		std::cerr << " rejected" ;
		
	while ((!done) & (it<max_LS_it)) {
		// new try obtained by quadratic interpolation
		nalpha=-(phip*alpha*alpha)/(2*(phi_alpha-phi_zero-phip*alpha));

        if (nalpha<1e-3*alpha ||  std::abs(nalpha-alpha)/alpha<1e-8)  
          nalpha=alpha/2;
		else if( phi_alpha-phi_zero>1e3*std::abs(phi_zero) )
			nalpha=alpha/10;
		alpha=nalpha;
		phi_alpha=phi(alpha);
		done=checkWolfe(alpha,phi_alpha);
		if (done)
			std::cerr << ", finished with steplength= "<< alpha<< ", fcn value= "<< phi_alpha<<std::endl;
		it=it+1;
	}
	if (!done){
		std::cerr << ", max. no. of iterations reached wiht steplength= "<< alpha
		<< ", fcn value= "<< phi_alpha<<std::endl;
		return false;
	}
	else{
		  step_length=alpha;		
		  return true;
	}

}

/**************************************************************************************************/
  // ElastoplasticTorsion::init_mesh()

  template <int dim>
  void ElastoplasticTorsion<dim>::init_mesh ()
  {
	// get parameters
	prm.enter_subsection ("Mesh & Refinement Parameters");
		const int domain_id=prm.get_integer("Code for the domain");
		const int init_ref=prm.get_integer("No of initial refinements");
	prm.leave_subsection ();
	

	if (domain_id==0){
    // For the unit disk around the origin
    GridGenerator::hyper_ball (triangulation);
    static const HyperBallBoundary<dim> boundary;
    triangulation.set_boundary (0, boundary);
					 }
	else if (domain_id==1){
	// For the unit square
    GridGenerator::hyper_cube (triangulation, 0, 1);}
	else if (domain_id==2){
	/* For Glowinski's domain
      ___    ___   __ 1
     |   |__|   |  __ .8
	 |	        |
	 |          |
	 |__________|  __ 0 

     |   |  |   |
     0  .4 .6   1 
	    
	*/
	Triangulation<dim> tria1;
	Triangulation<dim> tria2;
	Triangulation<dim> tria3;
	Triangulation<dim> tria4;
	Triangulation<dim> tria5;
	Triangulation<dim> tria6;
	GridGenerator::hyper_rectangle(tria1, Point<2>(0.0,0.0), Point<2>(0.4,0.8));
	GridGenerator::hyper_rectangle(tria2, Point<2>(0.0,0.8), Point<2>(0.4,1.0));
	GridGenerator::hyper_rectangle(tria3, Point<2>(0.4,0.0), Point<2>(0.6,0.8));
	GridGenerator::hyper_rectangle(tria4, Point<2>(0.6,0.0), Point<2>(1.0,0.8));
	GridGenerator::hyper_rectangle(tria5, Point<2>(0.6,0.8), Point<2>(1.0,1.0));
	GridGenerator::merge_triangulations (tria1, tria2, tria6);
	GridGenerator::merge_triangulations (tria6, tria3, tria6);
	GridGenerator::merge_triangulations (tria6, tria4, tria6);
	GridGenerator::merge_triangulations (tria6, tria5, triangulation);
							}
	// perform initial refinements
    triangulation.refine_global(init_ref);
}

/**************************************************************************************************/
  // ElastoplasticTorsion::solve(inner_it)
  // Performs one inner iteration

  template <int dim>
  bool ElastoplasticTorsion<dim>::solve (const int inner_it)
  {
	prm.enter_subsection ("Algorithm Parameters");
		const unsigned int max_CG_it=prm.get_integer("Max_CG_it");
		const double CG_tol=prm.get_double("CG_tol");
	prm.leave_subsection ();

    SolverControl solver_control (max_CG_it,CG_tol);
    SolverCG<>    solver (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix,0.25);

    solver.solve (system_matrix, newton_update, system_rhs,
                  preconditioner);
    hanging_node_constraints.distribute (newton_update);
   /******  save current quantities for line-search  **** */
   // Recall that phi(alpha)=J(u+alpha w)
    old_step=step_length;	
    old_phi_zero=phi_zero;
    phi_zero=phi(0); // phi(0)=J(u)
    old_phip=phip;
    phip=-1.0*(newton_update*system_rhs); //phi'(0)=J'(u) *w, rhs=-J'(u).
	if (inner_it==0)
		phip_zero=phip;

    if (phip>0){	// this should not happen, step back
		 std::cout << "Not a descent direction!" <<std::endl;
    	present_solution.add (-1.0*step_length, newton_update);
		step_length=step_length/2;
		phip=old_phip;
		return false;
					}
	else{
		if(determine_step_length(inner_it)){
			// update u_{n+1}=u_n+alpha w_n
		 	present_solution.add (step_length, newton_update);
		 	return true;}
		 	else return false;
		}
}



/*************************************************************************************************************/
// ElastoplasticTorsion::run
template <int dim>
void ElastoplasticTorsion<dim>::run ()
{

	// get parameters
	prm.enter_subsection ("Mesh & Refinement Parameters");
		const int adapt_ref=prm.get_integer("No of adaptive refinements");
	prm.leave_subsection ();
	prm.enter_subsection ("Algorithm Parameters");
		const int max_inner=prm.get_integer("Max_inner");
		const double eps=prm.get_double("eps");
		const double hi_eps=prm.get_double("hi_eps");
		const int hi_th=prm.get_integer("hi_th");
		const double init_p=prm.get_double("init_p");
		const double delta_p=prm.get_double("delta_p");
	prm.leave_subsection ();
	prm.enter_subsection ("Global Parameters");
		bool known_solution=prm.get_bool("known_solution");
		double actual_p=prm.get_double("p");
	prm.leave_subsection ();
	/************************/

	// init Timer
    Timer timer;
    double ptime=0.0;
    timer.start ();
    	
	// initalize mesh for the selected domain
	init_mesh();

	// setup FE space
    setup_system (true);
    set_boundary_values ();

	// init counters
	int global_it=0;		// Total inner iterations (counting both loops)
	int cycle=0; 	        // Total outer iterations (counting both loops)
	int refinement = 0;    //  Refinements  performed (adaptive) = outer iterations 2nd loop

  
    // prepare to start first loop  
	p=init_p;	
	bool well_solved=true;

	/*****************************          First loop      ***********************************/
	/****************** Prepare initial condition using increasing p  *************************/
	while(p<actual_p) // outer iteration, increasing p.
	{
    	std::cout <<"--Preparing initial condition with p="<<p<<" iter.= " << global_it<<  "  .-- "<< std::endl;
	    timer.restart(); 
        for (int inner_iteration=0; inner_iteration<max_inner; ++inner_iteration,++global_it)
		{
			assemble_system ();
	        well_solved=solve (inner_iteration);
	        print_it_message (global_it, known_solution);
            if(
	            ((system_rhs.l2_norm()/std::sqrt(system_rhs.size()) <1e-4) & (cycle<1)) |
    	        ((system_rhs.l2_norm()/std::sqrt(system_rhs.size()) <1e-5) & (cycle>=1)) |
    	        !well_solved
               )
               break;			
		}
		ptime=timer();
		if (well_solved)
			output_results (cycle);
		    		  
        if(known_solution){
     		process_multiplier(cycle,global_it,ptime);
			//dual_convergence_table.write_tex(dual_error_table_file);
		}
        refine_mesh();
        cycle++;  
        p+=delta_p;
    }
    /***************************    first loop finished        ********************/
    
    
    // prepare for second loop      
	p=actual_p;
	well_solved=true;
	
	
	/*****************************        Second loop         *********************************/
	/**************************** Solve problem for target p  *********************************/
	
    std::cout << "============ Solving  problem with p="   <<p << " =================="  << std::endl;
    /*****    Outer iteration - refining mesh  ******************/
    while ((cycle<adapt_ref) & well_solved) 
      {
		timer.restart();
		// inner iteration
        for (int inner_iteration=0; inner_iteration<max_inner; ++inner_iteration,++global_it)
          {	   
            assemble_system ();
            well_solved=solve (inner_iteration);
            print_it_message (global_it, known_solution);
            
            if( 
               ((system_rhs.l2_norm()/std::sqrt(system_rhs.size()) < eps) & (refinement<hi_th)) | 
               (( system_rhs.l2_norm()/	std::sqrt	(system_rhs.size()) <hi_eps)  | (!well_solved))
               )
               break;			
          }
          //inner iterations finished
        ptime=timer();
   		if (well_solved)
			output_results (cycle);
	    
		// compute and display error, if the explicit solution is known
		if(known_solution){
			process_multiplier(cycle,global_it,ptime);
			std::cout << "finished with H1 error: " <<  H1_error << ", dual error (L1): "
			<< dual_L1_error << "dual error (L infty): "<<dual_L_infty_error <<std::endl;
		} 

		// update counters						
		++refinement;
		++cycle;
		// refine mesh	
        std::cout << "******** Refined mesh " << cycle    << " ********"  << std::endl;
        refine_mesh();
	}// second loop

	// write convergence tables to file
	if(known_solution){
		format_convergence_tables();	
		std::string error_filename = "error"+Method+elements+".tex";
		std::ofstream error_table_file(error_filename.c_str());
		std::string dual_error_filename = "dual_error"+Method+elements+".tex";
		std::ofstream dual_error_table_file(dual_error_filename.c_str());
		convergence_table.write_tex(error_table_file);
		dual_convergence_table.write_tex(dual_error_table_file);
	}
  }//run()

}//namespace

/**********************************************************************************************/
// The main function
int main ()
{
	try
    {
		using namespace dealii;
		using namespace nsp;
		deallog.depth_console (0);

		ParameterHandler prm;
		ParameterReader param(prm);
		param.read_parameters("EPT.prm");
		ElastoplasticTorsion<2> ElastoplasticTorsionProblem(prm);
		ElastoplasticTorsionProblem	.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"    << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
