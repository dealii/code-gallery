/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */


#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <boost/math/special_functions/ellint_1.hpp>

#include <fstream>
#include <iostream>
#include <random>


namespace SwiftHohenbergSolver
{
  using namespace dealii;



  /// @brief This enum defines the five mesh types implemented
  ///        in this program and allows the user to pass which
  ///        mesh is desired to the solver at runtime. This is
  ///        useful for looping over different meshes.
  enum MeshType {HYPERCUBE, CYLINDER, SPHERE, TORUS, SINUSOID};

  
  /// @brief This enum defines the three initial conditions used
  ///        by the program. This allows for the solver class to
  ///        use a template argument to determine the desired
  ///        initial condition, which is helpful for setting up
  ///        loops to solve with a variety of different conditions
  enum InitialConditionType {HOTSPOT, PSUEDORANDOM, RANDOM};




  /// @brief This function warps points on a cyclindrical mesh by cosine wave along the central axis.
  ///        We use this function to generate the "sinusoid" mesh, which is the surface of revolution
  ///        bounded by the cosine wave.
  /// @tparam spacedim This is the dimension of the embedding space, which is where the input point lives
  /// @param p This is thel input point to be translated.
  /// @return The return as a tranlated point in the same dimensional space. This is the new point on the mesh.
  template<int spacedim>
  Point<spacedim> transform_function(const Point<spacedim>&p)
  {
    // Currently this only works for a 3-dimensional embedding space
    // because we are explicitly referencing the x, y, and z coordinates
    Assert(spacedim == 3, ExcNotImplemented());

    // Retruns a point where the x-coordinate is unchanged but the y and z coordinates are adjusted
    // by a cos wave of period 20, amplitude .5, and vertical shift 1
    return Point<spacedim>(p(0), p(1)*(1 + .5*std::cos((3.14159/10)*p(0))), p(2)*(1 + .5*std::cos((3.14159/10)*p(0))));
  }


  /// @brief Not currently implemented, but will function the same as above only with and undulary boundary curve rather
  ///        than a cosine boundary curve.
  /// @tparam spacedim See above
  /// @param p See above
  /// @return See above
  template<int spacedim>
  Point<spacedim> transform_function_2_electric_boogaloo(const Point<spacedim> &p)
  {
    Assert(spacedim == 3, ExcNotImplemented());
    return 0;
  }







  /// @brief  This is the class that holds all the important variables for the solver, as well as the important member
  ///         functions. This class is based off the HeatEquation class from step-26, so we won't go into full detail
  ///         on all the features, but we will highlight what has been changed for this problem.
  /// @tparam dim       This is the intrinsic dimension of the manifold we are solving on.
  /// @tparam spacedim  This is the dimension of the embedding space.
  /// @tparam MESH      This determines what manifold we are solving on
  /// @tparam ICTYPE    This determines what initial condition we use
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  class SHEquation
  {
  public:
    /// @brief Default constructor, initializes all variables and objects with default values
    SHEquation();

    
    /// @brief                          Overloaded constructor, allows user to pass values for important constants
    /// @param degree                   This is the degree of finite element used
    /// @param time_step_denominator    This determines what size timestep we use. The timestep is 1/time_step_denominator
    /// @param ref_num                  The number of times the mesh will be globally refined.
    /// @param r_constant               Constant for linear component, default 0.5
    /// @param g1_constant              Constant for quadratic component, default 0.5
    /// @param output_file_name         Self explanatory, default "solution-"
    /// @param end_time                 Determines when the solver stops, default 0.5, should be ~100 to see equilibrium solutions
    SHEquation(const unsigned int degree
                , double time_step_denominator
                , unsigned int ref_num
                /* , unsigned int iteration_number */
                , double r_constant = 0.5
                , double g1_constant = 0.5
                , std::string output_file_name = "solution-"
                , double end_time = 0.5);
    void run();

  private:
    void setup_system();
    void solve_time_step();
    void output_results() const;
    /// @brief This function calls a different grid generation function depending on the template argument MESH. Allows the solver object to generate
    ///        different mesh types based on the template parameter.
    void make_grid();

    /// @brief Generates a cylindrical mesh with radius 6 and width 6*pi by first creating a volumetric cylinder, extracting the boundary, and redefining the mesh as a cylinder, then
    ///        refining the mesh refinement_number times
    void make_cylinder();
    /// @brief Uses the same process as creating a cylinder, but then also warps the boundary of the cylinder by the function (1 + 0.5*cos(pi*x/10))
    void make_sinusoid();
    /// @brief Generates a spherical mesh of radius 6*pi using GridGenerator and refines it refinement_number times.
    void make_sphere();
    /// @brief Generates a torus mesh with inner radius 4 and outer radius 9 using GridGenerator and refines it refinement_number times.
    void make_torus();
    /// @brief Generates a hypercube mesh with sidelenth 12*pi using GridGenerator and refines it refinement_number times.
    void make_hypercube();


    /// @brief The degree of finite element to be used, default 1
    const unsigned int degree;

    /// @brief Object holding the mesh
    Triangulation<dim, spacedim> triangulation;
    /// @brief Object describing the finite element vectors at each node
    ///        (I believe this gives a basis for the finite elements at each node)
    FESystem<dim, spacedim>          fe;
    /// @brief Object which understands which finite elements are at each node
    DoFHandler<dim, spacedim>    dof_handler;

    /// @brief Describes the sparsity of the system matrix, allows for more efficient storage
    SparsityPattern      sparsity_pattern;

    /// @brief Object holding the system matrix, stored as a sparse matrix
    SparseMatrix<double> system_matrix;

    /// @brief Vector of coefficients for the solution in the current timestep
    ///        We solve for this in each timestep
    Vector<double> solution;
    /// @brief Stores the solution from the previous timestep. Used to compute non-linear terms
    Vector<double> old_solution;
    /// @brief Stores the coefficients of the right hand side function(in terms of the finite elements)
    ///        Is the RHS for the linear system
    Vector<double> system_rhs;

    /// @brief Stores the current time, in the units of the problem
    double       time;
    /// @brief The amount time is increased each iteration/ the denominator of the discretized time derivative
    double       time_step;
    /// @brief Counts the number of iterations that have ellapsed
    unsigned int timestep_number;
    /// @brief Used to compute the time_step: time_step = 1/timestep_denominator
    unsigned int timestep_denominator;
    /// @brief Determines how much to globally refine each mesh
    unsigned int refinement_number;

    /// @brief Coefficient of the linear term in the SH equation. This is often taken to be constant and g_1 allowed to vary
    const double r;
    /// @brief Coefficient of the quadratic term in the SH equation. Determines whether hexagonal lattices can form
    const double g1;
    /// @brief A control parameter for the cubic term. Can be useful for testing, in this code we let k=1 in all cases
    const double k;

    /// @brief Name used to create output file. Should not include extension
    const std::string output_file_name;

    /// @brief Determines when the solver terminates, endtime of ~100 are useful to see equilibrium results
    const double end_time;
  };


  /// @brief The function which applies zero Dirichlet boundary conditions, and is
  ///        not being used by the solver currently. Leaving the code in case this
  ///        is ever needed.
  /// @tparam spacedim The dimension of the points which the function takes as input
  template <int spacedim>
  class BoundaryValues : public Function<spacedim>
  {
  public:
    BoundaryValues()
      : Function<spacedim>(2)
    {}

    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override;
  };



  /// @brief            Returns 0 for all points. This is the output for the boundary
  /// @tparam spacedim  The dimension of points that are input
  /// @param p          The input point
  /// @param component  Determines whether we are solving for u or v.
  ///                   This determines which part of the system we are solving
  /// @return           0; This is the boundary value for all points
  template <int spacedim>
  double BoundaryValues<spacedim>::value(const Point<spacedim> & p,
                                    const unsigned int component) const
  {
   (void)component;
    AssertIndexRange(component, 2);

    return 0.;
  }

  /// @brief            This class holds the initial condition function we will use for the solver.
  ///                   Note that this class takes both MeshType and InitialConditionType as parameters.
  ///                   This class is capable of producing several different initial conditions without
  ///                   having to change the code each time, which makes it useful for running longer
  ///                   experiments without having to stop the code each time. The downside of this is
  ///                   the code is that the class is rather large, and functions have to be defined
  ///                   multiple times to be compatible with the different configurations of MESH and
  ///                   ICTYPE. Because of this, our implementation is not a good solution if more than
  ///                   a few variations of mesh and initial conditions need to be used.
  /// @tparam spacedim  The dimension of the input points
  /// @tparam MESH      The type of mesh to apply initial conditions to, of type MeshType
  /// @tparam ICTYPE    The type of initial condition to apply, of type InitialConditionType
  template<int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  class InitialCondition : public Function<spacedim>
  {
    private:
      /// @brief  The value of the parameter r, used to determine a bound for the magnitude of the initial conditions
      const double r;
      /// @brief  A center point, used to determine the location of the hot spot for the HotSpot initial condition
      Point<spacedim> center;
      /// @brief  Radius of the hot spot
      double radius;
      /// @brief  Stores the randomly generated coefficients for planar sine waves along the x-axis, used for psuedorandom initial conditions
      double x_sin_coefficients[10];
      /// @brief  Stores the randomly generated coefficients for planar sine waves along the y-axis, used for psuedorandom initial conditions
      double y_sin_coefficients[10];

    public:
      /// @brief  The default constructor for the class. Initializes a function of 2 parameters and sets r and radius to default values.
      ///         The constructor also loops through the coefficient arrays and stores the random coefficients for the psuedorandom initial condition.
      InitialCondition()
      : Function<spacedim>(2),
        r(0.5),
        radius(.5)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
        }
      }

      /// @brief        An overloaded constructor, takes r and radius as parameters and uses these for initialization. Also loops through
      ///               the coefficient arrays and stores the random coefficients for the psuedorandom initial condition.
      /// @param r      The value of the r parameter in the SH equation
      /// @param radius The radius of the hot spot
      InitialCondition(const double r,
                        const double radius)
      : Function<spacedim>(2),
        r(r),
        radius(radius)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
        }
      }

      /// @brief            The return value of the initial condition function. This function is highly overloaded to account for a variety
      ///                   of different initial condition and mesh configurations, based on the template parameter given.
      ///
      ///                   Note that each initial condition sets the v component to 1e18. The v initial condition should not effect our solutions,
      ///                   and this is a good way to make any bugs causing v's initial condition to affect the solution easy to detect
      ///
      ///                   The RANDOM initial condition type does not change from mesh to mesh, it just returns a random number between -sqrt(r) and sqrt(r)
      ///
      ///                   The HOTSPOT initial condition changes the center depending on the input mesh type so that the hotspot is on the surface of the mesh
      ///
      ///                   The PSEUDORANDOM initial condition generates a function by summing up 10 sine waves in the x and y directions, with periods chosen so
      ///                   that the smallest period wave can still be resolved by a mesh with global refinement 5 or higher. On the plane, the value at each point
      ///                   is the product of the x sine sum and the y sine sum evaluated at the point. On the cylinder and Sinusoid, the x component is still used
      ///                   for the x sine sum, but we use ((arctan(y, z) - pi)/pi)*6*pi for the y sine sum. This wraps the psuedorandom function around the cylinder
      ///                   so that we can compare it to the same initial conditions on the plane. This function will run for the torus and sphere, but it has not been
      ///                   implemented to be comparable to the plane.
      /// @param p 
      /// @param component 
      /// @return 
      virtual double value(const Point<spacedim> &p, const unsigned int component) const override;
  };

  /// @brief              Places a small hot spot in the center of the plane on the u solution, and set v to a large number
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<2, HYPERCUBE, HOTSPOT>::value(
    const Point<2> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      if(p.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Places the hot spot in the center of the cylinder, on the positive z side
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, CYLINDER, HOTSPOT>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      const Point<3> center(0, 0, 6);
      const Point<3> compare(p - center);
      if(compare.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Places the hot spot on the outside of the sphere, along the positive x axis
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, SPHERE, HOTSPOT>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      const Point<3> center(18.41988074, 0, 0);
      const Point<3> compare(p - center);
      if(compare.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Places the hot spot on the outside of the torus, along the x axis
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, TORUS, HOTSPOT>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      const Point<3> center(13., 0, 0);
      const Point<3> compare(p - center);
      if(compare.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Places the hot spot in the center of the sinusoid, on the positive z side
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, SINUSOID, HOTSPOT>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      const Point<3> center(0, 0, 9.);
      const Point<3> compare(p - center);
      if(compare.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Returns the value of the psuedorandom function at the input point, as described above
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<2, HYPERCUBE, PSUEDORANDOM>::value(
    const Point<2> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double y_val = 0;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        y_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(1)/((i+1)*1.178097245));
      }

      return x_val*y_val;
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Returns the value of the psuedorandom function at the input point, as described above
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, CYLINDER, PSUEDORANDOM>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double w_val = 0;
      double width = ((std::atan2(p(1),p(2)) - 3.1415926)/3.1415926)*18.84955592;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        w_val += y_sin_coefficients[i]*std::sin(2*3.141592653*width/((i+1)*1.178097245));
      }

      return x_val*w_val;
    }
    else{
      return 1e18;
    }
  }

  /// @brief              NOTE: Not particularly useful at the moment. Returns the value of the psuedorandom function at the input point, as described above
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, SPHERE, PSUEDORANDOM>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double y_val = 0;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        y_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(1)/((i+1)*1.178097245));
      }

      return x_val*y_val;
    }
    else{
      return 1e18;
    }
  }

  /// @brief              NOTE: Not particularly useful at the moment. Returns the value of the psuedorandom function at the input point, as described above
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, TORUS, PSUEDORANDOM>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double z_val = 0;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        z_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(2)/((i+1)*1.178097245));
      }

      return x_val*z_val;
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Returns the value of the psuedorandom function at the input point, as described above
  /// @param p            The input point
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, SINUSOID, PSUEDORANDOM>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double w_val = 0;
      double width = ((std::atan2(p(1),p(2)) - 3.1415926)/3.1415926)*18.84955592;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        w_val += y_sin_coefficients[i]*std::sin(2*3.141592653*width/((i+1)*1.178097245));
      }

      return x_val*w_val;
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Returns a random value between -sqrt(r) and sqrt(r)
  /// @param p            The input point, not used in this function
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<2, HYPERCUBE, RANDOM>::value(
    const Point<2> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Returns a random value between -sqrt(r) and sqrt(r)
  /// @param p            The input point, not used in this function
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, CYLINDER, RANDOM>::value(
    const Point<3> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Returns a random value between -sqrt(r) and sqrt(r)
  /// @param p            The input point, not used in this function
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, SPHERE, RANDOM>::value(
    const Point<3> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Returns a random value between -sqrt(r) and sqrt(r)
  /// @param p            The input point, not used in this function
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, TORUS, RANDOM>::value(
    const Point<3> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }

  /// @brief              Returns a random value between -sqrt(r) and sqrt(r)
  /// @param p            The input point, not used in this function
  /// @param component    Determines whether the input is for u or v
  /// @return             The value of the initial solution at the point
  template <>
  double InitialCondition<3, SINUSOID, RANDOM>::value(
    const Point<3> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }

  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  SHEquation<dim, spacedim, MESH, ICTYPE>::SHEquation()
    : degree(1)
    , fe(FE_Q<dim, spacedim>(degree), 2)
    , dof_handler(triangulation)
    , time_step(1. / 1500)
    , timestep_denominator(1500)
    , refinement_number(4)
    , r(0.5)
    , g1(0.5)
    , k(1.)
    , output_file_name("solution-")
    , end_time(0.5)
  {}

  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  SHEquation<dim, spacedim, MESH, ICTYPE>::SHEquation(const unsigned int degree,
                                            double       time_step_denominator,
                                            unsigned int ref_num,
                                            double       r_constant,
                                            double       g1_constant,
                                            std::string  output_file_name,
                                            double       end_time)
    : degree(degree)
    , fe(FE_Q<dim, spacedim>(degree), 2)
    , dof_handler(triangulation)
    , time_step(1. / time_step_denominator)
    , timestep_denominator(time_step_denominator)
    , refinement_number(ref_num)
    , r(r_constant)
    , g1(g1_constant)
    , k(1.)
    , output_file_name(output_file_name)
    , end_time(end_time)
  {}

  /// @brief              Distrubutes the finite element vectors to each DoF, creates the system matrix, solution, old_solution, and system_rhs vectors,
  ///                     and outputs the number of DoF's to the console.
  /// @tparam dim         The dimension of the manifold
  /// @tparam spacedim    The dimension of the ambient space
  /// @tparam MESH        The type of mesh being used, doesn't change how this function works
  /// @tparam ICTYPE      The type of initial condition used, doesn't change how this function works
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    // Counts the DoF's for outputting to consolse
    const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
    const unsigned int n_u = dofs_per_component[0],
                       n_v = dofs_per_component[1];

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: " << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_v << ')' << std::endl;

    DynamicSparsityPattern                dsp(dof_handler.n_dofs());

    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }


  /// @brief              Uses a direct solver to invert the system matrix, then multiplies the RHS vector by the inverted matrix to get the solution.
  ///                     Also includes a timer feature, which is currently commented out, but can be helpful to compute how long a run will take
  /// @tparam dim         The dimension of the manifold
  /// @tparam spacedim    The dimension of the ambient space
  /// @tparam MESH        The type of mesh being used, doesn't change how this function works
  /// @tparam ICTYPE      The type of initial condition used, doesn't change how this function works
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::solve_time_step()
  {
    // std::cout << "Solving linear system" << std::endl;
    // Timer timer;

    SparseDirectUMFPACK direct_solver;

    direct_solver.initialize(system_matrix);

    direct_solver.vmult(solution, system_rhs);

    // timer.stop();
    // std::cout << "done (" << timer.cpu_time() << " s)" << std::endl;
  }



  /// @brief              Converts the solution vector into a .vtu file and labels the outputs as u and v
  /// @tparam dim         The dimension of the manifold
  /// @tparam spacedim    The dimension of the ambient space
  /// @tparam MESH        The type of mesh being used, doesn't change how this function works
  /// @tparam ICTYPE      The type of initial condition used, doesn't change how this function works
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::output_results() const
  {
    std::vector<std::string> solution_names(1, "u");
    solution_names.emplace_back("v");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(1,
                     DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim, spacedim> data_out;
    data_out.add_data_vector(dof_handler,
                             solution,
                             solution_names,
                             interpretation /*,
                             DataOut<dim, spacedim>::type_dof_data*/);

    data_out.build_patches(degree + 1);

    // Takes the output_file_name string and appends timestep_number with up to three leading 0's
    const std::string filename = 
      output_file_name + Utilities::int_to_string(timestep_number, 3) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);
  }

  // Below are all the different template cases for the make_grid() function
  template <>
  void SHEquation<2, 2, HYPERCUBE, HOTSPOT>::make_grid()
  {
    make_hypercube();
  }

  template <>
  void SHEquation<2, 3, CYLINDER, HOTSPOT>::make_grid()
  {
    make_cylinder();
  }

  template <>
  void SHEquation<2, 3, SPHERE, HOTSPOT>::make_grid()
  {
    make_sphere();
  }

  template <>
  void SHEquation<2, 3, TORUS, HOTSPOT>::make_grid()
  {
    make_torus();
  }

  template <>
  void SHEquation<2, 3, SINUSOID, HOTSPOT>::make_grid()
  {
    make_sinusoid();
  }

  template <>
  void SHEquation<2, 2, HYPERCUBE, PSUEDORANDOM>::make_grid()
  {
    make_hypercube();
  }

  template <>
  void SHEquation<2, 3, CYLINDER, PSUEDORANDOM>::make_grid()
  {
    make_cylinder();
  }

  template <>
  void SHEquation<2, 3, SPHERE, PSUEDORANDOM>::make_grid()
  {
    make_sphere();
  }

  template <>
  void SHEquation<2, 3, TORUS, PSUEDORANDOM>::make_grid()
  {
    make_torus();
  }

  template <>
  void SHEquation<2, 3, SINUSOID, PSUEDORANDOM>::make_grid()
  {
    make_sinusoid();
  }

  template <>
  void SHEquation<2, 2, HYPERCUBE, RANDOM>::make_grid()
  {
    make_hypercube();
  }

  template <>
  void SHEquation<2, 3, CYLINDER, RANDOM>::make_grid()
  {
    make_cylinder();
  }

  template <>
  void SHEquation<2, 3, SPHERE, RANDOM>::make_grid()
  {
    make_sphere();
  }

  template <>
  void SHEquation<2, 3, TORUS, RANDOM>::make_grid()
  {
    make_torus();
  }

  template <>
  void SHEquation<2, 3, SINUSOID, RANDOM>::make_grid()
  {
    make_sinusoid();
  }


  /// @brief              Runs the solver. First it creates the mesh and sets up the system, then constructs the system matrix, and finally loops over time to create
  ///                     the RHS vector and solve the system at each step
  /// @tparam dim         The dimension of the manifold
  /// @tparam spacedim    The dimension of the ambient space
  /// @tparam MESH        The type of mesh being used
  /// @tparam ICTYPE      The type of initial condition used, doesn't change how this function works
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::run()
  {
    make_grid();

    setup_system();

    // Counts total time ellapsed
    time            = 0.0;
    // Counts number of iterations
    timestep_number = 0;

    // Sets the random seed so runs are repeatable, remove for varying random initial conditions
    std::srand(314);

    InitialCondition<spacedim, MESH, ICTYPE> initial_conditions(r, 0.5);

    // Applies the initial conditions to the old_solution
    VectorTools::interpolate(dof_handler,
                             initial_conditions,
                             old_solution);
    solution = old_solution;

    // Outputs initial solution
    output_results();

    // Sets up the quadrature formula and FEValues object
    const QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim, spacedim> fe_values(fe, quadrature_formula, 
                                      update_values | update_gradients |
                                      update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    // The vector which stores the global indices that each local index connects to
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Extracts the finite elements associated to u and v
    const FEValuesExtractors::Scalar u(0);
    const FEValuesExtractors::Scalar v(1);

    // Loops over the cells to create the system matrix. We do this only once becase the timestep is constant
    for(const auto &cell : dof_handler.active_cell_iterators()){
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      cell->get_dof_indices(local_dof_indices);

      for(const unsigned int q_index : fe_values.quadrature_point_indices()){

        for(const unsigned int i : fe_values.dof_indices()){
          // These are the ith finite elements associated to u and v
          const double phi_i_u                   = fe_values[u].value(i, q_index);
          const Tensor<1, spacedim> grad_phi_i_u = fe_values[u].gradient(i, q_index);
          const double phi_i_v                   = fe_values[v].value(i, q_index);
          const Tensor<1, spacedim> grad_phi_i_v = fe_values[v].gradient(i, q_index);

          for(const unsigned int j : fe_values.dof_indices())
          {
            // These are the jth finite elements associated to u and v
            const double phi_j_u                   = fe_values[u].value(j, q_index);
            const Tensor<1, spacedim> grad_phi_j_u = fe_values[u].gradient(j, q_index);
            const double phi_j_v                   = fe_values[v].value(j, q_index);
            const Tensor<1, spacedim> grad_phi_j_v = fe_values[v].gradient(j, q_index);

            // This formula comes from expanding the PDE system
            cell_matrix(i, j) += (phi_i_u*phi_j_u - time_step*r*phi_i_u*phi_j_u
                                    + time_step*phi_i_u*phi_j_v - time_step*grad_phi_i_u*grad_phi_j_v
                                    + phi_i_v*phi_j_u - grad_phi_i_v*grad_phi_j_u 
                                    - phi_i_v*phi_j_v)*fe_values.JxW(q_index);
          }
        }
      }

      // Loops over the dof indices to fill the entries of the system_matrix with the local data
      for(unsigned int i : fe_values.dof_indices()){
        for(unsigned int j : fe_values.dof_indices()){
          system_matrix.add(local_dof_indices[i], 
                            local_dof_indices[j],
                            cell_matrix(i, j));
        }
      }
        }

    // Loops over time, incrementing by timestep, to create the RHS, solve the linear system, then output the result
    while (time <= end_time)
      {
        // Increments time and timestep_number
        time += time_step;
        ++timestep_number;

        // Outputs to console the number of iterations and current time. Currently outputs once every "second"
        if(timestep_number%timestep_denominator == 0){
          std::cout << "Time step " << timestep_number << " at t=" << time
                    << std::endl;
        }

        // Resets the system_rhs vector. THIS IS VERY IMPORTANT TO ENSURE THE SYSTEM IS SOLVED CORRECTLY AT EACH TIMESTEP
        system_rhs = 0;

        // Loops over cells, then quadrature points, then dof indices to construct the RHS
        for(const auto &cell : dof_handler.active_cell_iterators()){
          // Resets the cell_rhs. THIS IS ALSO VERY IMPORTANT TO ENSURE THE SYSTEM IS SOLVED CORRECTLY
          cell_rhs = 0;

          // Resets the FEValues object to only the current cell
          fe_values.reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          // Loop over the quadrature points
          for(const unsigned int q_index : fe_values.quadrature_point_indices()){
            // Stores the value of the previous solution at the quadrature point
            double Un1 = 0;
            
            // Loops over the dof indices to get the value of Un1
            for(const unsigned int i : fe_values.dof_indices()){
              Un1 += old_solution(local_dof_indices[i])*fe_values[u].value(i, q_index);
            }

            // Loops over the dof indices, using Un1 to construct the RHS for the current timestep. Un1 is used to account for the nonlinear terms in the SH equation
            for(const unsigned int i : fe_values.dof_indices()){
              cell_rhs(i) += (Un1 + time_step*g1*std::pow(Un1, 2) - time_step*k*std::pow(Un1, 3))
                              *fe_values[u].value(i, q_index)*fe_values.JxW(q_index);
            }
          }

          // Loops over the dof indices to store the local data in the global RHS vector
          for(unsigned int i : fe_values.dof_indices()){
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }


        }
        // This is where Dirichlet conditions are applied, or Neumann conditions if the code is commented out
        /* {
          BoundaryValues<spacedim> boundary_values_function;
          boundary_values_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_function,
                                                   boundary_values);

          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,
                                             system_rhs);
        } */

        solve_time_step();

        // Outputs the solution at regular intervals, currently once every "second" The SH equation evolves slowly in time, so this saves disk space
        if(timestep_number%timestep_denominator == 0){
          output_results();
        }

        old_solution = solution;
      }
  }

  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_cylinder()
  {
    // Creates a volumetric cylinder
    Triangulation<3> cylinder;
    GridGenerator::cylinder(cylinder, 6, 18.84955592);

    // Extracts the boundary mesh with ID 0, which happens to be the tube part of the cylinder
    GridGenerator::extract_boundary_mesh(cylinder, triangulation, {0});

    // The manifold information is lost upon boundary extraction. This sets the mesh boundary type to be a cylinder again
    const CylindricalManifold<dim, spacedim> boundary;
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, boundary);

    triangulation.refine_global(refinement_number);
  }

  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_sinusoid()
  {
    // Same process as above
    Triangulation<3> cylinder;
    GridGenerator::cylinder(cylinder, 6, 18.84955592);

    GridGenerator::extract_boundary_mesh(cylinder, triangulation, {0});

    const CylindricalManifold<dim, spacedim> boundary;
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, boundary);

    triangulation.refine_global(refinement_number);

    // We warp the mesh after refinement to avoid a jagged mesh. We can't tell the code that the boundary should be a perfect sine wave, so we only warp after the
    // mesh is fine enough to resolve this
    GridTools::transform(transform_function<spacedim>, triangulation);
  }
  
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_sphere()
  {
    GridGenerator::hyper_sphere(triangulation, Point<3>(0, 0, 0), 18.41988074);
    triangulation.refine_global(refinement_number);
  }

  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_torus()
  {
    GridGenerator::torus(triangulation, 9., 4.);
    triangulation.refine_global(refinement_number);
  }
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_hypercube()
  {
    GridGenerator::hyper_cube(triangulation, -18.84955592, 18.84955592);
    triangulation.refine_global(refinement_number);
  }
} // namespace SwiftHohenbergSolver



int main()
{
  using namespace SwiftHohenbergSolver;

  // An array of mesh types. We itterate over this to allow for longer runs without having to stop the code
  MeshType mesh_types[5] = {HYPERCUBE, CYLINDER, SPHERE, TORUS, SINUSOID};
  // An array of initial condition types. We itterate this as well, for the same reason
  InitialConditionType ic_types[3] = {HOTSPOT, PSUEDORANDOM, RANDOM};

  // Controls how long the code runs
  const double end_time = 100.;

  // The number of times we refine the hypercube mesh
  const unsigned int ref_num = 6;

  // The timestep will be 1/timestep_denominator
  const unsigned int timestep_denominator = 25;

  // Loops over mesh types, then initial condition types, then loops over values of g_1
  for(const auto MESH : mesh_types){
    for(const auto ICTYPE: ic_types){
      for(int i = 0; i < 8; ++i){
        // The value of g_1 passed to the solver object
        const double g_constant = 0.2*i;

        // Used to distinguish the start of each run
        std::cout<< std::endl << std::endl;

        try{
          // Switch statement that determines what template parameters are used by the solver object. Template parameters must be known at compile time, so we cannot
          // pass this as a varible unfortunately. In each case, we create a filename string (named appropriately for the particular case), output to the console what
          // we are running, create the solver object, and call run(). Note that for the cylinder, sphere, and sinusoid we decrease the refinement number by 1. This keeps
          // the number of dofs used in these cases comparable to the number of dofs on the 2D hypercube (otherwise the number of dofs is much larger). For the torus, we
          // decrease the refinement number by 2.
          switch (MESH)
          {
          case HYPERCUBE:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "HYPERCUBE-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 2, HYPERCUBE, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "HYPERCUBE-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 2, HYPERCUBE, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "HYPERCUBE-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 2, HYPERCUBE, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          case CYLINDER:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "CYLINDER-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, CYLINDER, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-1, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "CYLINDER-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, CYLINDER, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-1, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "CYLINDER-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, CYLINDER, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-1, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          case SPHERE:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "SPHERE-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SPHERE, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-1, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "SPHERE-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SPHERE, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-1, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "SPHERE-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SPHERE, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-1, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          case TORUS:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "TORUS-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, TORUS, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-2, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "TORUS-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, TORUS, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-2, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "TORUS-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, TORUS, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-2, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          case SINUSOID:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "SINUSOID-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SINUSOID, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-1, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "SINUSOID-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SINUSOID, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-1, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "SINUSOID-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SINUSOID, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num-1, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          default:
            break;
          }
        }
        catch (std::exception &exc)
        {
          std::cout << "An error occured" << std::endl;
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
        catch (...)
        {
          std::cout << "Error occured, made it past first catch" << std::endl;
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
    }
  }
  return 0;
}
