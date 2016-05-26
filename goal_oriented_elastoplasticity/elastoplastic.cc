/* ---------------------------------------------------------------------
 * $Id: elastoplastic.cc 31592 2013-11-08 16:47:28Z Ghorashi $
 *
 * Copyright (C) 2012 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Seyed Shahram Ghorashi, Bauhaus-Universit\"at Weimar, 2014
 *          Joerg Frohne, Texas A&M University and
 *                        University of Siegen, 2012, 2013
 *          Wolfgang Bangerth, Texas A&M University, 2012, 2013
 *          Timo Heister, Texas A&M University, 2013
 */

// @sect3{Include files}
// The set of include files is not much of a surprise any more at this time:
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/solution_transfer.h>

// And here the only two new things among the header files: an include file in
// which symmetric tensors of rank 2 and 4 are implemented, as introduced in
// the introduction:
#include <deal.II/base/symmetric_tensor.h>

// And a header that implements filters for iterators looping over all
// cells. We will use this when selecting only those cells for output that are
// owned by the present process in a %parallel program:
#include <deal.II/grid/filtered_iterator.h>

#include <fstream>
#include <iostream>

// This final include file provides the <code>mkdir</code> function
// that we will use to create a directory for output files, if necessary:
#include <sys/stat.h>

namespace ElastoPlastic
{
  using namespace dealii;

  void
  extrude_triangulation(const Triangulation<2, 2> &input,
                        const unsigned int n_slices,
                        const double height,
                        Triangulation<3,3> &result)
  {
    //  Assert (input.n_levels() == 1,
    //          ExcMessage ("The input triangulations must be coarse meshes."));
    Assert(result.n_cells()==0, ExcMessage("resultin Triangulation need to be empty upon calling extrude_triangulation."));
    Assert(height>0, ExcMessage("The height in extrude_triangulation needs to be positive."));
    Assert(n_slices>=2, ExcMessage("The number of slices in extrude_triangulation needs to be at least 2."));

    std::vector<Point<3> > points(n_slices*input.n_used_vertices());
    std::vector<CellData<3> > cells;
    cells.reserve((n_slices-1)*input.n_active_cells());

    for (unsigned int slice=0; slice<n_slices; ++slice)
      {
        for (unsigned int i=0; i<input.n_vertices(); ++i)

          {
            if (input.get_used_vertices()[i])
              {
                const Point<2> &v = input.get_vertices()[i];
                points[i+slice*input.n_vertices()](0) = v(0);
                points[i+slice*input.n_vertices()](1) = v(1);
                points[i+slice*input.n_vertices()](2) = height * slice / (n_slices-1);
              }
          }
      }

    for (Triangulation<2,2>::cell_iterator
         cell = input.begin_active(); cell != input.end(); ++cell)
      {
        for (unsigned int slice=0; slice<n_slices-1; ++slice)
          {
            CellData<3> this_cell;
            for (unsigned int v=0; v<GeometryInfo<2>::vertices_per_cell; ++v)
              {
                this_cell.vertices[v]
                  = cell->vertex_index(v)+slice*input.n_used_vertices();
                this_cell.vertices[v+GeometryInfo<2>::vertices_per_cell]
                  = cell->vertex_index(v)+(slice+1)*input.n_used_vertices();
              }

            this_cell.material_id = cell->material_id();
            cells.push_back(this_cell);
          }
      }

    SubCellData s;
    types::boundary_id bid=0;
    s.boundary_quads.reserve(input.n_active_lines()*(n_slices-1) + input.n_active_cells()*2);
    for (Triangulation<2,2>::cell_iterator
         cell = input.begin_active(); cell != input.end(); ++cell)
      {
        CellData<2> quad;
        for (unsigned int f=0; f<4; ++f)
          if (cell->at_boundary(f))
            {
              quad.boundary_id = cell->face(f)->boundary_indicator();
              bid = std::max(bid, quad.boundary_id);
              for (unsigned int slice=0; slice<n_slices-1; ++slice)
                {
                  quad.vertices[0] = cell->face(f)->vertex_index(0)+slice*input.n_used_vertices();
                  quad.vertices[1] = cell->face(f)->vertex_index(1)+slice*input.n_used_vertices();
                  quad.vertices[2] = cell->face(f)->vertex_index(0)+(slice+1)*input.n_used_vertices();
                  quad.vertices[3] = cell->face(f)->vertex_index(1)+(slice+1)*input.n_used_vertices();
                  s.boundary_quads.push_back(quad);
                }
            }
      }

    for (Triangulation<2,2>::cell_iterator
         cell = input.begin_active(); cell != input.end(); ++cell)
      {
        CellData<2> quad;
        quad.boundary_id = bid + 1;
        quad.vertices[0] = cell->vertex_index(0);
        quad.vertices[1] = cell->vertex_index(1);
        quad.vertices[2] = cell->vertex_index(2);
        quad.vertices[3] = cell->vertex_index(3);
        s.boundary_quads.push_back(quad);

        quad.boundary_id = bid + 2;
        for (int i=0; i<4; ++i)
          quad.vertices[i] += (n_slices-1)*input.n_used_vertices();
        s.boundary_quads.push_back(quad);
      }

    result.create_triangulation (points,
                                 cells,
                                 s);
  }

  namespace Evaluation
  {


    template <int dim>
    double get_von_Mises_stress(const SymmetricTensor<2, dim> &stress)
    {

      //      if (dim == 2)
      //      {
      //        von_Mises_stress = std::sqrt(  stress[0][0]*stress[0][0]
      //                                                         + stress[1][1]*stress[1][1]
      //                                                         - stress[0][0]*stress[1][1]
      //                                                         + 3*stress[0][1]*stress[0][1]);
      //      }else if (dim == 3)
      //      {
      //        von_Mises_stress = std::sqrt(  stress[0][0]*stress[0][0]
      //                                                       + stress[1][1]*stress[1][1]
      //                                                       + stress[2][2]*stress[2][2]
      //                                                       - stress[0][0]*stress[1][1]
      //                                                       - stress[1][1]*stress[2][2]
      //                                                       - stress[0][0]*stress[2][2]
      //                                                         + 3*( stress[0][1]*stress[0][1]
      //                                                              +stress[1][2]*stress[1][2]
      //                                                              +stress[0][2]*stress[0][2]) );
      //      }

      // -----------------------------------------------
      // "Perforated_strip_tension"
      // plane stress
//      const double von_Mises_stress = std::sqrt(  stress[0][0]*stress[0][0]
//                                                + stress[1][1]*stress[1][1]
//                                                - stress[0][0]*stress[1][1]
//                                                + 3*stress[0][1]*stress[0][1]);
      // -----------------------------------------------
      // otherwise
      // plane strain / 3d case
      const double von_Mises_stress = std::sqrt(1.5) * (deviator(stress)).norm();
      // -----------------------------------------------



      return von_Mises_stress;
    }


    template <int dim>
    class PointValuesEvaluation
    {
    public:
      PointValuesEvaluation (const Point<dim>  &evaluation_point);

      void compute (const DoFHandler<dim>  &dof_handler,
                    const Vector<double>   &solution,
                    Vector<double>         &point_values);

      DeclException1 (ExcEvaluationPointNotFound,
                      Point<dim>,
                      << "The evaluation point " << arg1
                      << " was not found among the vertices of the present grid.");
    private:
      const Point<dim>  evaluation_point;
    };


    template <int dim>
    PointValuesEvaluation<dim>::
    PointValuesEvaluation (const Point<dim>  &evaluation_point)
      :
      evaluation_point (evaluation_point)
    {}



    template <int dim>
    void
    PointValuesEvaluation<dim>::
    compute (const DoFHandler<dim>  &dof_handler,
             const Vector<double>   &solution,
             Vector<double>         &point_values)
    {
      const unsigned int dofs_per_vertex = dof_handler.get_fe().dofs_per_vertex;
      AssertThrow (point_values.size() == dofs_per_vertex,
                   ExcDimensionMismatch (point_values.size(), dofs_per_vertex));
      point_values = 1e20;

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      bool evaluation_point_found = false;
      for (; (cell!=endc) && !evaluation_point_found; ++cell)
        {
          if (cell->is_locally_owned() && !evaluation_point_found)
            for (unsigned int vertex=0;
                 vertex<GeometryInfo<dim>::vertices_per_cell;
                 ++vertex)
              {
                if (cell->vertex(vertex).distance (evaluation_point)
                    <
                    cell->diameter() * 1e-8)
                  {
                    for (unsigned int id=0; id!=dofs_per_vertex; ++id)
                      {
                        point_values[id] = solution(cell->vertex_dof_index(vertex,id));
                      }

                    evaluation_point_found = true;
                    break;
                  }
              }
        }

      AssertThrow (evaluation_point_found,
                   ExcEvaluationPointNotFound(evaluation_point));
    }


  }

  // @sect3{The <code>PointHistory</code> class}

  // As was mentioned in the introduction, we have to store the old stress in
  // quadrature point so that we can compute the residual forces at this point
  // during the next time step. This alone would not warrant a structure with
  // only one member, but in more complicated applications, we would have to
  // store more information in quadrature points as well, such as the history
  // variables of plasticity, etc. In essence, we have to store everything
  // that affects the present state of the material here, which in plasticity
  // is determined by the deformation history variables.
  //
  // We will not give this class any meaningful functionality beyond being
  // able to store data, i.e. there are no constructors, destructors, or other
  // member functions. In such cases of `dumb' classes, we usually opt to
  // declare them as <code>struct</code> rather than <code>class</code>, to
  // indicate that they are closer to C-style structures than C++-style
  // classes.
  template <int dim>
  struct PointHistory
  {
    SymmetricTensor<2,dim> old_stress;
    SymmetricTensor<2,dim> old_strain;
    Point<dim> point;
  };


  // @sect3{The <code>ConstitutiveLaw</code> class template}

  // This class provides an interface for a constitutive law, i.e., for the
  // relationship between strain $\varepsilon(\mathbf u)$ and stress
  // $\sigma$. In this example we are using an elastoplastic material behavior
  // with linear, isotropic hardening. Such materials are characterized by
  // Young's modulus $E$, Poisson's ratio $\nu$, the initial yield stress
  // $\sigma_0$ and the isotropic hardening parameter $\gamma$.  For $\gamma =
  // 0$ we obtain perfect elastoplastic behavior.
  //
  // As explained in the paper that describes this program, the first Newton
  // steps are solved with a completely elastic material model to avoid having
  // to deal with both nonlinearities (plasticity and contact) at once. To this
  // end, this class has a function <code>set_sigma_0()</code> that we use later
  // on to simply set $\sigma_0$ to a very large value -- essentially
  // guaranteeing that the actual stress will not exceed it, and thereby
  // producing an elastic material. When we are ready to use a plastic model, we
  // set $\sigma_0$ back to its proper value, using the same function.  As a
  // result of this approach, we need to leave <code>sigma_0</code> as the only
  // non-const member variable of this class.
  template <int dim>
  class ConstitutiveLaw
  {
  public:
    ConstitutiveLaw (const double E,
                     const double nu,
                     const double sigma_0,
                     const double gamma);

    void
    set_sigma_0 (double sigma_zero);

    bool
    get_stress_strain_tensor (const SymmetricTensor<2, dim> &strain_tensor,
                              SymmetricTensor<4, dim> &stress_strain_tensor) const;

    bool
    get_grad_stress_strain_tensor (const SymmetricTensor<2, dim> &strain_tensor,
                                   const std::vector<Tensor<2, dim> > &point_hessian,
                                   Tensor<5, dim> &stress_strain_tensor_grad) const;

    void
    get_linearized_stress_strain_tensors (const SymmetricTensor<2, dim> &strain_tensor,
                                          SymmetricTensor<4, dim> &stress_strain_tensor_linearized,
                                          SymmetricTensor<4, dim> &stress_strain_tensor) const;

  private:
    const double kappa;
    const double mu;
    double       sigma_0;
    const double gamma;

    const SymmetricTensor<4, dim> stress_strain_tensor_kappa;
    const SymmetricTensor<4, dim> stress_strain_tensor_mu;
  };

  // The constructor of the ConstitutiveLaw class sets the required material
  // parameter for our deformable body. Material parameters for elastic
  // isotropic media can be defined in a variety of ways, such as the pair $E,
  // \nu$ (elastic modulus and Poisson's number), using the Lame parameters
  // $\lambda,mu$ or several other commonly used conventions. Here, the
  // constructor takes a description of material parameters in the form of
  // $E,\nu$, but since this turns out to these are not the coefficients that
  // appear in the equations of the plastic projector, we immediately convert
  // them into the more suitable set $\kappa,\mu$ of bulk and shear moduli.  In
  // addition, the constructor takes $\sigma_0$ (the yield stress absent any
  // plastic strain) and $\gamma$ (the hardening parameter) as arguments. In
  // this constructor, we also compute the two principal components of the
  // stress-strain relation and its linearization.
  template <int dim>
  ConstitutiveLaw<dim>::ConstitutiveLaw (double E,
                                         double nu,
                                         double sigma_0,
                                         double gamma)
    :
    //--------------------
    // Plane stress
//    kappa (((E*(1+2*nu)) / (std::pow((1+nu),2))) / (3 * (1 - 2 * (nu / (1+nu))))),
//    mu (((E*(1+2*nu)) / (std::pow((1+nu),2))) / (2 * (1 + (nu / (1+nu))))),
    //--------------------
    // 3d and plane strain
    kappa (E / (3 * (1 - 2 * nu))),
    mu (E / (2 * (1 + nu))),
    //--------------------
    sigma_0(sigma_0),
    gamma(gamma),
    stress_strain_tensor_kappa (kappa
                                * outer_product(unit_symmetric_tensor<dim>(),
                                                unit_symmetric_tensor<dim>())),
    stress_strain_tensor_mu (2 * mu
                             * (identity_tensor<dim>()
                                - outer_product(unit_symmetric_tensor<dim>(),
                                                unit_symmetric_tensor<dim>()) / 3.0))
  {}


  template <int dim>
  void
  ConstitutiveLaw<dim>::set_sigma_0 (double sigma_zero)
  {
    sigma_0 = sigma_zero;
  }


  // @sect4{ConstitutiveLaw::get_stress_strain_tensor}

  // This is the principal component of the constitutive law. It projects the
  // deviatoric part of the stresses in a quadrature point back to the yield
  // stress (i.e., the original yield stress $\sigma_0$ plus the term that
  // describes linear isotropic hardening).  We need this function to calculate
  // the nonlinear residual in PlasticityContactProblem::residual_nl_system. The
  // computations follow the formulas laid out in the introduction.
  //
  // The function returns whether the quadrature point is plastic to allow for
  // some statistics downstream on how many of the quadrature points are
  // plastic and how many are elastic.
  template <int dim>
  bool
  ConstitutiveLaw<dim>::
  get_stress_strain_tensor (const SymmetricTensor<2, dim> &strain_tensor,
                            SymmetricTensor<4, dim> &stress_strain_tensor) const
  {
    SymmetricTensor<2, dim> stress_tensor;
    stress_tensor = (stress_strain_tensor_kappa + stress_strain_tensor_mu)
                    * strain_tensor;

//    const SymmetricTensor<2, dim> deviator_stress_tensor = deviator(stress_tensor);
//    const double deviator_stress_tensor_norm = deviator_stress_tensor.norm();
    const double von_Mises_stress = Evaluation::get_von_Mises_stress(stress_tensor);

    stress_strain_tensor = stress_strain_tensor_mu;
    if (von_Mises_stress > sigma_0)
      {
        const double beta = sigma_0 / von_Mises_stress;
        stress_strain_tensor *= (gamma + (1 - gamma) * beta);
      }

    stress_strain_tensor += stress_strain_tensor_kappa;

    return (von_Mises_stress > sigma_0);
  }


  template <int dim>
  bool
  ConstitutiveLaw<dim>::
  get_grad_stress_strain_tensor (const SymmetricTensor<2, dim> &strain_tensor,
                                 const std::vector<Tensor<2, dim> > &point_hessian,
                                 Tensor<5, dim> &stress_strain_tensor_grad) const
  {
    SymmetricTensor<2, dim> stress_tensor;
    stress_tensor = (stress_strain_tensor_kappa + stress_strain_tensor_mu)
                    * strain_tensor;

    const SymmetricTensor<2, dim> deviator_stress_tensor = deviator(stress_tensor);
    const double deviator_stress_tensor_norm = deviator_stress_tensor.norm();
    const double von_Mises_stress = Evaluation::get_von_Mises_stress(stress_tensor);

    if (von_Mises_stress > sigma_0)
      {
        const SymmetricTensor<2, dim> deviator_strain_tensor = deviator(strain_tensor);
        const double deviator_strain_tensor_norm = deviator_strain_tensor.norm();
        const double multiplier = -(1-gamma)*sigma_0/(2*mu*std::pow(deviator_strain_tensor_norm,3));

        Vector<double> multiplier_vector(dim);
        multiplier_vector = 0;

        for (unsigned int i=0; i!=dim; ++i)
          for (unsigned int m=0; m!=dim; ++m)
            for (unsigned int n=0; n!=dim; ++n)
              {
                multiplier_vector(i) += deviator_strain_tensor[m][n] *
                                        ( 0.5*( point_hessian[m][n][i] + point_hessian[n][m][i] )
                                          + ( m==n && dim==2 ? -1/dim*(point_hessian[0][0][i]
                                                                       + point_hessian[1][1][i]) : 0 )
                                          + ( m==n && dim==3 ? -1/dim*(point_hessian[0][0][i]
                                                                       + point_hessian[1][1][i]
                                                                       + point_hessian[2][2][i]) : 0 ) );
              }

        // -----------------------------------------------
        // "Perforated_strip_tension"
        // plane stress
//      const double VM_factor = std::sqrt(2);
        // -----------------------------------------------
        // otherwise
        // plane strain / 3d case
        const double VM_factor = std::sqrt(1.5);
        // -----------------------------------------------

        for (unsigned int i=0; i!=dim; ++i)
          for (unsigned int j=0; j!=dim; ++j)
            for (unsigned int k=0; k!=dim; ++k)
              for (unsigned int l=0; l!=dim; ++l)
                for (unsigned int m=0; m!=dim; ++m)
                  {
                    stress_strain_tensor_grad[i][j][k][l][m] = 1/VM_factor
                                                               * multiplier
                                                               * stress_strain_tensor_mu[i][j][k][l]
                                                               * multiplier_vector(m);
                  }

      }
    else
      {
        stress_strain_tensor_grad = 0;
      }

    return (von_Mises_stress > sigma_0);
  }


  // @sect4{ConstitutiveLaw::get_linearized_stress_strain_tensors}

  // This function returns the linearized stress strain tensor, linearized
  // around the solution $u^{i-1}$ of the previous Newton step $i-1$.  The
  // parameter <code>strain_tensor</code> (commonly denoted
  // $\varepsilon(u^{i-1})$) must be passed as an argument, and serves as the
  // linearization point. The function returns the derivative of the nonlinear
  // constitutive law in the variable stress_strain_tensor, as well as the
  // stress-strain tensor of the linearized problem in
  // stress_strain_tensor_linearized.  See
  // PlasticityContactProblem::assemble_nl_system where this function is used.
  template <int dim>
  void
  ConstitutiveLaw<dim>::
  get_linearized_stress_strain_tensors (const SymmetricTensor<2, dim> &strain_tensor,
                                        SymmetricTensor<4, dim> &stress_strain_tensor_linearized,
                                        SymmetricTensor<4, dim> &stress_strain_tensor) const
  {
    SymmetricTensor<2, dim> stress_tensor;
    stress_tensor = (stress_strain_tensor_kappa + stress_strain_tensor_mu)
                    * strain_tensor;

    stress_strain_tensor = stress_strain_tensor_mu;
    stress_strain_tensor_linearized = stress_strain_tensor_mu;

    SymmetricTensor<2, dim> deviator_stress_tensor = deviator(stress_tensor);
    const double deviator_stress_tensor_norm = deviator_stress_tensor.norm();
    const double von_Mises_stress = Evaluation::get_von_Mises_stress(stress_tensor);

    if (von_Mises_stress > sigma_0)
      {
        const double beta = sigma_0 / von_Mises_stress;
        stress_strain_tensor *= (gamma + (1 - gamma) * beta);
        stress_strain_tensor_linearized *= (gamma + (1 - gamma) * beta);
        deviator_stress_tensor /= deviator_stress_tensor_norm;
        stress_strain_tensor_linearized -= (1 - gamma) * beta * 2 * mu
                                           * outer_product(deviator_stress_tensor,
                                                           deviator_stress_tensor);
      }

    stress_strain_tensor += stress_strain_tensor_kappa;
    stress_strain_tensor_linearized += stress_strain_tensor_kappa;
  }

  // Finally, below we will need a function that computes the rotation matrix
  // induced by a displacement at a given point. In fact, of course, the
  // displacement at a single point only has a direction and a magnitude, it
  // is the change in direction and magnitude that induces rotations. In
  // effect, the rotation matrix can be computed from the gradients of a
  // displacement, or, more specifically, from the curl.
  //
  // The formulas by which the rotation matrices are determined are a little
  // awkward, especially in 3d. For 2d, there is a simpler way, so we
  // implement this function twice, once for 2d and once for 3d, so that we
  // can compile and use the program in both space dimensions if so desired --
  // after all, deal.II is all about dimension independent programming and
  // reuse of algorithm thoroughly tested with cheap computations in 2d, for
  // the more expensive computations in 3d. Here is one case, where we have to
  // implement different algorithms for 2d and 3d, but then can write the rest
  // of the program in a way that is independent of the space dimension.
  //
  // So, without further ado to the 2d implementation:
  Tensor<2,2>
  get_rotation_matrix (const std::vector<Tensor<1,2> > &grad_u)
  {
    // First, compute the curl of the velocity field from the gradients. Note
    // that we are in 2d, so the rotation is a scalar:
    const double curl = (grad_u[1][0] - grad_u[0][1]);

    // From this, compute the angle of rotation:
    const double angle = std::atan (curl);

    // And from this, build the antisymmetric rotation matrix:
    const double t[2][2] = {{ cos(angle), sin(angle) },
      {-sin(angle), cos(angle) }
    };
    return Tensor<2,2>(t);
  }


  // The 3d case is a little more contrived:
  Tensor<2,3>
  get_rotation_matrix (const std::vector<Tensor<1,3> > &grad_u)
  {
    // Again first compute the curl of the velocity field. This time, it is a
    // real vector:
    const Point<3> curl (grad_u[2][1] - grad_u[1][2],
                         grad_u[0][2] - grad_u[2][0],
                         grad_u[1][0] - grad_u[0][1]);

    // From this vector, using its magnitude, compute the tangent of the angle
    // of rotation, and from it the actual angle:
    const double tan_angle = std::sqrt(curl*curl);
    const double angle = std::atan (tan_angle);

    // Now, here's one problem: if the angle of rotation is too small, that
    // means that there is no rotation going on (for example a translational
    // motion). In that case, the rotation matrix is the identity matrix.
    //
    // The reason why we stress that is that in this case we have that
    // <code>tan_angle==0</code>. Further down, we need to divide by that
    // number in the computation of the axis of rotation, and we would get
    // into trouble when dividing doing so. Therefore, let's shortcut this and
    // simply return the identity matrix if the angle of rotation is really
    // small:
    if (angle < 1e-9)
      {
        static const double rotation[3][3]
        = {{ 1, 0, 0}, { 0, 1, 0 }, { 0, 0, 1 } };
        static const Tensor<2,3> rot(rotation);
        return rot;
      }

    // Otherwise compute the real rotation matrix. The algorithm for this is
    // not exactly obvious, but can be found in a number of books,
    // particularly on computer games where rotation is a very frequent
    // operation. Online, you can find a description at
    // http://www.makegames.com/3drotation/ and (this particular form, with
    // the signs as here) at
    // http://www.gamedev.net/reference/articles/article1199.asp:
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double t = 1-c;

    const Point<3> axis = curl/tan_angle;
    const double rotation[3][3]
    = {{
        t *axis[0] *axis[0]+c,
        t *axis[0] *axis[1]+s *axis[2],
        t *axis[0] *axis[2]-s *axis[1]
      },
      {
        t *axis[0] *axis[1]-s *axis[2],
        t *axis[1] *axis[1]+c,
        t *axis[1] *axis[2]+s *axis[0]
      },
      {
        t *axis[0] *axis[2]+s *axis[1],
        t *axis[1] *axis[1]-s *axis[0],
        t *axis[2] *axis[2]+c
      }
    };
    return Tensor<2,3>(rotation);
  }


  // <h3>Equation data: Body forces, boundary forces,
  // incremental boundary values</h3>
  //
  // The following should be relatively standard. We need classes for
  // the boundary forcing term (which we here choose to be zero)
  // and incremental boundary values.
  namespace EquationData
  {

    /*
    template <int dim>
    class BoundaryForce : public Function<dim>
    {
    public:
      BoundaryForce ();

      virtual
      double value (const Point<dim> &p,
                    const unsigned int component = 0) const;

      virtual
      void vector_value (const Point<dim> &p,
                         Vector<double> &values) const;
    };

    template <int dim>
    BoundaryForce<dim>::BoundaryForce ()
    :
    Function<dim>(dim)
    {}


    template <int dim>
    double
    BoundaryForce<dim>::value (const Point<dim> &,
                               const unsigned int) const
    {
      return 0.;
    }

    template <int dim>
    void
    BoundaryForce<dim>::vector_value (const Point<dim> &p,
                                      Vector<double> &values) const
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = BoundaryForce<dim>::value(p, c);
    }

    // @sect3{The <code>BodyForce</code> class}
    // Body forces are generally mediated by one of the four basic
    // physical types of forces:
    // gravity, strong and weak interaction, and electromagnetism. Unless one
    // wants to consider subatomic objects (for which quasistatic deformation is
    // irrelevant and an inappropriate description anyway), only gravity and
    // electromagnetic forces need to be considered. Let us, for simplicity
    // assume that our body has a certain mass density, but is either
    // non-magnetic and not electrically conducting or that there are no
    // significant electromagnetic fields around. In that case, the body forces
    // are simply <code>rho g</code>, where <code>rho</code> is the material
    // density and <code>g</code> is a vector in negative z-direction with
    // magnitude 9.81 m/s^2.  Both the density and <code>g</code> are defined in
    // the function, and we take as the density 7700 kg/m^3, a value commonly
    // assumed for steel.
    //
    // To be a little more general and to be able to do computations in 2d as
    // well, we realize that the body force is always a function returning a
    // <code>dim</code> dimensional vector. We assume that gravity acts along
    // the negative direction of the last, i.e. <code>dim-1</code>th
    // coordinate. The rest of the implementation of this function should be
    // mostly self-explanatory given similar definitions in previous example
    // programs. Note that the body force is independent of the location; to
    // avoid compiler warnings about unused function arguments, we therefore
    // comment out the name of the first argument of the
    // <code>vector_value</code> function:
    template <int dim>
    class BodyForce :  public Function<dim>
    {
    public:
      BodyForce ();

      virtual
      void
      vector_value (const Point<dim> &p,
                    Vector<double>   &values) const;

      virtual
      void
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const;
    };


    template <int dim>
    BodyForce<dim>::BodyForce ()
    :
    Function<dim> (dim)
    {}


    template <int dim>
    inline
    void
    BodyForce<dim>::vector_value (const Point<dim> &p,
                                  Vector<double>   &values) const
    {
      Assert (values.size() == dim,
              ExcDimensionMismatch (values.size(), dim));

      const double g   = 9.81;
      const double rho = 7700;

      values = 0;
      values(dim-1) = -rho * g;
    }



    template <int dim>
    void
    BodyForce<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                       std::vector<Vector<double> >   &value_list) const
    {
      const unsigned int n_points = points.size();

      Assert (value_list.size() == n_points,
              ExcDimensionMismatch (value_list.size(), n_points));

      for (unsigned int p=0; p<n_points; ++p)
        BodyForce<dim>::vector_value (points[p],
                                      value_list[p]);
    }

    // @sect3{The <code>IncrementalBoundaryValue</code> class}

    // In addition to body forces, movement can be induced by boundary forces
    // and forced boundary displacement. The latter case is equivalent to forces
    // being chosen in such a way that they induce certain displacement.
    //
    // For quasistatic displacement, typical boundary forces would be pressure
    // on a body, or tangential friction against another body. We chose a
    // somewhat simpler case here: we prescribe a certain movement of (parts of)
    // the boundary, or at least of certain components of the displacement
    // vector. We describe this by another vector-valued function that, for a
    // given point on the boundary, returns the prescribed displacement.
    //
    // Since we have a time-dependent problem, the displacement increment of the
    // boundary equals the displacement accumulated during the length of the
    // timestep. The class therefore has to know both the present time and the
    // length of the present time step, and can then approximate the incremental
    // displacement as the present velocity times the present timestep.
    //
    // For the purposes of this program, we choose a simple form of boundary
    // displacement: we displace the top boundary with constant velocity
    // downwards. The rest of the boundary is either going to be fixed (and is
    // then described using an object of type <code>ZeroFunction</code>) or free
    // (Neumann-type, in which case nothing special has to be done).  The
    // implementation of the class describing the constant downward motion
    // should then be obvious using the knowledge we gained through all the
    // previous example programs:
    template <int dim>
    class IncrementalBoundaryValues :  public Function<dim>
    {
    public:
      IncrementalBoundaryValues (const double present_time,
                                 const double present_timestep);

      virtual
      void
      vector_value (const Point<dim> &p,
                    Vector<double>   &values) const;

      virtual
      void
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const;

    private:
      const double velocity;
      const double present_time;
      const double present_timestep;
    };


    template <int dim>
    IncrementalBoundaryValues<dim>::
    IncrementalBoundaryValues (const double present_time,
                               const double present_timestep)
    :
    Function<dim> (dim),
    velocity (.1),
    present_time (present_time),
    present_timestep (present_timestep)
    {}


    template <int dim>
    void
    IncrementalBoundaryValues<dim>::
    vector_value (const Point<dim> &p,
                  Vector<double>   &values) const
    {
      Assert (values.size() == dim,
              ExcDimensionMismatch (values.size(), dim));

      values = 0;
      values(2) = -present_timestep * velocity;
    }



    template <int dim>
    void
    IncrementalBoundaryValues<dim>::
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> >   &value_list) const
    {
      const unsigned int n_points = points.size();

      Assert (value_list.size() == n_points,
              ExcDimensionMismatch (value_list.size(), n_points));

      for (unsigned int p=0; p<n_points; ++p)
        IncrementalBoundaryValues<dim>::vector_value (points[p],
            value_list[p]);
    }
    */

    // ----------------------------- TimoshenkoBeam ---------------------------------------
    /*
    template <int dim>
    class IncrementalBoundaryForce : public Function<dim>
    {
    public:
      IncrementalBoundaryForce (const double present_time,
                                const double end_time);

      virtual
      void vector_value (const Point<dim> &p,
                         Vector<double> &values) const;

      virtual
      void
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const;
    private:
      const double present_time,
                   end_time,
                   shear_force,
                   length,
                   depth,
                   thickness;
    };

    template <int dim>
    IncrementalBoundaryForce<dim>::
    IncrementalBoundaryForce (const double present_time,
                                const double end_time)
    :
    Function<dim>(dim),
    present_time (present_time),
    end_time (end_time),
    shear_force (2e4),
    length (.48),
    depth (.12),
    thickness (.01)
    {}

    template <int dim>
    void
    IncrementalBoundaryForce<dim>::vector_value (const Point<dim> &p,
                                                 Vector<double> &values) const
    {
      AssertThrow (values.size() == dim,
          ExcDimensionMismatch (values.size(), dim));
      AssertThrow (dim == 2, ExcNotImplemented());

      // compute traction on the right face of Timoshenko beam problem, t_bar
      double inertia_moment = (thickness*std::pow(depth,3)) / 12;

      double x = p(0);
      double y = p(1);

      AssertThrow(std::fabs(x-length)<1e-12, ExcNotImplemented());

      values(0) = 0;
      values(1) = - shear_force/(2*inertia_moment) * ( depth*depth/4-y*y );

      // compute the fraction of imposed force
      const double frac = present_time/end_time;

      values *= frac;
    }

    template <int dim>
    void
    IncrementalBoundaryForce<dim>::
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> >   &value_list) const
    {
      const unsigned int n_points = points.size();

      Assert (value_list.size() == n_points,
              ExcDimensionMismatch (value_list.size(), n_points));

      for (unsigned int p=0; p<n_points; ++p)
        IncrementalBoundaryForce<dim>::vector_value (points[p],
            value_list[p]);
    }


    template <int dim>
    class BodyForce :  public ZeroFunction<dim>
    {
    public:
      BodyForce () : ZeroFunction<dim> (dim) {}
    };

    template <int dim>
    class IncrementalBoundaryValues :  public Function<dim>
    {
    public:
      IncrementalBoundaryValues (const double present_time,
                                 const double end_time);

      virtual
      void
      vector_value (const Point<dim> &p,
                    Vector<double>   &values) const;

      virtual
      void
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const;

    private:
      const double present_time,
                   end_time,
                   shear_force,
                   Youngs_modulus,
                   Poissons_ratio,
                   length,
                   depth,
                   thickness;
    };


    template <int dim>
    IncrementalBoundaryValues<dim>::
    IncrementalBoundaryValues (const double present_time,
                               const double end_time)
    :
    Function<dim> (dim),
    present_time (present_time),
    end_time (end_time),
    shear_force (2e4),
    Youngs_modulus (2.e11),
    Poissons_ratio (.3),
    length (.48),
    depth (.12),
    thickness (.01)
    {}


    template <int dim>
    void
    IncrementalBoundaryValues<dim>::
    vector_value (const Point<dim> &p,
                  Vector<double>   &values) const
    {
      AssertThrow (values.size() == dim,
                   ExcDimensionMismatch (values.size(), dim));
      AssertThrow (dim == 2, ExcNotImplemented());


      // compute exact displacement of Timoshenko beam problem, u_bar
      double inertia_moment = (thickness*std::pow(depth,3)) / 12;

      double x = p(0);
      double y = p(1);

      double fac = shear_force / (6*Youngs_modulus*inertia_moment);

      values(0) =  fac * y * ( (6*length-3*x)*x + (2+Poissons_ratio)*(y*y-depth*depth/4) );
      values(1) = -fac* ( 3*Poissons_ratio*y*y*(length-x) + 0.25*(4+5*Poissons_ratio)*depth*depth*x + (3*length-x)*x*x );

      // compute the fraction of imposed force
      const double frac = present_time/end_time;

      values *= frac;
    }



    template <int dim>
    void
    IncrementalBoundaryValues<dim>::
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> >   &value_list) const
    {
      const unsigned int n_points = points.size();

      Assert (value_list.size() == n_points,
              ExcDimensionMismatch (value_list.size(), n_points));

      for (unsigned int p=0; p<n_points; ++p)
        IncrementalBoundaryValues<dim>::vector_value (points[p],
            value_list[p]);
    }
    */

    // ------------------------- Thick_tube_internal_pressure ----------------------------------
    /*
    template <int dim>
      class IncrementalBoundaryForce : public Function<dim>
      {
      public:
        IncrementalBoundaryForce (const double present_time,
                                  const double end_time);

        virtual
        void vector_value (const Point<dim> &p,
                           Vector<double> &values) const;

        virtual
        void
        vector_value_list (const std::vector<Point<dim> > &points,
                           std::vector<Vector<double> >   &value_list) const;
      private:
        const double present_time,
                     end_time,
                     pressure,
                     inner_radius;
      };

      template <int dim>
      IncrementalBoundaryForce<dim>::
      IncrementalBoundaryForce (const double present_time,
                                const double end_time)
      :
      Function<dim>(dim),
      present_time (present_time),
      end_time (end_time),
      pressure (0.6*2.4e8),
    //    pressure (1.94e8),
      inner_radius(.1)
      {}

      template <int dim>
      void
      IncrementalBoundaryForce<dim>::vector_value (const Point<dim> &p,
                                                   Vector<double> &values) const
      {
        AssertThrow (dim == 2, ExcNotImplemented());
        AssertThrow (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));

        const double eps = 1.e-7 * inner_radius,
                     radius = p.norm();
        // compute traction on the inner boundary, t_bar
        AssertThrow(radius < (eps+inner_radius), ExcInternalError());

        const double theta = std::atan2(p(1),p(0));

        values(0) = pressure * std::cos(theta);
        values(1) = pressure * std::sin(theta);

        // compute the fraction of imposed force
        const double frac = present_time/end_time;

        values *= frac;
      }

      template <int dim>
      void
      IncrementalBoundaryForce<dim>::
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const
      {
        const unsigned int n_points = points.size();

        Assert (value_list.size() == n_points,
                ExcDimensionMismatch (value_list.size(), n_points));

        for (unsigned int p=0; p<n_points; ++p)
          IncrementalBoundaryForce<dim>::vector_value (points[p],
              value_list[p]);
      }


      template <int dim>
      class BodyForce :  public ZeroFunction<dim>
      {
      public:
        BodyForce () : ZeroFunction<dim> (dim) {}
      };


      template <int dim>
      class IncrementalBoundaryValues :  public Function<dim>
      {
      public:
        IncrementalBoundaryValues (const double present_time,
                                   const double end_time);

        virtual
        void
        vector_value (const Point<dim> &p,
                      Vector<double>   &values) const;

        virtual
        void
        vector_value_list (const std::vector<Point<dim> > &points,
                           std::vector<Vector<double> >   &value_list) const;

      private:
        const double present_time,
                     end_time;
      };


      template <int dim>
      IncrementalBoundaryValues<dim>::
      IncrementalBoundaryValues (const double present_time,
                                 const double end_time)
      :
      Function<dim> (dim),
      present_time (present_time),
      end_time (end_time)
      {}


      template <int dim>
      void
      IncrementalBoundaryValues<dim>::
      vector_value (const Point<dim> &p,
                    Vector<double>   &values) const
      {
        AssertThrow (values.size() == dim,
                     ExcDimensionMismatch (values.size(), dim));
        AssertThrow (dim == 2, ExcNotImplemented());

        values = 0.;
      }



      template <int dim>
      void
      IncrementalBoundaryValues<dim>::
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const
      {
        const unsigned int n_points = points.size();

        Assert (value_list.size() == n_points,
                ExcDimensionMismatch (value_list.size(), n_points));

        for (unsigned int p=0; p<n_points; ++p)
          IncrementalBoundaryValues<dim>::vector_value (points[p],
              value_list[p]);
      }
      */

    // ------------------------- Perforated_strip_tension ----------------------------------
    /*
    template <int dim>
    class IncrementalBoundaryForce : public Function<dim>
    {
    public:
      IncrementalBoundaryForce (const double present_time,
                                const double end_time);

      virtual
      void vector_value (const Point<dim> &p,
                         Vector<double> &values) const;

      virtual
      void
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const;
    private:
      const double present_time,
                   end_time;
    };

    template <int dim>
    IncrementalBoundaryForce<dim>::
    IncrementalBoundaryForce (const double present_time,
                              const double end_time)
    :
    Function<dim>(dim),
    present_time (present_time),
    end_time (end_time)
    {}

    template <int dim>
    void
    IncrementalBoundaryForce<dim>::vector_value (const Point<dim> &p,
                                                 Vector<double> &values) const
    {
      AssertThrow (values.size() == dim,
                   ExcDimensionMismatch (values.size(), dim));

      values = 0;

      // compute the fraction of imposed force
      const double frac = present_time/end_time;

      values *= frac;
    }

    template <int dim>
    void
    IncrementalBoundaryForce<dim>::
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> >   &value_list) const
    {
      const unsigned int n_points = points.size();

      Assert (value_list.size() == n_points,
              ExcDimensionMismatch (value_list.size(), n_points));

      for (unsigned int p=0; p<n_points; ++p)
        IncrementalBoundaryForce<dim>::vector_value (points[p],
            value_list[p]);
    }


    template <int dim>
    class BodyForce :  public ZeroFunction<dim>
    {
    public:
      BodyForce () : ZeroFunction<dim> (dim) {}
    };


    template <int dim>
    class IncrementalBoundaryValues :  public Function<dim>
    {
    public:
      IncrementalBoundaryValues (const double present_time,
                                 const double end_time);

      virtual
      void
      vector_value (const Point<dim> &p,
                    Vector<double>   &values) const;

      virtual
      void
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const;

    private:
      const double present_time,
                   end_time,
                   imposed_displacement,
                   height;
    };


    template <int dim>
    IncrementalBoundaryValues<dim>::
    IncrementalBoundaryValues (const double present_time,
                               const double end_time)
    :
    Function<dim> (dim),
    present_time (present_time),
    end_time (end_time),
    imposed_displacement (0.00055),
    height (0.18)
    {}


    template <int dim>
    void
    IncrementalBoundaryValues<dim>::
    vector_value (const Point<dim> &p,
                  Vector<double>   &values) const
    {
      AssertThrow (values.size() == dim,
                   ExcDimensionMismatch (values.size(), dim));

      const double eps = 1.e-8 * height;

      values = 0.;

      // impose displacement only on the top edge
      if (std::abs(p[1]-height) < eps)
      {
        // compute the fraction of imposed displacement
        const double inc_frac = 1/end_time;

        values(1) = inc_frac*imposed_displacement;
      }

    }



    template <int dim>
    void
    IncrementalBoundaryValues<dim>::
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> >   &value_list) const
    {
      const unsigned int n_points = points.size();

      Assert (value_list.size() == n_points,
              ExcDimensionMismatch (value_list.size(), n_points));

      for (unsigned int p=0; p<n_points; ++p)
        IncrementalBoundaryValues<dim>::vector_value (points[p],
            value_list[p]);
    }
    */

    // ------------------------- Cantiliver_beam_3d ----------------------------------
    template <int dim>
    class IncrementalBoundaryForce : public Function<dim>
    {
    public:
      IncrementalBoundaryForce (const double present_time,
                                const double end_time);

      virtual
      void vector_value (const Point<dim> &p,
                         Vector<double> &values) const;

      virtual
      void
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const;

    private:
      const double present_time,
            end_time,
            pressure,
            height;
    };

    template <int dim>
    IncrementalBoundaryForce<dim>::
    IncrementalBoundaryForce (const double present_time,
                              const double end_time)
      :
      Function<dim>(dim),
      present_time (present_time),
      end_time (end_time),
      pressure (6e6),
      height (200e-3)
    {}

    template <int dim>
    void
    IncrementalBoundaryForce<dim>::vector_value (const Point<dim> &p,
                                                 Vector<double> &values) const
    {
      AssertThrow (dim == 3, ExcNotImplemented());
      AssertThrow (values.size() == dim,
                   ExcDimensionMismatch (values.size(), dim));

      const double eps = 1.e-7 * height;

      // pressure should be imposed on the top surface, y = height
      AssertThrow(std::abs(p[1]-(height/2)) < eps, ExcInternalError());

      values = 0;

      values(1) = -pressure;

      // compute the fraction of imposed force
      const double frac = present_time/end_time;

      values *= frac;
    }

    template <int dim>
    void
    IncrementalBoundaryForce<dim>::
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> >   &value_list) const
    {
      const unsigned int n_points = points.size();

      Assert (value_list.size() == n_points,
              ExcDimensionMismatch (value_list.size(), n_points));

      for (unsigned int p=0; p<n_points; ++p)
        IncrementalBoundaryForce<dim>::vector_value (points[p], value_list[p]);
    }


    template <int dim>
    class BodyForce :  public ZeroFunction<dim>
    {
    public:
      BodyForce () : ZeroFunction<dim> (dim) {}
    };


    template <int dim>
    class IncrementalBoundaryValues :  public Function<dim>
    {
    public:
      IncrementalBoundaryValues (const double present_time,
                                 const double end_time);

      virtual
      void
      vector_value (const Point<dim> &p,
                    Vector<double>   &values) const;

      virtual
      void
      vector_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &value_list) const;

    private:
      const double present_time,
            end_time;
    };


    template <int dim>
    IncrementalBoundaryValues<dim>::
    IncrementalBoundaryValues (const double present_time,
                               const double end_time)
      :
      Function<dim> (dim),
      present_time (present_time),
      end_time (end_time)
    {}


    template <int dim>
    void
    IncrementalBoundaryValues<dim>::
    vector_value (const Point<dim> &p,
                  Vector<double>   &values) const
    {
      AssertThrow (values.size() == dim,
                   ExcDimensionMismatch (values.size(), dim));
      AssertThrow (dim == 3, ExcNotImplemented());

      values = 0.;
    }


    template <int dim>
    void
    IncrementalBoundaryValues<dim>::
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> >   &value_list) const
    {
      const unsigned int n_points = points.size();

      Assert (value_list.size() == n_points,
              ExcDimensionMismatch (value_list.size(), n_points));

      for (unsigned int p=0; p<n_points; ++p)
        IncrementalBoundaryValues<dim>::vector_value (points[p], value_list[p]);
    }

    // -------------------------------------------------------------------------------
  }


  namespace DualFunctional
  {

    template <int dim>
    class DualFunctionalBase : public Subscriptor
    {
    public:
      virtual
      void
      assemble_rhs (const DoFHandler<dim>      &dof_handler,
                    const Vector<double>       &solution,
                    const ConstitutiveLaw<dim> &constitutive_law,
                    const DoFHandler<dim>      &dof_handler_dual,
                    Vector<double>             &rhs_dual) const = 0;
    };


    template <int dim>
    class PointValuesEvaluation : public DualFunctionalBase<dim>
    {
    public:
      PointValuesEvaluation (const Point<dim> &evaluation_point);

      virtual
      void
      assemble_rhs (const DoFHandler<dim>      &dof_handler,
                    const Vector<double>       &solution,
                    const ConstitutiveLaw<dim> &constitutive_law,
                    const DoFHandler<dim>      &dof_handler_dual,
                    Vector<double>             &rhs_dual) const;

      DeclException1 (ExcEvaluationPointNotFound,
                      Point<dim>,
                      << "The evaluation point " << arg1
                      << " was not found among the vertices of the present grid.");

    protected:
      const Point<dim> evaluation_point;
    };


    template <int dim>
    PointValuesEvaluation<dim>::
    PointValuesEvaluation (const Point<dim> &evaluation_point)
      :
      evaluation_point (evaluation_point)
    {}


    template <int dim>
    void
    PointValuesEvaluation<dim>::
    assemble_rhs (const DoFHandler<dim>      &dof_handler,
                  const Vector<double>       &solution,
                  const ConstitutiveLaw<dim> &constitutive_law,
                  const DoFHandler<dim>      &dof_handler_dual,
                  Vector<double>             &rhs_dual) const
    {
      rhs_dual.reinit (dof_handler_dual.n_dofs());
      const unsigned int dofs_per_vertex = dof_handler_dual.get_fe().dofs_per_vertex;

      typename DoFHandler<dim>::active_cell_iterator
      cell_dual = dof_handler_dual.begin_active(),
      endc_dual = dof_handler_dual.end();
      for (; cell_dual!=endc_dual; ++cell_dual)
        for (unsigned int vertex=0;
             vertex<GeometryInfo<dim>::vertices_per_cell;
             ++vertex)
          if (cell_dual->vertex(vertex).distance(evaluation_point)
              < cell_dual->diameter()*1e-8)
            {
              for (unsigned int id=0; id!=dofs_per_vertex; ++id)
                {
                  rhs_dual(cell_dual->vertex_dof_index(vertex,id)) = 1;
                }
              return;
            }

      AssertThrow (false, ExcEvaluationPointNotFound(evaluation_point));
    }


    template <int dim>
    class PointXDerivativesEvaluation : public DualFunctionalBase<dim>
    {
    public:
      PointXDerivativesEvaluation (const Point<dim> &evaluation_point);

      virtual
      void
      assemble_rhs (const DoFHandler<dim>      &dof_handler,
                    const Vector<double>       &solution,
                    const ConstitutiveLaw<dim> &constitutive_law,
                    const DoFHandler<dim>      &dof_handler_dual,
                    Vector<double>             &rhs_dual) const;

      DeclException1 (ExcEvaluationPointNotFound,
                      Point<dim>,
                      << "The evaluation point " << arg1
                      << " was not found among the vertices of the present grid.");

    protected:
      const Point<dim> evaluation_point;
    };


    template <int dim>
    PointXDerivativesEvaluation<dim>::
    PointXDerivativesEvaluation (const Point<dim> &evaluation_point)
      :
      evaluation_point (evaluation_point)
    {}


    template <int dim>
    void
    PointXDerivativesEvaluation<dim>::
    assemble_rhs (const DoFHandler<dim>      &dof_handler,
                  const Vector<double>       &solution,
                  const ConstitutiveLaw<dim> &constitutive_law,
                  const DoFHandler<dim>      &dof_handler_dual,
                  Vector<double>             &rhs_dual) const
    {
      rhs_dual.reinit (dof_handler_dual.n_dofs());
      const unsigned int dofs_per_vertex = dof_handler_dual.get_fe().dofs_per_vertex;

      QGauss<dim> quadrature(4);
      FEValues<dim>  fe_values (dof_handler_dual.get_fe(), quadrature,
                                update_gradients |
                                update_quadrature_points  |
                                update_JxW_values);
      const unsigned int n_q_points = fe_values.n_quadrature_points;
      Assert ( n_q_points==quadrature.size() , ExcInternalError() );
      const unsigned int dofs_per_cell = dof_handler_dual.get_fe().dofs_per_cell;

      Vector<double> cell_rhs (dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      double total_volume = 0;

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_dual.begin_active(),
      endc = dof_handler_dual.end();
      for (; cell!=endc; ++cell)
        if (cell->center().distance(evaluation_point) <=
            cell->diameter())
          {
            fe_values.reinit (cell);
            cell_rhs = 0;

            for (unsigned int q=0; q<n_q_points; ++q)
              {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  {
                    const unsigned int
                    component_i = dof_handler_dual.get_fe().system_to_component_index(i).first;

                    cell_rhs(i) += fe_values.shape_grad(i,q)[0] *
                                   fe_values.JxW (q);
                  }

                total_volume += fe_values.JxW (q);
              }

            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                rhs_dual(local_dof_indices[i]) += cell_rhs(i);
              }
          }

      AssertThrow (total_volume > 0,
                   ExcEvaluationPointNotFound(evaluation_point));

      rhs_dual.scale (1./total_volume);
    }



    template <int dim>
    class MeanDisplacementFace : public DualFunctionalBase<dim>
    {
    public:
      MeanDisplacementFace (const unsigned int face_id,
                            const std::vector<bool> comp_mask);

      virtual
      void
      assemble_rhs (const DoFHandler<dim>      &dof_handler,
                    const Vector<double>       &solution,
                    const ConstitutiveLaw<dim> &constitutive_law,
                    const DoFHandler<dim>      &dof_handler_dual,
                    Vector<double>             &rhs_dual) const;

    protected:
      const unsigned int face_id;
      const std::vector<bool> comp_mask;
    };


    template <int dim>
    MeanDisplacementFace<dim>::
    MeanDisplacementFace (const unsigned int face_id,
                          const std::vector<bool> comp_mask )
      :
      face_id (face_id),
      comp_mask (comp_mask)
    {
      AssertThrow(comp_mask.size() == dim,
                  ExcDimensionMismatch (comp_mask.size(), dim) );
    }


    template <int dim>
    void
    MeanDisplacementFace<dim>::
    assemble_rhs (const DoFHandler<dim>      &dof_handler,
                  const Vector<double>       &solution,
                  const ConstitutiveLaw<dim> &constitutive_law,
                  const DoFHandler<dim>      &dof_handler_dual,
                  Vector<double>             &rhs_dual) const
    {
      AssertThrow (dim >= 2, ExcNotImplemented());

      rhs_dual.reinit (dof_handler_dual.n_dofs());

      const QGauss<dim-1> face_quadrature(dof_handler_dual.get_fe().tensor_degree()+1);
      FEFaceValues<dim> fe_face_values (dof_handler_dual.get_fe(), face_quadrature,
                                        update_values | update_JxW_values);

      const unsigned int  dofs_per_vertex = dof_handler_dual.get_fe().dofs_per_vertex;
      const unsigned int  dofs_per_cell = dof_handler_dual.get_fe().dofs_per_cell;
      const unsigned int  n_face_q_points = face_quadrature.size();

      AssertThrow(dofs_per_vertex == dim,
                  ExcDimensionMismatch (dofs_per_vertex, dim) );

      std::vector<unsigned int> comp_vector(dofs_per_vertex);
      for (unsigned int i=0; i!=dofs_per_vertex; ++i)
        {
          if (comp_mask[i])
            {
              comp_vector[i] = 1;
            }
        }

      Vector<double>       cell_rhs (dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      // bound_size : size of the boundary, in 2d is the length
      //              and in the 3d case, area
      double bound_size = 0.;

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_dual.begin_active(),
      endc = dof_handler_dual.end();
      bool evaluation_face_found = false;
      for (; cell!=endc; ++cell)
        {
          cell_rhs = 0;
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->face(face)->at_boundary()
                  &&
                  cell->face(face)->boundary_indicator() == face_id)
                {
                  if (!evaluation_face_found)
                    {
                      evaluation_face_found = true;
                    }
                  fe_face_values.reinit (cell, face);

                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                      bound_size += fe_face_values.JxW(q_point);

                      for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                          const unsigned int
                          component_i = dof_handler_dual.get_fe().system_to_component_index(i).first;

                          cell_rhs(i) += (fe_face_values.shape_value(i,q_point) *
                                          comp_vector[component_i] *
                                          fe_face_values.JxW(q_point));
                        }

                    }

                }
            }

          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              rhs_dual(local_dof_indices[i]) += cell_rhs(i);
            }

        }

      AssertThrow(evaluation_face_found, ExcInternalError());

      rhs_dual /= bound_size;
    }



    template <int dim>
    class MeanStressFace : public DualFunctionalBase<dim>
    {
    public:
      MeanStressFace (const unsigned int face_id,
                      const std::vector<std::vector<unsigned int> > &comp_stress);

      virtual
      void
      assemble_rhs (const DoFHandler<dim>      &dof_handler,
                    const Vector<double>       &solution,
                    const ConstitutiveLaw<dim> &constitutive_law,
                    const DoFHandler<dim>      &dof_handler_dual,
                    Vector<double>             &rhs_dual) const;

    protected:
      const unsigned int face_id;
      const std::vector<std::vector<unsigned int> >  comp_stress;
    };


    template <int dim>
    MeanStressFace<dim>::
    MeanStressFace (const unsigned int face_id,
                    const std::vector<std::vector<unsigned int> > &comp_stress )
      :
      face_id (face_id),
      comp_stress (comp_stress)
    {
      AssertThrow(comp_stress.size() == dim,
                  ExcDimensionMismatch (comp_stress.size(), dim) );
    }


    template <int dim>
    void
    MeanStressFace<dim>::
    assemble_rhs (const DoFHandler<dim>      &dof_handler,
                  const Vector<double>       &solution,
                  const ConstitutiveLaw<dim> &constitutive_law,
                  const DoFHandler<dim>      &dof_handler_dual,
                  Vector<double>             &rhs_dual) const
    {
      AssertThrow (dim >= 2, ExcNotImplemented());

      rhs_dual.reinit (dof_handler_dual.n_dofs());

      const QGauss<dim-1> face_quadrature(dof_handler_dual.get_fe().tensor_degree()+1);

      FEFaceValues<dim> fe_face_values (dof_handler.get_fe(), face_quadrature,
                                        update_gradients);
      FEFaceValues<dim> fe_face_values_dual (dof_handler_dual.get_fe(), face_quadrature,
                                             update_gradients | update_JxW_values);

      const unsigned int  dofs_per_cell_dual = dof_handler_dual.get_fe().dofs_per_cell;
      const unsigned int  n_face_q_points = face_quadrature.size();

      std::vector<SymmetricTensor<2, dim> > strain_tensor(n_face_q_points);
      SymmetricTensor<4, dim> stress_strain_tensor;

      Vector<double>      cell_rhs (dofs_per_cell_dual);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell_dual);

      // bound_size : size of the boundary, in 2d is the length
      //              and in the 3d case, area
      double bound_size = 0.;

      bool evaluation_face_found = false;

      typename DoFHandler<dim>::active_cell_iterator
      cell_dual = dof_handler_dual.begin_active(),
      endc_dual = dof_handler_dual.end(),
      cell = dof_handler.begin_active();

      const FEValuesExtractors::Vector displacement(0);

      for (; cell_dual!=endc_dual; ++cell_dual, ++cell)
        {
          cell_rhs = 0;
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell_dual->face(face)->at_boundary()
                  &&
                  cell_dual->face(face)->boundary_indicator() == face_id)
                {
                  if (!evaluation_face_found)
                    {
                      evaluation_face_found = true;
                    }

                  fe_face_values.reinit (cell, face);
                  fe_face_values_dual.reinit (cell_dual, face);

                  fe_face_values[displacement].get_function_symmetric_gradients(solution,
                      strain_tensor);

                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                      bound_size += fe_face_values_dual.JxW(q_point);

                      constitutive_law.get_stress_strain_tensor(strain_tensor[q_point],
                                                                stress_strain_tensor);

                      for (unsigned int i=0; i<dofs_per_cell_dual; ++i)
                        {
                          const SymmetricTensor<2, dim>
                          stress_phi_i = stress_strain_tensor
                                         * fe_face_values_dual[displacement].symmetric_gradient(i, q_point);

                          for (unsigned int k=0; k!=dim; ++k)
                            {
                              for (unsigned int l=0; l!=dim; ++l)
                                {
                                  if ( comp_stress[k][l] == 1 )
                                    {
                                      cell_rhs(i) += stress_phi_i[k][l]
                                                     *
                                                     fe_face_values_dual.JxW(q_point);
                                    }

                                }
                            }

                        }

                    }

                }
            }

          cell_dual->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dofs_per_cell_dual; ++i)
            {
              rhs_dual(local_dof_indices[i]) += cell_rhs(i);
            }

        }

      AssertThrow(evaluation_face_found, ExcInternalError());

      rhs_dual /= bound_size;

    }


    template <int dim>
    class MeanStressDomain : public DualFunctionalBase<dim>
    {
    public:
      MeanStressDomain (const std::string &base_mesh,
                        const std::vector<std::vector<unsigned int> > &comp_stress);

      virtual
      void
      assemble_rhs (const DoFHandler<dim>      &dof_handler,
                    const Vector<double>       &solution,
                    const ConstitutiveLaw<dim> &constitutive_law,
                    const DoFHandler<dim>      &dof_handler_dual,
                    Vector<double>             &rhs_dual) const;

    protected:
      const std::string base_mesh;
      const std::vector<std::vector<unsigned int> >  comp_stress;
    };


    template <int dim>
    MeanStressDomain<dim>::
    MeanStressDomain (const std::string &base_mesh,
                      const std::vector<std::vector<unsigned int> > &comp_stress )
      :
      base_mesh (base_mesh),
      comp_stress (comp_stress)
    {
      AssertThrow(comp_stress.size() == dim,
                  ExcDimensionMismatch (comp_stress.size(), dim) );
    }


    template <int dim>
    void
    MeanStressDomain<dim>::
    assemble_rhs (const DoFHandler<dim>      &dof_handler,
                  const Vector<double>       &solution,
                  const ConstitutiveLaw<dim> &constitutive_law,
                  const DoFHandler<dim>      &dof_handler_dual,
                  Vector<double>             &rhs_dual) const
    {
      AssertThrow (base_mesh == "Cantiliver_beam_3d", ExcNotImplemented());
      AssertThrow (dim == 3, ExcNotImplemented());

      // Mean stress at the specified domain is of interest.
      // The interest domains are located on the bottom and top of the flanges
      // close to the clamped face, z = 0
      // top domain: height/2 - thickness_flange <= y <= height/2
      //             0 <= z <= 2 * thickness_flange
      // bottom domain: -height/2 <= y <= -height/2 + thickness_flange
      //             0 <= z <= 2 * thickness_flange

      const double height = 200e-3,
                   thickness_flange = 10e-3;

      rhs_dual.reinit (dof_handler_dual.n_dofs());

      const QGauss<dim> quadrature_formula(dof_handler_dual.get_fe().tensor_degree()+1);

      FEValues<dim> fe_values (dof_handler.get_fe(), quadrature_formula,
                               update_gradients);
      FEValues<dim> fe_values_dual (dof_handler_dual.get_fe(), quadrature_formula,
                                    update_gradients | update_JxW_values);

      const unsigned int  dofs_per_cell_dual = dof_handler_dual.get_fe().dofs_per_cell;
      const unsigned int  n_q_points = quadrature_formula.size();

      std::vector<SymmetricTensor<2, dim> > strain_tensor(n_q_points);
      SymmetricTensor<4, dim> stress_strain_tensor;

      Vector<double>      cell_rhs (dofs_per_cell_dual);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell_dual);

      // domain_size : size of the interested domain, in 2d is the area
      //              and in the 3d case, volume
      double domain_size = 0.;

      bool evaluation_domain_found = false;

      typename DoFHandler<dim>::active_cell_iterator
      cell_dual = dof_handler_dual.begin_active(),
      endc_dual = dof_handler_dual.end(),
      cell = dof_handler.begin_active();

      const FEValuesExtractors::Vector displacement(0);

      for (; cell_dual!=endc_dual; ++cell_dual, ++cell)
        {
          const double y = cell->center()[1],
                       z = cell->center()[2];
          // top domain: height/2 - thickness_flange <= y <= height/2
          //             0 <= z <= 2 * thickness_flange
          // bottom domain: -height/2 <= y <= -height/2 + thickness_flange
          //             0 <= z <= 2 * thickness_flange
          if ( ((z > 0) && (z < 2*thickness_flange)) &&
               ( ((y > height/2 - thickness_flange) && (y < height/2)) ||
                 ((y > -height/2) && (y < -height/2 + thickness_flange)) ) )
            {
              cell_rhs = 0;

              if (!evaluation_domain_found)
                {
                  evaluation_domain_found = true;
                }

              fe_values.reinit(cell);
              fe_values_dual.reinit(cell_dual);

              fe_values[displacement].get_function_symmetric_gradients(solution,
                                                                       strain_tensor);

              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                  domain_size += fe_values_dual.JxW(q_point);

                  constitutive_law.get_stress_strain_tensor(strain_tensor[q_point],
                                                            stress_strain_tensor);

                  for (unsigned int i=0; i<dofs_per_cell_dual; ++i)
                    {
                      const SymmetricTensor<2, dim>
                      stress_phi_i = stress_strain_tensor
                                     * fe_values_dual[displacement].symmetric_gradient(i, q_point);

                      for (unsigned int k=0; k!=dim; ++k)
                        {
                          for (unsigned int l=0; l!=dim; ++l)
                            {
                              if ( comp_stress[k][l] == 1 )
                                {
                                  cell_rhs(i) += stress_phi_i[k][l]
                                                 *
                                                 fe_values_dual.JxW(q_point);
                                }

                            }
                        }

                    }

                }

            }

          cell_dual->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dofs_per_cell_dual; ++i)
            {
              rhs_dual(local_dof_indices[i]) += cell_rhs(i);
            }

        }

      AssertThrow(evaluation_domain_found, ExcInternalError());

      rhs_dual /= domain_size;

    }


    template <int dim>
    class MeanStrainEnergyFace : public DualFunctionalBase<dim>
    {
    public:
      MeanStrainEnergyFace (const unsigned int face_id,
                            const Function<dim>      &lambda_function,
                            const Function<dim>      &mu_function );

      void assemble_rhs_nonlinear (const DoFHandler<dim> &primal_dof_handler,
                                   const Vector<double>  &primal_solution,
                                   const DoFHandler<dim> &dof_handler,
                                   Vector<double>        &rhs) const;

    protected:
      const unsigned int face_id;
      const SmartPointer<const Function<dim> >       lambda_function;
      const SmartPointer<const Function<dim> >       mu_function;
    };


    template <int dim>
    MeanStrainEnergyFace<dim>::
    MeanStrainEnergyFace (const unsigned int face_id,
                          const Function<dim>      &lambda_function,
                          const Function<dim>      &mu_function )
      :
      face_id (face_id),
      lambda_function (&lambda_function),
      mu_function (&mu_function)
    {}


    template <int dim>
    void
    MeanStrainEnergyFace<dim>::
    assemble_rhs_nonlinear (const DoFHandler<dim> &primal_dof_handler,
                            const Vector<double>  &primal_solution,
                            const DoFHandler<dim> &dof_handler,
                            Vector<double>        &rhs) const
    {
      // Assemble right hand side of the dual problem when the quantity of interest is
      // a nonlinear functinoal. In this case, the QoI should be linearized which depends
      // on the solution of the primal problem.
      // The extracter of the linearized QoI functional is the gradient of the the original
      // QoI functional with the primal solution values.

      AssertThrow (dim >= 2, ExcNotImplemented());

      rhs.reinit (dof_handler.n_dofs());

      const QGauss<dim-1> face_quadrature(dof_handler.get_fe().tensor_degree()+1);
      FEFaceValues<dim> primal_fe_face_values (primal_dof_handler.get_fe(), face_quadrature,
                                               update_quadrature_points |
                                               update_gradients | update_hessians |
                                               update_JxW_values);

      FEFaceValues<dim> fe_face_values (dof_handler.get_fe(), face_quadrature,
                                        update_values);

      const unsigned int  dofs_per_vertex = primal_dof_handler.get_fe().dofs_per_vertex;
      const unsigned int  n_face_q_points = face_quadrature.size();
      const unsigned int  dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

      AssertThrow(dofs_per_vertex == dim,
                  ExcDimensionMismatch (dofs_per_vertex, dim) );

      std::vector< std::vector< Tensor<1,dim> > > primal_solution_gradients;
      primal_solution_gradients.resize(n_face_q_points);

      std::vector<std::vector<Tensor<2,dim> > >   primal_solution_hessians;
      primal_solution_hessians.resize (n_face_q_points);

      for (unsigned int i=0; i!=n_face_q_points; ++i)
        {
          primal_solution_gradients[i].resize (dofs_per_vertex);
          primal_solution_hessians[i].resize  (dofs_per_vertex);
        }

      std::vector<double>   lambda_values (n_face_q_points);
      std::vector<double>   mu_values (n_face_q_points);

      Vector<double>      cell_rhs (dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      // bound_size : size of the boundary, in 2d is the length
      //              and in the 3d case, area
      double bound_size  = 0.;

      bool evaluation_face_found = false;

      typename DoFHandler<dim>::active_cell_iterator
      primal_cell = primal_dof_handler.begin_active(),
      primal_endc = primal_dof_handler.end();

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

      for (; cell!=endc; ++cell, ++primal_cell)
        {
          cell_rhs = 0;
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->face(face)->at_boundary()
                  &&
                  cell->face(face)->boundary_indicator() == face_id)
                {
                  if (!evaluation_face_found)
                    {
                      evaluation_face_found = true;
                    }
                  primal_fe_face_values.reinit (primal_cell, face);

                  primal_fe_face_values.get_function_grads (primal_solution,
                                                            primal_solution_gradients);

                  primal_fe_face_values.get_function_hessians (primal_solution,
                                                               primal_solution_hessians);

                  lambda_function->value_list (primal_fe_face_values.get_quadrature_points(), lambda_values);
                  mu_function->value_list     (primal_fe_face_values.get_quadrature_points(), mu_values);

                  fe_face_values.reinit (cell, face);

                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                      bound_size += primal_fe_face_values.JxW(q_point);

                      for (unsigned int m=0; m<dofs_per_cell; ++m)
                        {
                          const unsigned int
                          component_m = dof_handler.get_fe().system_to_component_index(m).first;

                          for (unsigned int i=0; i!=dofs_per_vertex; ++i)
                            {
                              for (unsigned int j=0; j!=dofs_per_vertex; ++j)
                                {
                                  cell_rhs(m) += fe_face_values.shape_value(m,q_point) *
                                                 (
                                                   lambda_values[q_point] *
                                                   (
                                                     primal_solution_hessians[q_point][i][i][component_m] * primal_solution_gradients[q_point][j][j]
                                                     +
                                                     primal_solution_gradients[q_point][i][i] * primal_solution_hessians[q_point][j][j][component_m]
                                                   )
                                                   +
                                                   mu_values[q_point] *
                                                   (
                                                     2*primal_solution_hessians[q_point][j][i][component_m] * primal_solution_gradients[q_point][j][i]
                                                     +
                                                     primal_solution_hessians[q_point][i][j][component_m] * primal_solution_gradients[q_point][j][i]
                                                     +
                                                     primal_solution_gradients[q_point][i][j] * primal_solution_hessians[q_point][j][i][component_m]
                                                   )
                                                 ) *
                                                 primal_fe_face_values.JxW(q_point);

                                }
                            }

                        } // end loop DoFs


                    }  // end loop Gauss points

                }  // end if face
            }  // end loop face

          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              rhs(local_dof_indices[i]) += cell_rhs(i);
            }

        }  // end loop cell

      AssertThrow(evaluation_face_found, ExcInternalError());

      rhs.scale (1./(2*bound_size));

    }


  }


  // DualSolver class
  template <int dim>
  class DualSolver
  {
  public:
    DualSolver (const Triangulation<dim>                        &triangulation,
                const FESystem<dim>                             &fe,
                const Vector<double>                            &solution,
                const ConstitutiveLaw<dim>                      &constitutive_law,
                const DualFunctional::DualFunctionalBase<dim>   &dual_functional,
                const unsigned int                              &timestep_no,
                const std::string                               &output_dir,
                const std::string                               &base_mesh,
                const double                                    &present_time,
                const double                                    &end_time);

    void compute_error_DWR (Vector<float> &estimated_error_per_cell);

    ~DualSolver ();

  private:
    void setup_system ();
    void compute_dirichlet_constraints ();
    void assemble_matrix ();
    void assemble_rhs ();
    void solve ();
    void output_results ();

    const FESystem<dim>     fe;
    DoFHandler<dim>         dof_handler;
    const Vector<double>    solution;

    const unsigned int      fe_degree;


    const unsigned int      fe_degree_dual;
    FESystem<dim>           fe_dual;
    DoFHandler<dim>         dof_handler_dual;

    const QGauss<dim>       quadrature_formula;
    const QGauss<dim - 1>   face_quadrature_formula;

    ConstraintMatrix        constraints_hanging_nodes_dual;
    ConstraintMatrix        constraints_dirichlet_and_hanging_nodes_dual;

    SparsityPattern         sparsity_pattern_dual;
    SparseMatrix<double>    system_matrix_dual;
    Vector<double>          system_rhs_dual;
    Vector<double>          solution_dual;

    const ConstitutiveLaw<dim> constitutive_law;

    const SmartPointer<const Triangulation<dim> > triangulation;
    const SmartPointer<const DualFunctional::DualFunctionalBase<dim> > dual_functional;

    unsigned int            timestep_no;
    std::string             output_dir;
    const std::string       base_mesh;
    double                  present_time;
    double                  end_time;
  };


  template<int dim>
  DualSolver<dim>::
  DualSolver (const Triangulation<dim>                        &triangulation,
              const FESystem<dim>                             &fe,
              const Vector<double>                            &solution,
              const ConstitutiveLaw<dim>                      &constitutive_law,
              const DualFunctional::DualFunctionalBase<dim>   &dual_functional,
              const unsigned int                              &timestep_no,
              const std::string                               &output_dir,
              const std::string                               &base_mesh,
              const double                                    &present_time,
              const double                                    &end_time)
    :
    fe (fe),
    dof_handler (triangulation),
    solution(solution),
    fe_degree(fe.tensor_degree()),
    fe_degree_dual(fe_degree + 1),
    fe_dual(FE_Q<dim>(fe_degree_dual), dim),
    dof_handler_dual (triangulation),
    quadrature_formula (fe_degree_dual + 1),
    face_quadrature_formula (fe_degree_dual + 1),
    constitutive_law (constitutive_law),
    triangulation (&triangulation),
    dual_functional (&dual_functional),
    timestep_no (timestep_no),
    output_dir (output_dir),
    base_mesh (base_mesh),
    present_time (present_time),
    end_time (end_time)
  {}


  template<int dim>
  DualSolver<dim>::~DualSolver()
  {
    dof_handler_dual.clear ();
  }


  template<int dim>
  void DualSolver<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    dof_handler_dual.distribute_dofs (fe_dual);
    std::cout << "    Number of degrees of freedom in dual problem:  "
              << dof_handler_dual.n_dofs()
              << std::endl;

    constraints_hanging_nodes_dual.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler_dual,
                                             constraints_hanging_nodes_dual);
    constraints_hanging_nodes_dual.close ();

    compute_dirichlet_constraints();

    sparsity_pattern_dual.reinit (dof_handler_dual.n_dofs(),
                                  dof_handler_dual.n_dofs(),
                                  dof_handler_dual.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (dof_handler_dual, sparsity_pattern_dual);

//    constraints_hanging_nodes_dual.condense (sparsity_pattern_dual);
    constraints_dirichlet_and_hanging_nodes_dual.condense (sparsity_pattern_dual);

    sparsity_pattern_dual.compress();

    system_matrix_dual.reinit (sparsity_pattern_dual);

    solution_dual.reinit (dof_handler_dual.n_dofs());
    system_rhs_dual.reinit (dof_handler_dual.n_dofs());

  }

  template<int dim>
  void DualSolver<dim>::compute_dirichlet_constraints()
  {
    constraints_dirichlet_and_hanging_nodes_dual.clear ();
    constraints_dirichlet_and_hanging_nodes_dual.merge(constraints_hanging_nodes_dual);

    std::vector<bool> component_mask(dim);

    if (base_mesh == "Timoshenko beam")
      {
        VectorTools::interpolate_boundary_values(dof_handler_dual,
                                                 0,
                                                 EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                 constraints_dirichlet_and_hanging_nodes_dual,
                                                 ComponentMask());
      }
    else if (base_mesh == "Thick_tube_internal_pressure")
      {
        // the boundary x = 0
        component_mask[0] = true;
        component_mask[1] = false;
        VectorTools::interpolate_boundary_values (dof_handler_dual,
                                                  2,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes_dual,
                                                  component_mask);
        // the boundary y = 0
        component_mask[0] = false;
        component_mask[1] = true;
        VectorTools::interpolate_boundary_values (dof_handler_dual,
                                                  3,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes_dual,
                                                  component_mask);
      }
    else if (base_mesh == "Perforated_strip_tension")
      {
        // the boundary x = 0
        component_mask[0] = true;
        component_mask[1] = false;
        component_mask[2] = false;
        VectorTools::interpolate_boundary_values (dof_handler_dual,
                                                  4,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes_dual,
                                                  component_mask);
        // the boundary y = 0
        component_mask[0] = false;
        component_mask[1] = true;
        component_mask[2] = false;
        VectorTools::interpolate_boundary_values (dof_handler_dual,
                                                  1,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes_dual,
                                                  component_mask);
        // the boundary y = imposed incremental displacement
        component_mask[0] = false;
        component_mask[1] = true;
        component_mask[2] = false;
        VectorTools::interpolate_boundary_values (dof_handler_dual,
                                                  3,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes_dual,
                                                  component_mask);
      }
    else if (base_mesh == "Cantiliver_beam_3d")
      {
        // the boundary x = y = z = 0
        component_mask[0] = true;
        component_mask[1] = true;
        component_mask[2] = true;
        VectorTools::interpolate_boundary_values (dof_handler_dual,
                                                  1,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes_dual,
                                                  component_mask);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }

    constraints_dirichlet_and_hanging_nodes_dual.close();
  }


  template<int dim>
  void DualSolver<dim>::assemble_matrix()
  {
    FEValues<dim> fe_values(fe, quadrature_formula, update_gradients);

    FEValues<dim> fe_values_dual(fe_dual, quadrature_formula,
                                 update_values | update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell_dual = fe_dual.dofs_per_cell;
    const unsigned int n_q_points         = quadrature_formula.size();

    FullMatrix<double> cell_matrix (dofs_per_cell_dual, dofs_per_cell_dual);

    std::vector<types::global_dof_index>   local_dof_indices(dofs_per_cell_dual);

    typename DoFHandler<dim>::active_cell_iterator
    cell_dual = dof_handler_dual.begin_active(),
    endc_dual = dof_handler_dual.end(),
    cell = dof_handler.begin_active();

    const FEValuesExtractors::Vector displacement(0);

    for (; cell_dual != endc_dual; ++cell_dual, ++cell)
      if (cell_dual->is_locally_owned())
        {
          fe_values.reinit(cell);

          fe_values_dual.reinit(cell_dual);
          cell_matrix = 0;

          std::vector<SymmetricTensor<2, dim> > strain_tensor(n_q_points);
          fe_values[displacement].get_function_symmetric_gradients(solution,
                                                                   strain_tensor);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              SymmetricTensor<4, dim> stress_strain_tensor_linearized;
              SymmetricTensor<4, dim> stress_strain_tensor;
              constitutive_law.get_linearized_stress_strain_tensors(strain_tensor[q_point],
                                                                    stress_strain_tensor_linearized,
                                                                    stress_strain_tensor);

              for (unsigned int i = 0; i < dofs_per_cell_dual; ++i)
                {
                  const SymmetricTensor<2, dim>
                  stress_phi_i = stress_strain_tensor_linearized
                                 * fe_values_dual[displacement].symmetric_gradient(i, q_point);

                  for (unsigned int j = 0; j < dofs_per_cell_dual; ++j)
                    cell_matrix(i, j) += (stress_phi_i
                                          * fe_values_dual[displacement].symmetric_gradient(j, q_point)
                                          * fe_values_dual.JxW(q_point));

                }

            }

          cell_dual->get_dof_indices(local_dof_indices);
          constraints_dirichlet_and_hanging_nodes_dual.distribute_local_to_global(cell_matrix,
              local_dof_indices,
              system_matrix_dual);

        }

  }


  template<int dim>
  void DualSolver<dim>::assemble_rhs()
  {
    dual_functional->assemble_rhs (dof_handler, solution, constitutive_law,
                                   dof_handler_dual, system_rhs_dual);
    constraints_dirichlet_and_hanging_nodes_dual.condense (system_rhs_dual);
  }


  template<int dim>
  void DualSolver<dim>::solve()
  {
    // +++  direct solver +++++++++
    SparseDirectUMFPACK   A_direct;
    A_direct.initialize(system_matrix_dual);

    // After the decomposition, we can use A_direct like a matrix representing
    // the inverse of our system matrix, so to compute the solution we just
    // have to multiply with the right hand side vector:
    A_direct.vmult(solution_dual, system_rhs_dual);

    // ++++  iterative solver ++ CG ++++ doesn't work
//    SolverControl solver_control (5000, 1e-12);
//    SolverCG<> cg (solver_control);
//
//    PreconditionSSOR<> preconditioner;
//    preconditioner.initialize(system_matrix_dual, 1.2);
//
//    cg.solve (system_matrix_dual, solution_dual, system_rhs_dual,
//              preconditioner);

    // ++++  iterative solver ++ BiCGStab ++++++ doesn't work
//    SolverControl solver_control (5000, 1e-12);
//    SolverBicgstab<> bicgstab (solver_control);
//
//    PreconditionJacobi<> preconditioner;
//    preconditioner.initialize(system_matrix_dual, 1.0);
//
//    bicgstab.solve (system_matrix_dual, solution_dual, system_rhs_dual,
//                    preconditioner);

    // +++++++++++++++++++++++++++++++++++++++++++++++++

    constraints_dirichlet_and_hanging_nodes_dual.distribute (solution_dual);
  }

  template<int dim>
  void DualSolver<dim>::output_results()
  {
    std::string filename = (output_dir + "dual-solution-" +
                            Utilities::int_to_string(timestep_no, 4) + ".vtk");
    std::ofstream output (filename.c_str());
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler_dual);
    std::vector<std::string> solution_names;
    switch (dim)
      {
      case 1:
        solution_names.push_back ("displacement");
        break;
      case 2:
        solution_names.push_back ("x_displacement");
        solution_names.push_back ("y_displacement");
        break;
      case 3:
        solution_names.push_back ("x_displacement");
        solution_names.push_back ("y_displacement");
        solution_names.push_back ("z_displacement");
        break;
      default:
        Assert (false, ExcNotImplemented());
      }
    data_out.add_data_vector (solution_dual, solution_names);
    data_out.build_patches ();
    data_out.write_vtk (output);
  }

  template<int dim>
  void DualSolver<dim>::compute_error_DWR (Vector<float> &estimated_error_per_cell)
  {
    Assert (estimated_error_per_cell.size() == triangulation->n_global_active_cells(),
            ExcDimensionMismatch (estimated_error_per_cell.size(), triangulation->n_global_active_cells()));

    // solve the dual problem
    setup_system ();
    assemble_matrix ();
    assemble_rhs ();
    solve ();
    output_results ();

    // compuate the dual weights
    Vector<double> primal_solution (dof_handler_dual.n_dofs());
    FETools::interpolate (dof_handler,
                          solution,
                          dof_handler_dual,
                          constraints_dirichlet_and_hanging_nodes_dual,
                          primal_solution);

    ConstraintMatrix constraints_hanging_nodes;
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints_hanging_nodes);
    constraints_hanging_nodes.close();
    Vector<double> dual_weights (dof_handler_dual.n_dofs());
    FETools::interpolation_difference (dof_handler_dual,
                                       constraints_dirichlet_and_hanging_nodes_dual,
                                       solution_dual,
                                       dof_handler,
                                       constraints_hanging_nodes,
                                       dual_weights);

    // estimate the error
    FEValues<dim> fe_values(fe_dual, quadrature_formula,
                            update_values    |
                            update_gradients |
                            update_hessians  |
                            update_quadrature_points |
                            update_JxW_values);

    const unsigned int n_q_points      = quadrature_formula.size();
    std::vector<SymmetricTensor<2, dim> > strain_tensor(n_q_points);
    SymmetricTensor<4, dim> stress_strain_tensor_linearized;
    SymmetricTensor<4, dim> stress_strain_tensor;
    Tensor<5, dim>          stress_strain_tensor_grad;
    std::vector<std::vector<Tensor<2,dim> > > cell_hessians (n_q_points);
    for (unsigned int i=0; i!=n_q_points; ++i)
      {
        cell_hessians[i].resize (dim);
      }
    std::vector<Vector<double> > dual_weights_cell_values (n_q_points, Vector<double>(dim));

    const EquationData::BodyForce<dim> body_force;
    std::vector<Vector<double> > body_force_values (n_q_points, Vector<double>(dim));
    const FEValuesExtractors::Vector displacement(0);


    FEFaceValues<dim> fe_face_values_cell(fe_dual, face_quadrature_formula,
                                          update_values           |
                                          update_quadrature_points|
                                          update_gradients        |
                                          update_JxW_values       |
                                          update_normal_vectors),
                                          fe_face_values_neighbor (fe_dual, face_quadrature_formula,
                                              update_values     |
                                              update_gradients  |
                                              update_JxW_values |
                                              update_normal_vectors);
    FESubfaceValues<dim> fe_subface_values_cell (fe_dual, face_quadrature_formula,
                                                 update_gradients);

    const unsigned int n_face_q_points = face_quadrature_formula.size();
    std::vector<Vector<double> > jump_residual (n_face_q_points, Vector<double>(dim));
    std::vector<Vector<double> > dual_weights_face_values (n_face_q_points, Vector<double>(dim));

    std::vector<std::vector<Tensor<1,dim> > > cell_grads(n_face_q_points);
    for (unsigned int i=0; i!=n_face_q_points; ++i)
      {
        cell_grads[i].resize (dim);
      }
    std::vector<std::vector<Tensor<1,dim> > > neighbor_grads(n_face_q_points);
    for (unsigned int i=0; i!=n_face_q_points; ++i)
      {
        neighbor_grads[i].resize (dim);
      }
    SymmetricTensor<2, dim> q_cell_strain_tensor;
    SymmetricTensor<2, dim> q_neighbor_strain_tensor;
    SymmetricTensor<4, dim> cell_stress_strain_tensor;
    SymmetricTensor<4, dim> neighbor_stress_strain_tensor;


    typename std::map<typename DoFHandler<dim>::face_iterator, Vector<double> >
    face_integrals;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_dual.begin_active(),
    endc = dof_handler_dual.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          for (unsigned int face_no=0;
               face_no<GeometryInfo<dim>::faces_per_cell;
               ++face_no)
            {
              face_integrals[cell->face(face_no)].reinit (dim);
              face_integrals[cell->face(face_no)] = -1e20;
            }
        }

    std::vector<Vector<float> > error_indicators_vector;
    error_indicators_vector.resize( triangulation->n_active_cells(),
                                    Vector<float>(dim) );

    // ----------------- estimate_some -------------------------
    cell = dof_handler_dual.begin_active();
    unsigned int present_cell = 0;
    for (; cell!=endc; ++cell, ++present_cell)
      if (cell->is_locally_owned())
        {
          // --------------- integrate_over_cell -------------------
          fe_values.reinit(cell);
          body_force.vector_value_list(fe_values.get_quadrature_points(),
                                       body_force_values);
          fe_values[displacement].get_function_symmetric_gradients(primal_solution,
                                                                   strain_tensor);
          fe_values.get_function_hessians(primal_solution, cell_hessians);

          fe_values.get_function_values(dual_weights,
                                        dual_weights_cell_values);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              constitutive_law.get_linearized_stress_strain_tensors(strain_tensor[q_point],
                                                                    stress_strain_tensor_linearized,
                                                                    stress_strain_tensor);
              constitutive_law.get_grad_stress_strain_tensor(strain_tensor[q_point],
                                                             cell_hessians[q_point],
                                                             stress_strain_tensor_grad);

              for (unsigned int i=0; i!=dim; ++i)
                {
                  error_indicators_vector[present_cell](i) +=
                    body_force_values[q_point](i)*
                    dual_weights_cell_values[q_point](i)*
                    fe_values.JxW(q_point);
                  for (unsigned int j=0; j!=dim; ++j)
                    {
                      for (unsigned int k=0; k!=dim; ++k)
                        {
                          for (unsigned int l=0; l!=dim; ++l)
                            {
                              error_indicators_vector[present_cell](i) +=
                                ( stress_strain_tensor[i][j][k][l]*
                                  0.5*(cell_hessians[q_point][k][l][j]
                                       +
                                       cell_hessians[q_point][l][k][j])
                                  + stress_strain_tensor_grad[i][j][k][l][j] * strain_tensor[q_point][k][l]
                                ) *
                                dual_weights_cell_values[q_point](i) *
                                fe_values.JxW(q_point);
                            }
                        }
                    }

                }

            }
          // -------------------------------------------------------
          // compute face_integrals
          for (unsigned int face_no=0;
               face_no<GeometryInfo<dim>::faces_per_cell;
               ++face_no)
            {
              if (cell->face(face_no)->at_boundary())
                {
                  for (unsigned int id=0; id!=dim; ++id)
                    {
                      face_integrals[cell->face(face_no)](id) = 0;
                    }
                  continue;
                }

              if ((cell->neighbor(face_no)->has_children() == false) &&
                  (cell->neighbor(face_no)->level() == cell->level()) &&
                  (cell->neighbor(face_no)->index() < cell->index()))
                continue;

              if (cell->at_boundary(face_no) == false)
                if (cell->neighbor(face_no)->level() < cell->level())
                  continue;


              if (cell->face(face_no)->has_children() == false)
                {
                  // ------------- integrate_over_regular_face -----------
                  fe_face_values_cell.reinit(cell, face_no);
                  fe_face_values_cell.get_function_grads (primal_solution,
                                                          cell_grads);

                  Assert (cell->neighbor(face_no).state() == IteratorState::valid,
                          ExcInternalError());
                  const unsigned int
                  neighbor_neighbor = cell->neighbor_of_neighbor (face_no);
                  const typename DoFHandler<dim>::active_cell_iterator
                  neighbor = cell->neighbor(face_no);

                  fe_face_values_neighbor.reinit(neighbor, neighbor_neighbor);
                  fe_face_values_neighbor.get_function_grads (primal_solution,
                                                              neighbor_grads);

                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                      q_cell_strain_tensor = 0.;
                      q_neighbor_strain_tensor = 0.;
                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          for (unsigned int j=0; j!=dim; ++j)
                            {
                              q_cell_strain_tensor[i][j] = 0.5*(cell_grads[q_point][i][j] +
                                                                cell_grads[q_point][j][i] );
                              q_neighbor_strain_tensor[i][j] = 0.5*(neighbor_grads[q_point][i][j] +
                                                                    neighbor_grads[q_point][j][i] );
                            }
                        }

                      constitutive_law.get_stress_strain_tensor (q_cell_strain_tensor,
                                                                 cell_stress_strain_tensor);
                      constitutive_law.get_stress_strain_tensor (q_neighbor_strain_tensor,
                                                                 neighbor_stress_strain_tensor);

                      jump_residual[q_point] = 0.;
                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          for (unsigned int j=0; j!=dim; ++j)
                            {
                              for (unsigned int k=0; k!=dim; ++k)
                                {
                                  for (unsigned int l=0; l!=dim; ++l)
                                    {
                                      jump_residual[q_point](i) += (cell_stress_strain_tensor[i][j][k][l]*
                                                                    q_cell_strain_tensor[k][l]
                                                                    -
                                                                    neighbor_stress_strain_tensor[i][j][k][l]*
                                                                    q_neighbor_strain_tensor[k][l] )*
                                                                   fe_face_values_cell.normal_vector(q_point)[j];
                                    }
                                }
                            }
                        }

                    }

                  fe_face_values_cell.get_function_values (dual_weights,
                                                           dual_weights_face_values);

                  Vector<double> face_integral_vector(dim);
                  face_integral_vector = 0;
                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          face_integral_vector(i) += jump_residual[q_point](i) *
                                                     dual_weights_face_values[q_point](i) *
                                                     fe_face_values_cell.JxW(q_point);
                        }
                    }

                  Assert (face_integrals.find (cell->face(face_no)) != face_integrals.end(),
                          ExcInternalError());

                  for (unsigned int i=0; i!=dim; ++i)
                    {
                      Assert (face_integrals[cell->face(face_no)](i) == -1e20,
                              ExcInternalError());
                      face_integrals[cell->face(face_no)](i) = face_integral_vector(i);

                    }

                  // -----------------------------------------------------
                }
              else
                {
                  // ------------- integrate_over_irregular_face ---------
                  const typename DoFHandler<dim>::face_iterator
                  face = cell->face(face_no);
                  const typename DoFHandler<dim>::cell_iterator
                  neighbor = cell->neighbor(face_no);
                  Assert (neighbor.state() == IteratorState::valid,
                          ExcInternalError());
                  Assert (neighbor->has_children(),
                          ExcInternalError());

                  const unsigned int
                  neighbor_neighbor = cell->neighbor_of_neighbor (face_no);

                  for (unsigned int subface_no=0;
                       subface_no<face->n_children(); ++subface_no)
                    {
                      const typename DoFHandler<dim>::active_cell_iterator
                      neighbor_child = cell->neighbor_child_on_subface (face_no, subface_no);
                      Assert (neighbor_child->face(neighbor_neighbor) ==
                              cell->face(face_no)->child(subface_no),
                              ExcInternalError());

                      fe_subface_values_cell.reinit (cell, face_no, subface_no);
                      fe_subface_values_cell.get_function_grads (primal_solution,
                                                                 cell_grads);
                      fe_face_values_neighbor.reinit (neighbor_child,
                                                      neighbor_neighbor);
                      fe_face_values_neighbor.get_function_grads (primal_solution,
                                                                  neighbor_grads);

                      for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        {
                          q_cell_strain_tensor = 0.;
                          q_neighbor_strain_tensor = 0.;
                          for (unsigned int i=0; i!=dim; ++i)
                            {
                              for (unsigned int j=0; j!=dim; ++j)
                                {
                                  q_cell_strain_tensor[i][j] = 0.5*(cell_grads[q_point][i][j] +
                                                                    cell_grads[q_point][j][i] );
                                  q_neighbor_strain_tensor[i][j] = 0.5*(neighbor_grads[q_point][i][j] +
                                                                        neighbor_grads[q_point][j][i] );
                                }
                            }

                          constitutive_law.get_stress_strain_tensor (q_cell_strain_tensor,
                                                                     cell_stress_strain_tensor);
                          constitutive_law.get_stress_strain_tensor (q_neighbor_strain_tensor,
                                                                     neighbor_stress_strain_tensor);

                          jump_residual[q_point] = 0.;
                          for (unsigned int i=0; i!=dim; ++i)
                            {
                              for (unsigned int j=0; j!=dim; ++j)
                                {
                                  for (unsigned int k=0; k!=dim; ++k)
                                    {
                                      for (unsigned int l=0; l!=dim; ++l)
                                        {
                                          jump_residual[q_point](i) += (-cell_stress_strain_tensor[i][j][k][l]*
                                                                        q_cell_strain_tensor[k][l]
                                                                        +
                                                                        neighbor_stress_strain_tensor[i][j][k][l]*
                                                                        q_neighbor_strain_tensor[k][l] )*
                                                                       fe_face_values_neighbor.normal_vector(q_point)[j];
                                        }
                                    }
                                }
                            }

                        }

                      fe_face_values_neighbor.get_function_values (dual_weights,
                                                                   dual_weights_face_values);

                      Vector<double> face_integral_vector(dim);
                      face_integral_vector = 0;
                      for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        {
                          for (unsigned int i=0; i!=dim; ++i)
                            {
                              face_integral_vector(i) += jump_residual[q_point](i) *
                                                         dual_weights_face_values[q_point](i) *
                                                         fe_face_values_neighbor.JxW(q_point);
                            }
                        }

                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          face_integrals[neighbor_child->face(neighbor_neighbor)](i) = face_integral_vector(i);
                        }

                    }

                  Vector<double> sum (dim);
                  sum = 0;
                  for (unsigned int subface_no=0;
                       subface_no<face->n_children(); ++subface_no)
                    {
                      Assert (face_integrals.find(face->child(subface_no)) !=
                              face_integrals.end(),
                              ExcInternalError());
                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          Assert (face_integrals[face->child(subface_no)](i) != -1e20,
                                  ExcInternalError());
                          sum(i) += face_integrals[face->child(subface_no)](i);
                        }
                    }
                  for (unsigned int i=0; i!=dim; ++i)
                    {
                      face_integrals[face](i) = sum(i);
                    }


                  // -----------------------------------------------------
                }


            }
        }
    // ----------------------------------------------------------

    present_cell=0;
    cell = dof_handler_dual.begin_active();
    for (; cell!=endc; ++cell, ++present_cell)
      if (cell->is_locally_owned())
        {
          for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell;
               ++face_no)
            {
              Assert(face_integrals.find(cell->face(face_no)) !=
                     face_integrals.end(),
                     ExcInternalError());

              for (unsigned int id=0; id!=dim; ++id)
                {
                  error_indicators_vector[present_cell](id)
                  -= 0.5*face_integrals[cell->face(face_no)](id);
                }

            }

          estimated_error_per_cell(present_cell) = error_indicators_vector[present_cell].l2_norm();

        }
  }



  // @sect3{The <code>PlasticityContactProblem</code> class template}

  // This is the main class of this program and supplies all functions
  // and variables needed to describe
  // the nonlinear contact problem. It is
  // close to step-41 but with some additional
  // features like handling hanging nodes,
  // a Newton method, using Trilinos and p4est
  // for parallel distributed computing.
  // To deal with hanging nodes makes
  // life a bit more complicated since
  // we need another ConstraintMatrix now.
  // We create a Newton method for the
  // active set method for the contact
  // situation and to handle the nonlinear
  // operator for the constitutive law.
  //
  // The general layout of this class is very much like for most other tutorial programs.
  // To make our life a bit easier, this class reads a set of input parameters from an input file. These
  // parameters, using the ParameterHandler class, are declared in the <code>declare_parameters</code>
  // function (which is static so that it can be called before we even create an object of the current
  // type), and a ParameterHandler object that has been used to read an input file will then be passed
  // to the constructor of this class.
  //
  // The remaining member functions are by and large as we have seen in several of the other tutorial
  // programs, though with additions for the current nonlinear system. We will comment on their purpose
  // as we get to them further below.
  template <int dim>
  class ElastoPlasticProblem
  {
  public:
    ElastoPlasticProblem (const ParameterHandler &prm);

    void run ();

    static void declare_parameters (ParameterHandler &prm);

  private:
    void make_grid ();
    void setup_system ();
    void compute_dirichlet_constraints ();
    void assemble_newton_system (const TrilinosWrappers::MPI::Vector &linearization_point,
                                 const TrilinosWrappers::MPI::Vector &delta_linearization_point);
    void compute_nonlinear_residual (const TrilinosWrappers::MPI::Vector &linearization_point);
    void solve_newton_system ();
    void solve_newton ();
    void compute_error ();
    void compute_error_residual (const TrilinosWrappers::MPI::Vector &tmp_solution);
    void refine_grid ();
    void move_mesh (const TrilinosWrappers::MPI::Vector &displacement) const;
    void output_results (const std::string &filename_base);

    // Next are three functions that handle the history variables stored in each
    // quadrature point. The first one is called before the first timestep to
    // set up a pristine state for the history variables. It only works on
    // those quadrature points on cells that belong to the present processor:
    void setup_quadrature_point_history ();

    // The second one updates the history variables at the end of each
    // timestep:
    void update_quadrature_point_history ();

    // As far as member variables are concerned, we start with ones that we use to
    // indicate the MPI universe this program runs on, and then two numbers
    // telling us how many participating processors there are, and where in
    // this world we are., a stream we use to let
    // exactly one processor produce output to the console (see step-17) and
    // a variable that is used to time the various sections of the program:
    MPI_Comm           mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    // The next group describes the mesh and the finite element space.
    // In particular, for this parallel program, the finite element
    // space has associated with it variables that indicate which degrees
    // of freedom live on the current processor (the index sets, see
    // also step-40 and the @ref distributed documentation module) as
    // well as a variety of constraints: those imposed by hanging nodes,
    // by Dirichlet boundary conditions, and by the active set of
    // contact nodes. Of the three ConstraintMatrix variables defined
    // here, the first only contains hanging node constraints, the
    // second also those associated with Dirichlet boundary conditions,
    // and the third these plus the contact constraints.
    //
    // The variable <code>active_set</code> consists of those degrees
    // of freedom constrained by the contact, and we use
    // <code>fraction_of_plastic_q_points_per_cell</code> to keep
    // track of the fraction of quadrature points on each cell where
    // the stress equals the yield stress. The latter is only used to
    // create graphical output showing the plastic zone, but not for
    // any further computation; the variable is a member variable of
    // this class since the information is computed as a by-product
    // of computing the residual, but is used only much later. (Note
    // that the vector is a vector of length equal to the number of
    // active cells on the <i>local mesh</i>; it is never used to
    // exchange information between processors and can therefore be
    // a regular deal.II vector.)
    const unsigned int                        n_initial_global_refinements;
    parallel::distributed::Triangulation<dim> triangulation;

    const unsigned int fe_degree;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    IndexSet           locally_owned_dofs;
    IndexSet           locally_relevant_dofs;

    ConstraintMatrix   constraints_hanging_nodes;
    ConstraintMatrix   constraints_dirichlet_and_hanging_nodes;

    Vector<float>      fraction_of_plastic_q_points_per_cell;

    // One difference of this program is that we declare the quadrature
    // formula in the class declaration. The reason is that in all the other
    // programs, it didn't do much harm if we had used different quadrature
    // formulas when computing the matrix and the right hand side, for
    // example. However, in the present case it does: we store information in
    // the quadrature points, so we have to make sure all parts of the program
    // agree on where they are and how many there are on each cell. Thus, let
    // us first declare the quadrature formula that will be used throughout...
    const QGauss<dim>          quadrature_formula;
    const QGauss<dim - 1>      face_quadrature_formula;

    // ... and then also have a vector of history objects, one per quadrature
    // point on those cells for which we are responsible (i.e. we don't store
    // history data for quadrature points on cells that are owned by other
    // processors).
    std::vector<PointHistory<dim> > quadrature_point_history;

    // The way this object is accessed is through a <code>user pointer</code>
    // that each cell, face, or edge holds: it is a <code>void*</code> pointer
    // that can be used by application programs to associate arbitrary data to
    // cells, faces, or edges. What the program actually does with this data
    // is within its own responsibility, the library just allocates some space
    // for these pointers, and application programs can set and read the
    // pointers for each of these objects.


    // The next block of variables corresponds to the solution
    // and the linear systems we need to form. In particular, this
    // includes the Newton matrix and right hand side; the vector
    // that corresponds to the residual (i.e., the Newton right hand
    // side) but from which we have not eliminated the various
    // constraints and that is used to determine which degrees of
    // freedom need to be constrained in the next iteration; and
    // a vector that corresponds to the diagonal of the $B$ matrix
    // briefly mentioned in the introduction and discussed in the
    // accompanying paper.
    TrilinosWrappers::SparseMatrix    newton_matrix;

    TrilinosWrappers::MPI::Vector     solution;
    TrilinosWrappers::MPI::Vector     incremental_displacement;
    TrilinosWrappers::MPI::Vector     newton_rhs;
    TrilinosWrappers::MPI::Vector     newton_rhs_residual;

    // The next block of variables is then related to the time dependent
    // nature of the problem: they denote the length of the time interval
    // which we want to simulate, the present time and number of time step,
    // and length of present timestep:
    double       present_time;
    double       present_timestep;
    double       end_time;
    unsigned int timestep_no;

    // The next block contains the variables that describe the material
    // response:
    const double         e_modulus, nu, sigma_0, gamma;
    ConstitutiveLaw<dim> constitutive_law;

    // And then there is an assortment of other variables that are used
    // to identify the mesh we are asked to build as selected by the
    // parameter file, the obstacle that is being pushed into the
    // deformable body, the mesh refinement strategy, whether to transfer
    // the solution from one mesh to the next, and how many mesh
    // refinement cycles to perform. As possible, we mark these kinds
    // of variables as <code>const</code> to help the reader identify
    // which ones may or may not be modified later on (the output directory
    // being an exception -- it is never modified outside the constructor
    // but it is awkward to initialize in the member-initializer-list
    // following the colon in the constructor since there we have only
    // one shot at setting it; the same is true for the mesh refinement
    // criterion):
    const std::string                                  base_mesh;

    struct RefinementStrategy
    {
      enum value
      {
        refine_global,
        refine_percentage,
        refine_fix_dofs
      };
    };
    typename RefinementStrategy::value                 refinement_strategy;

    struct ErrorEstimationStrategy
    {
      enum value
      {
        kelly_error,
        residual_error,
        weighted_residual_error,
        weighted_kelly_error
      };
    };
    typename ErrorEstimationStrategy::value            error_estimation_strategy;

    Vector<float>                                      estimated_error_per_cell;

    const bool                                         transfer_solution;
    std::string                                        output_dir;
    TableHandler                                       table_results,
                                                       table_results_2,
                                                       table_results_3;

    unsigned int                                       current_refinement_cycle;

    const double                                       max_relative_error;
    float                                              relative_error;

    const bool                                         show_stresses;
  };


  // @sect3{Implementation of the <code>PlasticityContactProblem</code> class}

  // @sect4{PlasticityContactProblem::declare_parameters}

  // Let us start with the declaration of run-time parameters that can be
  // selected in the input file. These values will be read back in the
  // constructor of this class to initialize the member variables of this
  // class:
  template <int dim>
  void
  ElastoPlasticProblem<dim>::declare_parameters (ParameterHandler &prm)
  {
    prm.declare_entry("polynomial degree", "1",
                      Patterns::Integer(),
                      "Polynomial degree of the FE_Q finite element space, typically 1 or 2.");
    prm.declare_entry("number of initial refinements", "2",
                      Patterns::Integer(),
                      "Number of initial global mesh refinement steps before "
                      "the first computation.");
    prm.declare_entry("refinement strategy", "percentage",
                      Patterns::Selection("global|percentage"),
                      "Mesh refinement strategy:\n"
                      " global: one global refinement\n"
                      " percentage: a fixed percentage of cells gets refined using the selected error estimator.");
    prm.declare_entry("error estimation strategy", "kelly_error",
                      Patterns::Selection("kelly_error|residual_error|weighted_residual_error"),
                      "Error estimation strategy:\n"
                      " kelly_error: Kelly error estimator\n"
                      " residual_error: residual-based error estimator\n"
                      " weighted_residual_error: dual weighted residual (Goal-oriented) error estimator.\n");
    prm.declare_entry("maximum relative error","0.05",
                      Patterns::Double(),
                      "maximum relative error which plays the role of a criteria for refinement.");
    prm.declare_entry("number of cycles", "5",
                      Patterns::Integer(),
                      "Number of adaptive mesh refinement cycles to run.");
    prm.declare_entry("output directory", "",
                      Patterns::Anything(),
                      "Directory for output files (graphical output and benchmark "
                      "statistics). If empty, use the current directory.");
    prm.declare_entry("transfer solution", "true",
                      Patterns::Bool(),
                      "Whether the solution should be used as a starting guess "
                      "for the next finer mesh. If false, then the iteration starts at "
                      "zero on every mesh.");
    prm.declare_entry("base mesh", "Thick_tube_internal_pressure",
                      Patterns::Selection("Timoshenko beam|Thick_tube_internal_pressure|"
                                          "Perforated_strip_tension|Cantiliver_beam_3d"),
                      "Select the shape of the domain: 'box' or 'half sphere'");
    prm.declare_entry("elasticity modulus","2.e11",
                      Patterns::Double(),
                      "Elasticity modulus of the material in MPa (N/mm2)");
    prm.declare_entry("Poissons ratio","0.3",
                      Patterns::Double(),
                      "Poisson's ratio of the material");
    prm.declare_entry("yield stress","2.e11",
                      Patterns::Double(),
                      "Yield stress of the material in MPa (N/mm2)");
    prm.declare_entry("isotropic hardening parameter","0.",
                      Patterns::Double(),
                      "Isotropic hardening parameter of the material");
    prm.declare_entry("show stresses", "false",
                      Patterns::Bool(),
                      "Whether illustrates the stresses and von Mises stresses or not.");


  }


  // @sect4{The <code>PlasticityContactProblem</code> constructor}

  // Given the declarations of member variables as well as the
  // declarations of run-time parameters that are read from the input
  // file, there is nothing surprising in this constructor. In the body
  // we initialize the mesh refinement strategy and the output directory,
  // creating such a directory if necessary.
  template <int dim>
  ElastoPlasticProblem<dim>::
  ElastoPlasticProblem (const ParameterHandler &prm)
    :
    mpi_communicator(MPI_COMM_WORLD),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout(std::cout, this_mpi_process == 0),
    computing_timer(MPI_COMM_WORLD, pcout, TimerOutput::never,
                    TimerOutput::wall_times),

    n_initial_global_refinements (prm.get_integer("number of initial refinements")),
    triangulation(mpi_communicator),
    fe_degree (prm.get_integer("polynomial degree")),
    fe(FE_Q<dim>(QGaussLobatto<1>(fe_degree+1)), dim),
    dof_handler(triangulation),
    quadrature_formula (fe_degree + 1),
    face_quadrature_formula (fe_degree + 1),

    e_modulus (prm.get_double("elasticity modulus")),
    nu (prm.get_double("Poissons ratio")),
    sigma_0(prm.get_double("yield stress")),
    gamma (prm.get_double("isotropic hardening parameter")),
    constitutive_law (e_modulus,
                      nu,
                      sigma_0,
                      gamma),

    base_mesh (prm.get("base mesh")),

    transfer_solution (prm.get_bool("transfer solution")),
    table_results(),
    table_results_2(),
    table_results_3(),
    max_relative_error (prm.get_double("maximum relative error")),
    show_stresses (prm.get_bool("show stresses"))
  {
    std::string strat = prm.get("refinement strategy");
    if (strat == "global")
      refinement_strategy = RefinementStrategy::refine_global;
    else if (strat == "percentage")
      refinement_strategy = RefinementStrategy::refine_percentage;
    else
      AssertThrow (false, ExcNotImplemented());

    strat = prm.get("error estimation strategy");
    if (strat == "kelly_error")
      error_estimation_strategy = ErrorEstimationStrategy::kelly_error;
    else if (strat == "residual_error")
      error_estimation_strategy = ErrorEstimationStrategy::residual_error;
    else if (strat == "weighted_residual_error")
      error_estimation_strategy = ErrorEstimationStrategy::weighted_residual_error;
    else
      AssertThrow(false, ExcNotImplemented());

    output_dir = prm.get("output directory");
    if (output_dir != "" && *(output_dir.rbegin()) != '/')
      output_dir += "/";
    mkdir(output_dir.c_str(), 0777);

    pcout << "    Using output directory '" << output_dir << "'" << std::endl;
    pcout << "    FE degree " << fe_degree << std::endl;
    pcout << "    transfer solution "
          << (transfer_solution ? "true" : "false") << std::endl;
  }



  // @sect4{PlasticityContactProblem::make_grid}

  // The next block deals with constructing the starting mesh.
  // We will use the following helper function and the first
  // block of the <code>make_grid()</code> to construct a
  // mesh that corresponds to a half sphere. deal.II has a function
  // that creates such a mesh, but it is in the wrong location
  // and facing the wrong direction, so we need to shift and rotate
  // it a bit before using it.
  //
  // For later reference, as described in the documentation of
  // GridGenerator::half_hyper_ball(), the flat surface of the halfsphere
  // has boundary indicator zero, while the remainder has boundary
  // indicator one.
  Point<3>
  rotate_half_sphere (const Point<3> &in)
  {
    return Point<3>(in(2), in(1), -in(0));
  }

  template <int dim>
  void
  ElastoPlasticProblem<dim>::make_grid ()
  {
    if (base_mesh == "Timoshenko beam")
      {
        AssertThrow (dim == 2, ExcNotImplemented());

        const double length = .48,
                     depth  = .12;

        const Point<dim> point_1(0, -depth/2),
              point_2(length, depth/2);

        std::vector<unsigned int> repetitions(2);
        repetitions[0] = 4;
        repetitions[1] = 1;
        GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, point_1, point_2);


        // give the indicators to boundaries for specification,
        //
        //     ________100______
        //     |                |
        //   0 |                | 5
        //     |________________|
        //             100
        // 0 to essential boundary conditions (left edge) which are as default
        // 100 to the null boundaries (upper and lower edges) where we do not need to take care of them
        // 5 to the natural boundaries (right edge) for imposing the traction force
        typename Triangulation<dim>::cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (; cell!=endc; ++cell)
          {
            for (unsigned int face=0; face!=GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if ( std::fabs(cell->face(face)->center()(0)-length) < 1e-12 )
                  {
                    cell->face(face)->set_boundary_indicator(5);
                  }
                else if ( ( std::fabs(cell->face(face)->center()(1)-(depth/2)) < 1e-12 )
                          ||
                          ( std::fabs(cell->face(face)->center()(1)-(-depth/2)) < 1e-12 ) )
                  {
                    cell->face(face)->set_boundary_indicator(100);
                  }

              }
          }

        triangulation.refine_global(n_initial_global_refinements);

      }
    else if (base_mesh == "Thick_tube_internal_pressure")
      {
        // Example 1 from the paper: Zhong Z., .... A new numerical method for determining
        // collapse load-carrying capacity of structure made of elasto-plastic material,
        // J. Cent. South Univ. (2014) 21: 398-404
        AssertThrow (dim == 2, ExcNotImplemented());

        const Point<dim> center(0, 0);
        const double inner_radius = .1,
                     outer_radius = .2;
        GridGenerator::quarter_hyper_shell(triangulation,
                                           center, inner_radius, outer_radius,
                                           0, true);

        // give the indicators to boundaries for specification,

        /*    _____
             |     \
             |       \
           2 |         \ 1
             |_          \
               \          \
              0 \         |
                 |________|
                     3
        */
        // 0 - inner boundary  - natural boundary condition - impose the traction force
        // 1 - outer boundary  - free boundary - we do not need to take care of them
        // 2 - left boundary   - essential boundary condition - constrained to move along the x direction
        // 3 - bottom boundary - essential boundary condition - constrained to move along the y direction

        const HyperBallBoundary<dim> inner_boundary_description(center, inner_radius);
        triangulation.set_boundary (0, inner_boundary_description);

        const HyperBallBoundary<dim> outer_boundary_description(center, outer_radius);
        triangulation.set_boundary (1, outer_boundary_description);

        triangulation.refine_global(n_initial_global_refinements);

        triangulation.set_boundary (0);
        triangulation.set_boundary (1);

      }
    else if (base_mesh == "Perforated_strip_tension")
      {
        // Example 2 from the paper: Zhong Z., .... A new numerical method for determining
        // collapse load-carrying capacity of structure made of elasto-plastic material,
        // J. Cent. South Univ. (2014) 21: 398-404
        AssertThrow (dim == 3, ExcNotImplemented());

        const int dim_2d = 2;
        const Point<dim_2d> center_2d(0, 0);
        const double inner_radius = 0.05,
                     outer_radius = 0.1,
                     height = 0.18,
                     thickness = 0.004;
//                   thickness = 0.01;

        Triangulation<dim_2d> triangulation_1,
                      triangulation_2,
                      triangulation_2d;

        const double eps = 1e-7 * inner_radius;
        {
          Point<dim_2d> point;

          GridGenerator::quarter_hyper_shell(triangulation_1,
                                             center_2d, inner_radius, outer_radius,
                                             2);

          // Modify the triangulation_1
          typename Triangulation<dim_2d>::active_cell_iterator
          cell = triangulation_1.begin_active(),
          endc = triangulation_1.end();
          std::vector<bool> treated_vertices(triangulation_1.n_vertices(), false);
          for (; cell != endc; ++cell)
            {
              for (unsigned int f=0; f<GeometryInfo<dim_2d>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary() && cell->face(f)->center()(0)>eps &&
                    cell->face(f)->center()(1)>eps )
                  {
                    // distance of the face center from the center
                    point(0) = cell->face(f)->center()(0) - center_2d(0);
                    point(1) = cell->face(f)->center()(1) - center_2d(1);
                    if ( point.norm() > (inner_radius + eps) )
                      {
                        for (unsigned int v=0; v < GeometryInfo<dim_2d>::vertices_per_face; ++v)
                          {
                            unsigned int vv = cell->face(f)->vertex_index(v);
                            if (treated_vertices[vv] == false)
                              {
                                treated_vertices[vv] = true;
                                if (vv==1)
                                  {
                                    cell->face(f)->vertex(v) = center_2d+Point<dim_2d>(outer_radius,outer_radius);
                                  }
                              }
                          }
                      }

                  }
            }

        }

        // Make the triangulation_2, a rectangular above the triangulation_1
        {
          const Point<dim_2d> point1 (0, outer_radius),
                point2 (outer_radius, height);

          GridGenerator::hyper_rectangle(triangulation_2, point1, point2);

        }

        // make the triangulation_2d and refine it
        {
          // Merge the two triangulation_1 and triangulation_2
          GridGenerator::merge_triangulations(triangulation_1, triangulation_2, triangulation_2d);

          // Assign boundary indicators to the boundary faces
          /*
           *
           *    /\ y
           *     |
           *      _____3_____
           *     |          |
           *     |          |
           *   4 |          |
           *     |          |
           *     |          | 2
           *     |_         |
           *        \       |
           *      10 \      |
           *         |______|   ____________\  x
           *            1                   /
           */
          {
            typename Triangulation<dim_2d>::active_cell_iterator
            cell = triangulation_2d.begin_active(),
            endc = triangulation_2d.end();
            for (; cell != endc; ++cell)
              {
                for (unsigned int f=0; f<GeometryInfo<dim_2d>::faces_per_cell; ++f)
                  {
                    if (cell->face(f)->at_boundary())
                      {
                        if ( std::fabs(cell->face(f)->center()(1)) < eps )
                          {
                            cell->face(f)->set_boundary_indicator(1);
                          }
                        else if ( std::fabs(cell->face(f)->center()(0)-outer_radius) < eps )
                          {
                            cell->face(f)->set_boundary_indicator(2);
                          }
                        else if ( std::fabs(cell->face(f)->center()(1)-height) < eps )
                          {
                            cell->face(f)->set_boundary_indicator(3);
                          }
                        else if ( std::fabs(cell->face(f)->center()(0)) < eps )
                          {
                            cell->face(f)->set_boundary_indicator(4);
                          }
                        else
                          {
                            cell->face(f)->set_all_boundary_indicators(10);
                          }

                      }
                  }
              }

          }

          const HyperBallBoundary<dim_2d> inner_boundary_description(center_2d, inner_radius);
          triangulation_2d.set_boundary (10, inner_boundary_description);

          triangulation_2d.refine_global(3);

          triangulation_2d.set_boundary (10);
        }

        // Extrude the triangulation_2d and make it 3d
//      GridGenerator::extrude_triangulation(triangulation_2d,
//                                           2, thickness, triangulation);
        extrude_triangulation(triangulation_2d,
                              2, thickness, triangulation);

        // Assign boundary indicators to the boundary faces
        /*
         *
         *    /\ y
         *     |
         *      _____3_____
         *     |          |
         *     |          |
         *   4 |          |
         *     |    5|6   |
         *     |          | 2
         *     |_         |
         *        \       |
         *      10 \      |
         *         |______|   ____________\  x
         *            1                   /
         */
        {
          Point<dim> dist_vector;
          Point<dim> center(center_2d(0), center_2d(1), 0);

          typename Triangulation<dim>::active_cell_iterator
          cell = triangulation.begin_active(),
          endc = triangulation.end();
          for (; cell != endc; ++cell)
            {
              for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                {
                  if (cell->face(f)->at_boundary())
                    {
                      dist_vector = cell->face(f)->center() - center;

                      if ( std::fabs(dist_vector(1)) < eps )
                        {
                          cell->face(f)->set_boundary_indicator(1);
                        }
                      else if ( std::fabs(dist_vector(0)-outer_radius) < eps )
                        {
                          cell->face(f)->set_boundary_indicator(2);
                        }
                      else if ( std::fabs(dist_vector(1)-height) < eps )
                        {
                          cell->face(f)->set_boundary_indicator(3);
                        }
                      else if ( std::fabs(dist_vector(0)) < eps )
                        {
                          cell->face(f)->set_boundary_indicator(4);
                        }
                      else if ( std::fabs(dist_vector(2)) < eps )
                        {
                          cell->face(f)->set_boundary_indicator(5);
                        }
                      else if ( std::fabs(dist_vector(2)-thickness) < eps )
                        {
                          cell->face(f)->set_boundary_indicator(6);
                        }
                      else
                        {
                          cell->face(f)->set_all_boundary_indicators(10);
                        }

                    }
                }
            }

        }

        const CylinderBoundary<dim> inner_boundary_description(inner_radius, 2);
        triangulation.set_boundary (10, inner_boundary_description);

        triangulation.refine_global(n_initial_global_refinements);

        triangulation.set_boundary (10);

      }
    else if (base_mesh == "Cantiliver_beam_3d")
      {
        // A rectangular tube made of Aluminium
        // http://www.google.de/imgres?imgurl=http%3A%2F%2Fwww.americanaluminum.com%2Fimages%2Fstockshape-rectangletube.gif&imgrefurl=http%3A%2F%2Fwww.americanaluminum.com%2Fstandard%2FrectangleTube&h=280&w=300&tbnid=VPDNh4-DJz4wyM%3A&zoom=1&docid=9DoGJCkOeFqiSM&ei=L1AuVfG5GMvtO7DggdAF&tbm=isch&client=ubuntu&iact=rc&uact=3&dur=419&page=1&start=0&ndsp=33&ved=0CGYQrQMwFQ
        // approximation of beam 17250
        // units are in meter

        AssertThrow (dim == 3, ExcNotImplemented());

        const int dim_2d = 2;

        const double length = .7,
                     width = 80e-3,
                     height = 200e-3,
                     thickness_web = 10e-3,
                     thickness_flange = 10e-3;

        Triangulation<dim_2d> triangulation_b,
                      triangulation_t,
                      triangulation_l,
                      triangulation_r,
                      triangulation_2d;

        const double eps = 1e-7 * width;
        // Make the triangulation_b, a rectangular at the bottom of rectangular tube
        {
          const Point<dim_2d> point1 (-width/2, -height/2),
                point2 (width/2, -(height/2)+thickness_flange);

          std::vector<unsigned int> repetitions(dim_2d);
          repetitions[0] = 8;
          repetitions[1] = 1;

          GridGenerator::subdivided_hyper_rectangle(triangulation_b, repetitions, point1, point2);
        }

        // Make the triangulation_t, a rectangular at the top of rectangular tube
        {
          const Point<dim_2d> point1 (-width/2, (height/2)-thickness_flange),
                point2 (width/2, height/2);

          std::vector<unsigned int> repetitions(dim_2d);
          repetitions[0] = 8;
          repetitions[1] = 1;

          GridGenerator::subdivided_hyper_rectangle(triangulation_t, repetitions, point1, point2);
        }

        // Make the triangulation_l, a rectangular at the left of rectangular tube
        {
          const Point<dim_2d> point1 (-width/2, -(height/2)+thickness_flange),
                point2 (-(width/2)+thickness_web, (height/2)-thickness_flange);

          std::vector<unsigned int> repetitions(dim_2d);
          repetitions[0] = 1;
          repetitions[1] = 18;

          GridGenerator::subdivided_hyper_rectangle(triangulation_l, repetitions, point1, point2);
        }

        // Make the triangulation_r, a rectangular at the right of rectangular tube
        {
          const Point<dim_2d> point1 ((width/2)-thickness_web, -(height/2)+thickness_flange),
                point2 (width/2, (height/2)-thickness_flange);

          std::vector<unsigned int> repetitions(dim_2d);
          repetitions[0] = 1;
          repetitions[1] = 18;

          GridGenerator::subdivided_hyper_rectangle(triangulation_r, repetitions, point1, point2);
        }

        // make the triangulation_2d
        {
          // merging every two triangles to make triangulation_2d
          Triangulation<dim_2d> triangulation_bl,
                        triangulation_blr;

          GridGenerator::merge_triangulations(triangulation_b, triangulation_l, triangulation_bl);
          GridGenerator::merge_triangulations(triangulation_bl, triangulation_r, triangulation_blr);
          GridGenerator::merge_triangulations(triangulation_blr, triangulation_t, triangulation_2d);
        }

        // Extrude the triangulation_2d and make it 3d
        const unsigned int n_slices = length*1000/20 + 1;
        extrude_triangulation(triangulation_2d,
                              n_slices, length, triangulation);

        // Assign boundary indicators to the boundary faces
        /*
         *
         *                     A
         *            ---------*----------
         *           /                   /|
         *          /                   / |
         *         /                   /  |
         *        /       2    length /   |
         *       /                   /    |
         *      /                   /     |
         *     /                   /      |
         *    /        width      /       |
         *    --------------------        |
         *    | --------1-------. |       |
         *    | :               : |       |
         *    | :               : |h      |
         *    | :      y   z    : |e      |
         *    | :       | /     : |i     /
         *    |1:       |___ x  :1|g    /
         *    | :               : |h   /
         *    | :               : |t  /
         *    | :               : |  /
         *    | :               : | /
         *    | ----------------- |/
         *    ---------1----------/
         *
         *   face id:
         *   Essential boundary condition:
         *   1: z = 0: clamped, fixed in x, y and z directions
         *   Natural/Newmann boundary condition:
         *   2: y = height/2: traction face: pressure on the surface
         *   Quantity of interest:
         *   displacement at Point A (x=0, y=height/2, z=length)
         */
        {
          Point<dim> dist_vector;
          Point<dim> center(0, 0, 0);

          typename Triangulation<dim>::active_cell_iterator
          cell = triangulation.begin_active(),
          endc = triangulation.end();
          for (; cell != endc; ++cell)
            {
              for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                {
                  if (cell->face(f)->at_boundary())
                    {
                      dist_vector = cell->face(f)->center() - center;

                      if ( std::fabs(dist_vector(2)) < eps )
                        {
                          cell->face(f)->set_boundary_indicator(1);
                        }
                      else if ( std::fabs(dist_vector(1)-(height/2)) < eps )
                        {
                          cell->face(f)->set_boundary_indicator(2);
                        }
                      else
                        {
                          cell->face(f)->set_all_boundary_indicators(0);
                        }

                    }
                }
            }

        }

        triangulation.refine_global(n_initial_global_refinements);

      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }

    pcout << "    Number of active cells:       "
          << triangulation.n_active_cells()
          << std::endl;
  }



  // @sect4{PlasticityContactProblem::setup_system}

  // The next piece in the puzzle is to set up the DoFHandler, resize
  // vectors and take care of various other status variables such as
  // index sets and constraint matrices.
  //
  // In the following, each group of operations is put into a brace-enclosed
  // block that is being timed by the variable declared at the top of the
  // block (the constructor of the TimerOutput::Scope variable starts the
  // timed section, the destructor that is called at the end of the block
  // stops it again).
  template <int dim>
  void
  ElastoPlasticProblem<dim>::setup_system ()
  {
    /* setup dofs and get index sets for locally owned and relevant dofs */
    TimerOutput::Scope t(computing_timer, "Setup");
    {
      TimerOutput::Scope t(computing_timer, "Setup: distribute DoFs");
      dof_handler.distribute_dofs(fe);
      pcout << "    Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

      locally_owned_dofs = dof_handler.locally_owned_dofs();
      locally_relevant_dofs.clear();
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
    }

    /* setup hanging nodes and Dirichlet constraints */
    {
      TimerOutput::Scope t(computing_timer, "Setup: constraints");
      constraints_hanging_nodes.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler,
                                              constraints_hanging_nodes);
      constraints_hanging_nodes.close();

      pcout << "   Number of active cells: "
            << triangulation.n_global_active_cells() << std::endl
            << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

      compute_dirichlet_constraints();
    }

    /* initialization of vectors*/
    {
      TimerOutput::Scope t(computing_timer, "Setup: vectors");
      if (timestep_no==1 || current_refinement_cycle!=0)
        {
          solution.reinit(locally_relevant_dofs, mpi_communicator);
        }
      incremental_displacement.reinit(locally_relevant_dofs, mpi_communicator);
      newton_rhs.reinit(locally_owned_dofs, mpi_communicator);
      newton_rhs_residual.reinit(locally_owned_dofs, mpi_communicator);
      fraction_of_plastic_q_points_per_cell.reinit(triangulation.n_active_cells());
    }

    // Finally, we set up sparsity patterns and matrices.
    // We temporarily (ab)use the system matrix to also build the (diagonal)
    // matrix that we use in eliminating degrees of freedom that are in contact
    // with the obstacle, but we then immediately set the Newton matrix back
    // to zero.
    {
      TimerOutput::Scope t(computing_timer, "Setup: matrix");
      TrilinosWrappers::SparsityPattern sp(locally_owned_dofs,
                                           mpi_communicator);

      DoFTools::make_sparsity_pattern(dof_handler, sp,
                                      constraints_dirichlet_and_hanging_nodes, false,
                                      this_mpi_process);
      sp.compress();
      newton_matrix.reinit(sp);
    }
  }


  // @sect4{PlasticityContactProblem::compute_dirichlet_constraints}

  // This function, broken out of the preceding one, computes the constraints
  // associated with Dirichlet-type boundary conditions and puts them into the
  // <code>constraints_dirichlet_and_hanging_nodes</code> variable by merging
  // with the constraints that come from hanging nodes.
  //
  // As laid out in the introduction, we need to distinguish between two
  // cases:
  // - If the domain is a box, we set the displacement to zero at the bottom,
  //   and allow vertical movement in z-direction along the sides. As
  //   shown in the <code>make_grid()</code> function, the former corresponds
  //   to boundary indicator 6, the latter to 8.
  // - If the domain is a half sphere, then we impose zero displacement along
  //   the curved part of the boundary, associated with boundary indicator zero.
  template <int dim>
  void
  ElastoPlasticProblem<dim>::compute_dirichlet_constraints ()
  {
    constraints_dirichlet_and_hanging_nodes.reinit(locally_relevant_dofs);
    constraints_dirichlet_and_hanging_nodes.merge(constraints_hanging_nodes);

    std::vector<bool> component_mask(dim);

    if (base_mesh == "Timoshenko beam")
      {
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                 constraints_dirichlet_and_hanging_nodes,
                                                 ComponentMask());
      }
    else if (base_mesh == "Thick_tube_internal_pressure")
      {
        // the boundary x = 0
        component_mask[0] = true;
        component_mask[1] = false;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  2,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes,
                                                  component_mask);
        // the boundary y = 0
        component_mask[0] = false;
        component_mask[1] = true;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  3,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes,
                                                  component_mask);
      }
    else if (base_mesh == "Perforated_strip_tension")
      {
        // the boundary x = 0
        component_mask[0] = true;
        component_mask[1] = false;
        component_mask[2] = false;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  4,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes,
                                                  component_mask);
        // the boundary y = 0
        component_mask[0] = false;
        component_mask[1] = true;
        component_mask[2] = false;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes,
                                                  component_mask);
        // the boundary y = imposed incremental displacement
        component_mask[0] = false;
        component_mask[1] = true;
        component_mask[2] = false;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  3,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes,
                                                  component_mask);
      }
    else if (base_mesh == "Cantiliver_beam_3d")
      {
        // the boundary x = y = z = 0
        component_mask[0] = true;
        component_mask[1] = true;
        component_mask[2] = true;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  EquationData::IncrementalBoundaryValues<dim>(present_time, end_time),
                                                  constraints_dirichlet_and_hanging_nodes,
                                                  component_mask);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }


    constraints_dirichlet_and_hanging_nodes.close();
  }


  // @sect4{PlasticityContactProblem::assemble_newton_system}

  // Given the complexity of the problem, it may come as a bit of a surprise
  // that assembling the linear system we have to solve in each Newton iteration
  // is actually fairly straightforward. The following function builds the Newton
  // right hand side and Newton matrix. It looks fairly innocent because the
  // heavy lifting happens in the call to
  // <code>ConstitutiveLaw::get_linearized_stress_strain_tensors()</code> and in
  // particular in ConstraintMatrix::distribute_local_to_global(), using the
  // constraints we have previously computed.
  template <int dim>
  void
  ElastoPlasticProblem<dim>::
  assemble_newton_system (const TrilinosWrappers::MPI::Vector &linearization_point,
                          const TrilinosWrappers::MPI::Vector &delta_linearization_point)
  {
    TimerOutput::Scope t(computing_timer, "Assembling");

    types::boundary_id traction_surface_id;
    if (base_mesh == "Timoshenko beam")
      {
        traction_surface_id = 5;
      }
    else if (base_mesh == "Thick_tube_internal_pressure")
      {
        traction_surface_id = 0;
      }
    else if (base_mesh == "Cantiliver_beam_3d")
      {
        traction_surface_id = 2;
      }

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_values_face(fe, face_quadrature_formula,
                                     update_values | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();


    const EquationData::BodyForce<dim>     body_force;
    std::vector<Vector<double> >           body_force_values(n_q_points,
                                                             Vector<double>(dim));

    const EquationData::
    IncrementalBoundaryForce<dim>          boundary_force(present_time, end_time);
    std::vector<Vector<double> >           boundary_force_values(n_face_q_points,
        Vector<double>(dim));

    FullMatrix<double>                     cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>                         cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index>   local_dof_indices(dofs_per_cell);

//    std::vector<SymmetricTensor<2, dim> > strain_tensor(n_q_points);
    std::vector<SymmetricTensor<2, dim> > incremental_strain_tensor(n_q_points);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    const FEValuesExtractors::Vector displacement(0);

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs = 0;

          fe_values[displacement].get_function_symmetric_gradients(delta_linearization_point,
                                                                   incremental_strain_tensor);

          // For assembling the local right hand side contributions, we need
          // to access the prior linearized stress value in this quadrature
          // point. To get it, we use the user pointer of this cell that
          // points into the global array to the quadrature point data
          // corresponding to the first quadrature point of the present cell,
          // and then add an offset corresponding to the index of the
          // quadrature point we presently consider:
          const PointHistory<dim> *local_quadrature_points_history
            = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
          Assert (local_quadrature_points_history >=
                  &quadrature_point_history.front(),
                  ExcInternalError());
          Assert (local_quadrature_points_history <
                  &quadrature_point_history.back(),
                  ExcInternalError());

          // In addition, we need the values of the external body forces at
          // the quadrature points on this cell:
          body_force.vector_value_list(fe_values.get_quadrature_points(),
                                       body_force_values);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              SymmetricTensor<2, dim> tmp_strain_tensor_qpoint;
              tmp_strain_tensor_qpoint = local_quadrature_points_history[q_point].old_strain
                                         + incremental_strain_tensor[q_point];

              SymmetricTensor<4, dim> stress_strain_tensor_linearized;
              SymmetricTensor<4, dim> stress_strain_tensor;
              constitutive_law.get_linearized_stress_strain_tensors(tmp_strain_tensor_qpoint,
                                                                    stress_strain_tensor_linearized,
                                                                    stress_strain_tensor);

              Tensor<1, dim> rhs_values_body_force;
              for (unsigned int i = 0; i < dim; ++i)
                {
                  rhs_values_body_force[i] = body_force_values[q_point][i];
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  // Having computed the stress-strain tensor and its linearization,
                  // we can now put together the parts of the matrix and right hand side.
                  // In both, we need the linearized stress-strain tensor times the
                  // symmetric gradient of $\varphi_i$, i.e. the term $I_\Pi\varepsilon(\varphi_i)$,
                  // so we introduce an abbreviation of this term. Recall that the
                  // matrix corresponds to the bilinear form
                  // $A_{ij}=(I_\Pi\varepsilon(\varphi_i),\varepsilon(\varphi_j))$ in the
                  // notation of the accompanying publication, whereas the right
                  // hand side is $F_i=([I_\Pi-P_\Pi C]\varepsilon(\varphi_i),\varepsilon(\mathbf u))$
                  // where $u$ is the current linearization points (typically the last solution).
                  // This might suggest that the right hand side will be zero if the material
                  // is completely elastic (where $I_\Pi=P_\Pi$) but this ignores the fact
                  // that the right hand side will also contain contributions from
                  // non-homogeneous constraints due to the contact.
                  //
                  // The code block that follows this adds contributions that are due to
                  // boundary forces, should there be any.
                  const SymmetricTensor<2, dim>
                  stress_phi_i = stress_strain_tensor_linearized
                                 * fe_values[displacement].symmetric_gradient(i, q_point);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += (stress_phi_i
                                          * fe_values[displacement].symmetric_gradient(j, q_point)
                                          * fe_values.JxW(q_point));

                  cell_rhs(i) += (
                                   ( stress_phi_i
                                     * incremental_strain_tensor[q_point] )
                                   -
                                   ( ( stress_strain_tensor
                                       * fe_values[displacement].symmetric_gradient(i, q_point))
                                     * tmp_strain_tensor_qpoint )
                                   +
                                   ( fe_values[displacement].value(i, q_point)
                                     * rhs_values_body_force )
                                 ) * fe_values.JxW(q_point);

                }
            }

          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary()
                &&
                cell->face(face)->boundary_indicator() == traction_surface_id)
              {
                fe_values_face.reinit(cell, face);

                boundary_force.vector_value_list(fe_values_face.get_quadrature_points(),
                                                 boundary_force_values);

                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                  {
                    Tensor<1, dim> rhs_values;
                    for (unsigned int i = 0; i < dim; ++i)
                      {
                        rhs_values[i] = boundary_force_values[q_point][i];
                      }
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      cell_rhs(i) += (fe_values_face[displacement].value(i, q_point)
                                      * rhs_values
                                      * fe_values_face.JxW(q_point));
                  }
              }

          cell->get_dof_indices(local_dof_indices);
          constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_matrix, cell_rhs,
              local_dof_indices,
              newton_matrix,
              newton_rhs,
              true);

        }

    newton_matrix.compress(VectorOperation::add);
    newton_rhs.compress(VectorOperation::add);
  }



  // @sect4{PlasticityContactProblem::compute_nonlinear_residual}

  // The following function computes the nonlinear residual of the equation
  // given the current solution (or any other linearization point). This
  // is needed in the linear search algorithm where we need to try various
  // linear combinations of previous and current (trial) solution to
  // compute the (real, globalized) solution of the current Newton step.
  //
  // That said, in a slight abuse of the name of the function, it actually
  // does significantly more. For example, it also computes the vector
  // that corresponds to the Newton residual but without eliminating
  // constrained degrees of freedom. We need this vector to compute contact
  // forces and, ultimately, to compute the next active set. Likewise, by
  // keeping track of how many quadrature points we encounter on each cell
  // that show plastic yielding, we also compute the
  // <code>fraction_of_plastic_q_points_per_cell</code> vector that we
  // can later output to visualize the plastic zone. In both of these cases,
  // the results are not necessary as part of the line search, and so we may
  // be wasting a small amount of time computing them. At the same time, this
  // information appears as a natural by-product of what we need to do here
  // anyway, and we want to collect it once at the end of each Newton
  // step, so we may as well do it here.
  //
  // The actual implementation of this function should be rather obvious:
  template <int dim>
  void
  ElastoPlasticProblem<dim>::
  compute_nonlinear_residual (const TrilinosWrappers::MPI::Vector &linearization_point)
  {
    types::boundary_id traction_surface_id;
    if (base_mesh == "Timoshenko beam")
      {
        traction_surface_id = 5;
      }
    else if (base_mesh == "Thick_tube_internal_pressure")
      {
        traction_surface_id = 0;
      }
    else if (base_mesh == "Cantiliver_beam_3d")
      {
        traction_surface_id = 2;
      }

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points |
                            update_JxW_values);

    FEFaceValues<dim> fe_values_face(fe, face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                     update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    const EquationData::BodyForce<dim>     body_force;
    std::vector<Vector<double> >           body_force_values(n_q_points,
                                                             Vector<double>(dim));

    const EquationData::
    IncrementalBoundaryForce<dim>          boundary_force(present_time, end_time);
    std::vector<Vector<double> >           boundary_force_values(n_face_q_points,
        Vector<double>(dim));

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector displacement(0);

    newton_rhs_residual = 0;

    fraction_of_plastic_q_points_per_cell = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    unsigned int cell_number = 0;
    for (; cell != endc; ++cell, ++cell_number)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_rhs = 0;

          std::vector<SymmetricTensor<2, dim> > strain_tensors(n_q_points);
          fe_values[displacement].get_function_symmetric_gradients(linearization_point,
                                                                   strain_tensors);

          body_force.vector_value_list(fe_values.get_quadrature_points(),
                                       body_force_values);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              SymmetricTensor<4, dim> stress_strain_tensor;
              const bool q_point_is_plastic
                = constitutive_law.get_stress_strain_tensor(strain_tensors[q_point],
                                                            stress_strain_tensor);
              if (q_point_is_plastic)
                ++fraction_of_plastic_q_points_per_cell(cell_number);

              Tensor<1, dim> rhs_values_body_force;
              for (unsigned int i = 0; i < dim; ++i)
                {
                  rhs_values_body_force[i] = body_force_values[q_point][i];
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  cell_rhs(i) += (fe_values[displacement].value(i, q_point)
                                  * rhs_values_body_force
                                  -
                                  strain_tensors[q_point]
                                  * stress_strain_tensor
                                  * fe_values[displacement].symmetric_gradient(i, q_point)
                                 )
                                 * fe_values.JxW(q_point);

                  Tensor<1, dim> rhs_values;
                  rhs_values = 0;
                  cell_rhs(i) += (fe_values[displacement].value(i, q_point)
                                  * rhs_values
                                  * fe_values.JxW(q_point));
                }
            }

          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary()
                && cell->face(face)->boundary_indicator() == traction_surface_id)
              {
                fe_values_face.reinit(cell, face);

                boundary_force.vector_value_list(fe_values_face.get_quadrature_points(),
                                                 boundary_force_values);

                for (unsigned int q_point = 0; q_point < n_face_q_points;
                     ++q_point)
                  {
                    Tensor<1, dim> rhs_values;
                    for (unsigned int i = 0; i < dim; ++i)
                      {
                        rhs_values[i] = boundary_force_values[q_point][i];
                      }
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      cell_rhs(i) += (fe_values_face[displacement].value(i, q_point) * rhs_values
                                      * fe_values_face.JxW(q_point));
                  }
              }

          cell->get_dof_indices(local_dof_indices);
          constraints_dirichlet_and_hanging_nodes.distribute_local_to_global(cell_rhs,
              local_dof_indices,
              newton_rhs_residual);

        }

    fraction_of_plastic_q_points_per_cell /= quadrature_formula.size();
    newton_rhs_residual.compress(VectorOperation::add);

  }





  // @sect4{PlasticityContactProblem::solve_newton_system}

  // The last piece before we can discuss the actual Newton iteration
  // on a single mesh is the solver for the linear systems. There are
  // a couple of complications that slightly obscure the code, but
  // mostly it is just setup then solve. Among the complications are:
  //
  // - For the hanging nodes we have to apply
  //   the ConstraintMatrix::set_zero function to newton_rhs.
  //   This is necessary if a hanging node with solution value $x_0$
  //   has one neighbor with value $x_1$ which is in contact with the
  //   obstacle and one neighbor $x_2$ which is not in contact. Because
  //   the update for the former will be prescribed, the hanging node constraint
  //   will have an inhomogeneity and will look like $x_0 = x_1/2 + \text{gap}/2$.
  //   So the corresponding entries in the
  //   ride-hang-side are non-zero with a
  //   meaningless value. These values we have to
  //   to set to zero.
  // - Like in step-40, we need to shuffle between vectors that do and do
  //   do not have ghost elements when solving or using the solution.
  //
  // The rest of the function is similar to step-40 and
  // step-41 except that we use a BiCGStab solver
  // instead of CG. This is due to the fact that for very small hardening
  // parameters $\gamma$, the linear system becomes almost semidefinite though
  // still symmetric. BiCGStab appears to have an easier time with such linear
  // systems.
  template <int dim>
  void
  ElastoPlasticProblem<dim>::solve_newton_system ()
  {
    TimerOutput::Scope t(computing_timer, "Solve");

    TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
    distributed_solution = incremental_displacement;

    constraints_hanging_nodes.set_zero(distributed_solution);
    constraints_hanging_nodes.set_zero(newton_rhs);

    // ------- Solver Bicgstab --- Preconditioner AMG -------------------
//    TrilinosWrappers::PreconditionAMG preconditioner;
//    {
//      TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");
//
//      std::vector<std::vector<bool> > constant_modes;
//      DoFTools::extract_constant_modes(dof_handler, ComponentMask(),
//                                       constant_modes);
//
//      TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
//      additional_data.constant_modes = constant_modes;
//      additional_data.elliptic = true;
//      additional_data.n_cycles = 1;
//      additional_data.w_cycle = false;
//      additional_data.output_details = false;
//      additional_data.smoother_sweeps = 2;
//      additional_data.aggregation_threshold = 1e-2;
//
//      preconditioner.initialize(newton_matrix, additional_data);
//    }

//    {
//      TimerOutput::Scope t(computing_timer, "Solve: iterate");
//
//      TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
//
////      const double relative_accuracy = 1e-8;
//      const double relative_accuracy = 1e-2;
//      const double solver_tolerance  = relative_accuracy
//                                       * newton_matrix.residual(tmp, distributed_solution,
//                                                                newton_rhs);
//
//      SolverControl solver_control(newton_matrix.m(),
//                                   solver_tolerance);
//      SolverBicgstab<TrilinosWrappers::MPI::Vector> solver(solver_control);
//      solver.solve(newton_matrix, distributed_solution,
//                   newton_rhs, preconditioner);
//
//      pcout << "         Error: " << solver_control.initial_value()
//            << " -> " << solver_control.last_value() << " in "
//            << solver_control.last_step() << " Bicgstab iterations."
//            << std::endl;
//    }

    // ------- Solver CG --- Preconditioner SSOR -------------------
    TrilinosWrappers::PreconditionSSOR preconditioner;
    {
      TimerOutput::Scope t(computing_timer, "Solve: setup preconditioner");

      TrilinosWrappers::PreconditionSSOR::AdditionalData additional_data;
      preconditioner.initialize(newton_matrix, additional_data);
    }

    {
      TimerOutput::Scope t(computing_timer, "Solve: iterate");

      TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);

//      const double relative_accuracy = 1e-8;
      const double relative_accuracy = 1e-2;
      const double solver_tolerance  = relative_accuracy
                                       * newton_matrix.residual(tmp, distributed_solution,
                                                                newton_rhs);

//      SolverControl solver_control(newton_matrix.m(),
//                                   solver_tolerance);
      SolverControl solver_control(10*newton_matrix.m(),
                                   solver_tolerance);
      SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
      solver.solve(newton_matrix, distributed_solution,
                   newton_rhs, preconditioner);

      pcout << "         Error: " << solver_control.initial_value()
            << " -> " << solver_control.last_value() << " in "
            << solver_control.last_step() << " CG iterations."
            << std::endl;
    }
    // ........................................................

    constraints_dirichlet_and_hanging_nodes.distribute(distributed_solution);

    incremental_displacement = distributed_solution;
  }


  // @sect4{PlasticityContactProblem::solve_newton}

  // This is, finally, the function that implements the damped Newton method
  // on the current mesh. There are two nested loops: the outer loop for the Newton
  // iteration and the inner loop for the line search which
  // will be used only if necessary. To obtain a good and reasonable
  // starting value we solve an elastic problem in very first Newton step on each
  // mesh (or only on the first mesh if we transfer solutions between meshes). We
  // do so by setting the yield stress to an unreasonably large value in these
  // iterations and then setting it back to the correct value in subsequent
  // iterations.
  //
  // Other than this, the top part of this function should be reasonably
  // obvious:
  template <int dim>
  void
  ElastoPlasticProblem<dim>::solve_newton ()
  {
    TrilinosWrappers::MPI::Vector old_solution(locally_owned_dofs, mpi_communicator);
    TrilinosWrappers::MPI::Vector residual(locally_owned_dofs, mpi_communicator);
    TrilinosWrappers::MPI::Vector tmp_vector(locally_owned_dofs, mpi_communicator);
    TrilinosWrappers::MPI::Vector locally_relevant_tmp_vector(locally_relevant_dofs, mpi_communicator);
    TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
    TrilinosWrappers::MPI::Vector tmp_solution(locally_owned_dofs, mpi_communicator);

    double residual_norm;
    double previous_residual_norm = -std::numeric_limits<double>::max();

    double disp_norm,
           previous_disp_norm = 0;

    const double correct_sigma = sigma_0;

    const unsigned int max_newton_iter = 100;

    for (unsigned int newton_step = 1; newton_step <= max_newton_iter; ++newton_step)
      {
        if (newton_step == 1
            &&
            ((transfer_solution && timestep_no == 1)
             ||
             !transfer_solution))
          constitutive_law.set_sigma_0(1e+10);
        else
          constitutive_law.set_sigma_0(correct_sigma);

        pcout << " " << std::endl;
        pcout << "   Newton iteration " << newton_step << std::endl;

        pcout << "      Assembling system... " << std::endl;
        newton_matrix       = 0;
        newton_rhs          = 0;
        newton_rhs_residual = 0;

        tmp_solution = solution;
        tmp_solution += incremental_displacement;
        assemble_newton_system(tmp_solution,
                               incremental_displacement);

        pcout << "      Solving system... " << std::endl;
        solve_newton_system();

        // It gets a bit more hairy after we have computed the
        // trial solution $\tilde{\mathbf u}$ of the current Newton step.
        // We handle a highly nonlinear problem so we have to damp
        // Newton's method using a line search. To understand how we do this,
        // recall that in our formulation, we compute a trial solution
        // in each Newton step and not the update between old and new solution.
        // Since the solution set is a convex set, we will use a line
        // search that tries linear combinations of the
        // previous and the trial solution to guarantee that the
        // damped solution is in our solution set again.
        // At most we apply 5 damping steps.
        //
        // There are exceptions to when we use a line search. First,
        // if this is the first Newton step on any mesh, then we don't have
        // any point to compare the residual to, so we always accept a full
        // step. Likewise, if this is the second Newton step on the first mesh (or
        // the second on any mesh if we don't transfer solutions from
        // mesh to mesh), then we have computed the first of these steps using
        // just an elastic model (see how we set the yield stress sigma to
        // an unreasonably large value above). In this case, the first Newton
        // solution was a purely elastic one, the second one a plastic one,
        // and any linear combination would not necessarily be expected to
        // lie in the feasible set -- so we just accept the solution we just
        // got.
        //
        // In either of these two cases, we bypass the line search and just
        // update residual and other vectors as necessary.
        if ((newton_step==1)
            ||
            (transfer_solution && newton_step == 2 && current_refinement_cycle == 0)
            ||
            (!transfer_solution && newton_step == 2))
          {
            tmp_solution = solution;
            tmp_solution += incremental_displacement;
            compute_nonlinear_residual(tmp_solution);
            old_solution = incremental_displacement;

            residual = newton_rhs_residual;

            residual.compress(VectorOperation::insert);

            residual_norm = residual.l2_norm();

            pcout << "      Accepting Newton solution with residual: "
                  << residual_norm << std::endl;
          }
        else
          {
            for (unsigned int i = 0; i < 5; i++)
              {
                distributed_solution = incremental_displacement;

                const double alpha = std::pow(0.5, static_cast<double>(i));
                tmp_vector = old_solution;
                tmp_vector.sadd(1 - alpha, alpha, distributed_solution);

                TimerOutput::Scope t(computing_timer, "Residual and lambda");

                locally_relevant_tmp_vector = tmp_vector;
                tmp_solution = solution;
                tmp_solution += locally_relevant_tmp_vector;
                compute_nonlinear_residual(tmp_solution);
                residual = newton_rhs_residual;

                residual.compress(VectorOperation::insert);

                residual_norm = residual.l2_norm();

                pcout << "      Residual of the system: "
                      << residual_norm << std::endl
                      << "         with a damping parameter alpha = " << alpha
                      << std::endl;

                if (residual_norm < previous_residual_norm)
                  break;
              }

            incremental_displacement = tmp_vector;
            old_solution = incremental_displacement;
          }

        disp_norm = incremental_displacement.l2_norm();


        // The final step is to check for convergence. If the residual is
        // less than a threshold of $10^{-10}$, then we terminate
        // the iteration on the current mesh:
//        if (residual_norm < 1e-10)
        if (residual_norm < 1e-7)
          break;

        pcout << "    difference of two consecutive incremental displacement l2 norm : "
              << std::abs(disp_norm - previous_disp_norm) << std::endl;
        if ( std::abs(disp_norm - previous_disp_norm) < 1e-10 &&
             (residual_norm < 1e-5 || std::abs(residual_norm - previous_residual_norm)<1e-9) )
          {
            pcout << " Convergence by difference of two consecutive solution! " << std::endl;
            break;
          }


        previous_residual_norm = residual_norm;
        previous_disp_norm = disp_norm;
      }
  }

  // @sect4{PlasticityContactProblem::compute_error}

  template <int dim>
  void
  ElastoPlasticProblem<dim>::compute_error ()
  {
    TrilinosWrappers::MPI::Vector   tmp_solution(locally_owned_dofs, mpi_communicator);
    tmp_solution = solution;
    tmp_solution += incremental_displacement;

    estimated_error_per_cell.reinit (triangulation.n_active_cells());
    if (error_estimation_strategy == ErrorEstimationStrategy::kelly_error)
      {
        KellyErrorEstimator<dim>::estimate(dof_handler,
                                           QGauss<dim - 1>(fe.degree + 2),
                                           typename FunctionMap<dim>::type(),
                                           tmp_solution,
                                           estimated_error_per_cell);

      }
    else if (error_estimation_strategy == ErrorEstimationStrategy::residual_error)
      {
        compute_error_residual(tmp_solution);

      }
    else if (error_estimation_strategy == ErrorEstimationStrategy::weighted_residual_error)
      {
        // make a non-parallel copy of tmp_solution
        Vector<double> copy_solution(tmp_solution);

        // the dual function definition (it should be defined previously, e.g. input file)
        if (base_mesh == "Timoshenko beam")
          {
            double length = .48,
                   depth  = .12;

            const Point<dim> evaluation_point(length, -depth/2);

            DualFunctional::PointValuesEvaluation<dim> dual_functional(evaluation_point);

            DualSolver<dim> dual_solver(triangulation, fe,
                                        copy_solution,
                                        constitutive_law, dual_functional,
                                        timestep_no, output_dir, base_mesh,
                                        present_time, end_time);

            dual_solver.compute_error_DWR (estimated_error_per_cell);

          }
        else if (base_mesh == "Thick_tube_internal_pressure")
          {
            const unsigned int face_id = 0;
            std::vector<std::vector<unsigned int> > comp_stress(dim);
            for (unsigned int i=0; i!=dim; ++i)
              {
                comp_stress[i].resize(dim);
                for (unsigned int j=0; j!=dim; ++j)
                  {
                    comp_stress[i][j] = 1;
                  }
              }

            DualFunctional::MeanStressFace<dim> dual_functional(face_id, comp_stress);

            DualSolver<dim> dual_solver(triangulation, fe,
                                        copy_solution,
                                        constitutive_law, dual_functional,
                                        timestep_no, output_dir, base_mesh,
                                        present_time, end_time);

            dual_solver.compute_error_DWR (estimated_error_per_cell);

          }
        else if (base_mesh == "Perforated_strip_tension")
          {
            // .........................................
            // Mean stress_yy over the bottom boundary
            const unsigned int face_id = 1;
            std::vector<std::vector<unsigned int> > comp_stress(dim);
            for (unsigned int i=0; i!=dim; ++i)
              {
                comp_stress[i].resize(dim);
                for (unsigned int j=0; j!=dim; ++j)
                  {
                    comp_stress[i][j] = 0;
                  }
              }
            comp_stress[1][1] = 1;

            DualFunctional::MeanStressFace<dim> dual_functional(face_id, comp_stress);

            // .........................................

            DualSolver<dim> dual_solver(triangulation, fe,
                                        copy_solution,
                                        constitutive_law, dual_functional,
                                        timestep_no, output_dir, base_mesh,
                                        present_time, end_time);

            dual_solver.compute_error_DWR (estimated_error_per_cell);

          }
        else if (base_mesh == "Cantiliver_beam_3d")
          {
            // Quantity of interest:
            // -----------------------------------------------------------
            // displacement at Point A (x=0, y=height/2, z=length)
            /*
            const double length = .7,
                         height = 200e-3;

            const Point<dim> evaluation_point(0, height/2, length);

            DualFunctional::PointValuesEvaluation<dim> dual_functional(evaluation_point);
            */

            // -----------------------------------------------------------
            // Mean stress at the specified domain is of interest.
            // The interest domains are located on the bottom and top of the flanges
            // close to the clamped face, z = 0
            // top domain: height/2 - thickness_flange <= y <= height/2
            //             0 <= z <= 2 * thickness_flange
            // bottom domain: -height/2 <= y <= -height/2 + thickness_flange
            //             0 <= z <= 2 * thickness_flange

            std::vector<std::vector<unsigned int> > comp_stress(dim);
            for (unsigned int i=0; i!=dim; ++i)
              {
                comp_stress[i].resize(dim);
                for (unsigned int j=0; j!=dim; ++j)
                  {
                    comp_stress[i][j] = 1;
                  }
              }
            DualFunctional::MeanStressDomain<dim> dual_functional(base_mesh, comp_stress);

            // -----------------------------------------------------------

            DualSolver<dim> dual_solver(triangulation, fe,
                                        copy_solution,
                                        constitutive_law, dual_functional,
                                        timestep_no, output_dir, base_mesh,
                                        present_time, end_time);

            dual_solver.compute_error_DWR (estimated_error_per_cell);

          }
        else
          {
            AssertThrow(false, ExcNotImplemented());
          }


      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }


    relative_error = estimated_error_per_cell.l2_norm() / tmp_solution.l2_norm();

    pcout << "Estimated relative error = " << relative_error << std::endl;

  }

  template <int dim>
  void
  ElastoPlasticProblem<dim>::compute_error_residual (const TrilinosWrappers::MPI::Vector &tmp_solution)
  {
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values    |
                            update_gradients |
                            update_hessians  |
                            update_quadrature_points |
                            update_JxW_values);

    const unsigned int n_q_points      = quadrature_formula.size();
    std::vector<SymmetricTensor<2, dim> > strain_tensor(n_q_points);
    SymmetricTensor<4, dim> stress_strain_tensor_linearized;
    SymmetricTensor<4, dim> stress_strain_tensor;
    Tensor<5, dim>          stress_strain_tensor_grad;
    std::vector<std::vector<Tensor<2,dim> > > cell_hessians (n_q_points);
    for (unsigned int i=0; i!=n_q_points; ++i)
      {
        cell_hessians[i].resize (dim);
      }
    const EquationData::BodyForce<dim> body_force;

    std::vector<Vector<double> > body_force_values (n_q_points, Vector<double>(dim));
    const FEValuesExtractors::Vector displacement(0);


    FEFaceValues<dim> fe_face_values_cell(fe, face_quadrature_formula,
                                          update_values           |
                                          update_quadrature_points|
                                          update_gradients        |
                                          update_JxW_values       |
                                          update_normal_vectors),
                                          fe_face_values_neighbor (fe, face_quadrature_formula,
                                              update_values     |
                                              update_gradients  |
                                              update_JxW_values |
                                              update_normal_vectors);
    FESubfaceValues<dim> fe_subface_values_cell (fe, face_quadrature_formula,
                                                 update_gradients);

    const unsigned int n_face_q_points = face_quadrature_formula.size();
    std::vector<Vector<double> > jump_residual (n_face_q_points, Vector<double>(dim));
    std::vector<std::vector<Tensor<1,dim> > > cell_grads(n_face_q_points);
    for (unsigned int i=0; i!=n_face_q_points; ++i)
      {
        cell_grads[i].resize (dim);
      }
    std::vector<std::vector<Tensor<1,dim> > > neighbor_grads(n_face_q_points);
    for (unsigned int i=0; i!=n_face_q_points; ++i)
      {
        neighbor_grads[i].resize (dim);
      }
    SymmetricTensor<2, dim> q_cell_strain_tensor;
    SymmetricTensor<2, dim> q_neighbor_strain_tensor;
    SymmetricTensor<4, dim> cell_stress_strain_tensor;
    SymmetricTensor<4, dim> neighbor_stress_strain_tensor;


    typename std::map<typename DoFHandler<dim>::face_iterator, Vector<double> >
    face_integrals;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          for (unsigned int face_no=0;
               face_no<GeometryInfo<dim>::faces_per_cell;
               ++face_no)
            {
              face_integrals[cell->face(face_no)].reinit (dim);
              face_integrals[cell->face(face_no)] = -1e20;
            }
        }

    std::vector<Vector<float> > error_indicators_vector;
    error_indicators_vector.resize( triangulation.n_active_cells(),
                                    Vector<float>(dim) );

    // ----------------- estimate_some -------------------------
    cell = dof_handler.begin_active();
    unsigned int present_cell = 0;
    for (; cell!=endc; ++cell, ++present_cell)
      if (cell->is_locally_owned())
        {
          // --------------- integrate_over_cell -------------------
          fe_values.reinit(cell);
          body_force.vector_value_list(fe_values.get_quadrature_points(),
                                       body_force_values);
          fe_values[displacement].get_function_symmetric_gradients(tmp_solution,
                                                                   strain_tensor);
          fe_values.get_function_hessians(tmp_solution, cell_hessians);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              constitutive_law.get_linearized_stress_strain_tensors(strain_tensor[q_point],
                                                                    stress_strain_tensor_linearized,
                                                                    stress_strain_tensor);
              constitutive_law.get_grad_stress_strain_tensor(strain_tensor[q_point],
                                                             cell_hessians[q_point],
                                                             stress_strain_tensor_grad);

              for (unsigned int i=0; i!=dim; ++i)
                {
                  error_indicators_vector[present_cell](i) +=
                    body_force_values[q_point](i)*fe_values.JxW(q_point);
                  for (unsigned int j=0; j!=dim; ++j)
                    {
                      for (unsigned int k=0; k!=dim; ++k)
                        {
                          for (unsigned int l=0; l!=dim; ++l)
                            {
                              error_indicators_vector[present_cell](i) +=
                                ( stress_strain_tensor[i][j][k][l]*
                                  0.5*(cell_hessians[q_point][k][l][j]
                                       +
                                       cell_hessians[q_point][l][k][j])
                                  + stress_strain_tensor_grad[i][j][k][l][j] * strain_tensor[q_point][k][l]
                                ) *
                                fe_values.JxW(q_point);
                            }
                        }
                    }

                }

            }
          // -------------------------------------------------------
          // compute face_integrals
          for (unsigned int face_no=0;
               face_no<GeometryInfo<dim>::faces_per_cell;
               ++face_no)
            {
              if (cell->face(face_no)->at_boundary())
                {
                  for (unsigned int id=0; id!=dim; ++id)
                    {
                      face_integrals[cell->face(face_no)](id) = 0;
                    }
                  continue;
                }

              if ((cell->neighbor(face_no)->has_children() == false) &&
                  (cell->neighbor(face_no)->level() == cell->level()) &&
                  (cell->neighbor(face_no)->index() < cell->index()))
                continue;

              if (cell->at_boundary(face_no) == false)
                if (cell->neighbor(face_no)->level() < cell->level())
                  continue;


              if (cell->face(face_no)->has_children() == false)
                {
                  // ------------- integrate_over_regular_face -----------
                  fe_face_values_cell.reinit(cell, face_no);
                  fe_face_values_cell.get_function_grads (tmp_solution,
                                                          cell_grads);

                  Assert (cell->neighbor(face_no).state() == IteratorState::valid,
                          ExcInternalError());
                  const unsigned int
                  neighbor_neighbor = cell->neighbor_of_neighbor (face_no);
                  const typename DoFHandler<dim>::active_cell_iterator
                  neighbor = cell->neighbor(face_no);

                  fe_face_values_neighbor.reinit(neighbor, neighbor_neighbor);
                  fe_face_values_neighbor.get_function_grads (tmp_solution,
                                                              neighbor_grads);

                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                      q_cell_strain_tensor = 0.;
                      q_neighbor_strain_tensor = 0.;
                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          for (unsigned int j=0; j!=dim; ++j)
                            {
                              q_cell_strain_tensor[i][j] = 0.5*(cell_grads[q_point][i][j] +
                                                                cell_grads[q_point][j][i] );
                              q_neighbor_strain_tensor[i][j] = 0.5*(neighbor_grads[q_point][i][j] +
                                                                    neighbor_grads[q_point][j][i] );
                            }
                        }

                      constitutive_law.get_stress_strain_tensor (q_cell_strain_tensor,
                                                                 cell_stress_strain_tensor);
                      constitutive_law.get_stress_strain_tensor (q_neighbor_strain_tensor,
                                                                 neighbor_stress_strain_tensor);

                      jump_residual[q_point] = 0.;
                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          for (unsigned int j=0; j!=dim; ++j)
                            {
                              for (unsigned int k=0; k!=dim; ++k)
                                {
                                  for (unsigned int l=0; l!=dim; ++l)
                                    {
                                      jump_residual[q_point](i) += (cell_stress_strain_tensor[i][j][k][l]*
                                                                    q_cell_strain_tensor[k][l]
                                                                    -
                                                                    neighbor_stress_strain_tensor[i][j][k][l]*
                                                                    q_neighbor_strain_tensor[k][l] )*
                                                                   fe_face_values_cell.normal_vector(q_point)[j];
                                    }
                                }
                            }
                        }

                    }

                  Vector<double> face_integral_vector(dim);
                  face_integral_vector = 0;
                  for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          face_integral_vector(i) += jump_residual[q_point](i) *
                                                     fe_face_values_cell.JxW(q_point);
                        }
                    }

                  Assert (face_integrals.find (cell->face(face_no)) != face_integrals.end(),
                          ExcInternalError());

                  for (unsigned int i=0; i!=dim; ++i)
                    {
                      Assert (face_integrals[cell->face(face_no)](i) == -1e20,
                              ExcInternalError());
                      face_integrals[cell->face(face_no)](i) = face_integral_vector(i);

                    }

                  // -----------------------------------------------------
                }
              else
                {
                  // ------------- integrate_over_irregular_face ---------
                  const typename DoFHandler<dim>::face_iterator
                  face = cell->face(face_no);
                  const typename DoFHandler<dim>::cell_iterator
                  neighbor = cell->neighbor(face_no);
                  Assert (neighbor.state() == IteratorState::valid,
                          ExcInternalError());
                  Assert (neighbor->has_children(),
                          ExcInternalError());

                  const unsigned int
                  neighbor_neighbor = cell->neighbor_of_neighbor (face_no);

                  for (unsigned int subface_no=0;
                       subface_no<face->n_children(); ++subface_no)
                    {
                      const typename DoFHandler<dim>::active_cell_iterator
                      neighbor_child = cell->neighbor_child_on_subface (face_no, subface_no);
                      Assert (neighbor_child->face(neighbor_neighbor) ==
                              cell->face(face_no)->child(subface_no),
                              ExcInternalError());

                      fe_subface_values_cell.reinit (cell, face_no, subface_no);
                      fe_subface_values_cell.get_function_grads (tmp_solution,
                                                                 cell_grads);
                      fe_face_values_neighbor.reinit (neighbor_child,
                                                      neighbor_neighbor);
                      fe_face_values_neighbor.get_function_grads (tmp_solution,
                                                                  neighbor_grads);

                      for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        {
                          q_cell_strain_tensor = 0.;
                          q_neighbor_strain_tensor = 0.;
                          for (unsigned int i=0; i!=dim; ++i)
                            {
                              for (unsigned int j=0; j!=dim; ++j)
                                {
                                  q_cell_strain_tensor[i][j] = 0.5*(cell_grads[q_point][i][j] +
                                                                    cell_grads[q_point][j][i] );
                                  q_neighbor_strain_tensor[i][j] = 0.5*(neighbor_grads[q_point][i][j] +
                                                                        neighbor_grads[q_point][j][i] );
                                }
                            }

                          constitutive_law.get_stress_strain_tensor (q_cell_strain_tensor,
                                                                     cell_stress_strain_tensor);
                          constitutive_law.get_stress_strain_tensor (q_neighbor_strain_tensor,
                                                                     neighbor_stress_strain_tensor);

                          jump_residual[q_point] = 0.;
                          for (unsigned int i=0; i!=dim; ++i)
                            {
                              for (unsigned int j=0; j!=dim; ++j)
                                {
                                  for (unsigned int k=0; k!=dim; ++k)
                                    {
                                      for (unsigned int l=0; l!=dim; ++l)
                                        {
                                          jump_residual[q_point](i) += (-cell_stress_strain_tensor[i][j][k][l]*
                                                                        q_cell_strain_tensor[k][l]
                                                                        +
                                                                        neighbor_stress_strain_tensor[i][j][k][l]*
                                                                        q_neighbor_strain_tensor[k][l] )*
                                                                       fe_face_values_neighbor.normal_vector(q_point)[j];
                                        }
                                    }
                                }
                            }

                        }

                      Vector<double> face_integral_vector(dim);
                      face_integral_vector = 0;
                      for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        {
                          for (unsigned int i=0; i!=dim; ++i)
                            {
                              face_integral_vector(i) += jump_residual[q_point](i) *
                                                         fe_face_values_neighbor.JxW(q_point);
                            }
                        }

                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          face_integrals[neighbor_child->face(neighbor_neighbor)](i) = face_integral_vector(i);
                        }

                    }

                  Vector<double> sum (dim);
                  sum = 0;
                  for (unsigned int subface_no=0;
                       subface_no<face->n_children(); ++subface_no)
                    {
                      Assert (face_integrals.find(face->child(subface_no)) !=
                              face_integrals.end(),
                              ExcInternalError());
                      for (unsigned int i=0; i!=dim; ++i)
                        {
                          Assert (face_integrals[face->child(subface_no)](i) != -1e20,
                                  ExcInternalError());
                          sum(i) += face_integrals[face->child(subface_no)](i);
                        }
                    }
                  for (unsigned int i=0; i!=dim; ++i)
                    {
                      face_integrals[face](i) = sum(i);
                    }


                  // -----------------------------------------------------
                }


            }
        }
    // ----------------------------------------------------------

    present_cell=0;
    cell = dof_handler.begin_active();
    for (; cell!=endc; ++cell, ++present_cell)
      if (cell->is_locally_owned())
        {
          for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell;
               ++face_no)
            {
              Assert(face_integrals.find(cell->face(face_no)) !=
                     face_integrals.end(),
                     ExcInternalError());

              for (unsigned int id=0; id!=dim; ++id)
                {
                  error_indicators_vector[present_cell](id)
                  -= 0.5*face_integrals[cell->face(face_no)](id);
                }

            }

          estimated_error_per_cell(present_cell) = error_indicators_vector[present_cell].l2_norm();

        }

  }


  // @sect4{PlasticityContactProblem::refine_grid}

  // If you've made it this far into the deal.II tutorial, the following
  // function refining the mesh should not pose any challenges to you
  // any more. It refines the mesh, either globally or using the Kelly
  // error estimator, and if so asked also transfers the solution from
  // the previous to the next mesh. In the latter case, we also need
  // to compute the active set and other quantities again, for which we
  // need the information computed by <code>compute_nonlinear_residual()</code>.
  template <int dim>
  void
  ElastoPlasticProblem<dim>::refine_grid ()
  {
    // ---------------------------------------------------------------
    // Make a field variable for history varibales to be able to
    // transfer the data to the quadrature points of the new mesh
    FE_DGQ<dim> history_fe (1);
    DoFHandler<dim> history_dof_handler (triangulation);
    history_dof_handler.distribute_dofs (history_fe);
    std::vector< std::vector< Vector<double> > >
    history_stress_field (dim, std::vector< Vector<double> >(dim)),
                         local_history_stress_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
                         local_history_stress_fe_values (dim, std::vector< Vector<double> >(dim));


    std::vector< std::vector< Vector<double> > >
    history_strain_field (dim, std::vector< Vector<double> >(dim)),
                         local_history_strain_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
                         local_history_strain_fe_values (dim, std::vector< Vector<double> >(dim));

    for (unsigned int i=0; i<dim; i++)
      for (unsigned int j=0; j<dim; j++)
        {
          history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
          local_history_stress_values_at_qpoints[i][j].reinit(quadrature_formula.size());
          local_history_stress_fe_values[i][j].reinit(history_fe.dofs_per_cell);

          history_strain_field[i][j].reinit(history_dof_handler.n_dofs());
          local_history_strain_values_at_qpoints[i][j].reinit(quadrature_formula.size());
          local_history_strain_fe_values[i][j].reinit(history_fe.dofs_per_cell);
        }
    FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
                                             quadrature_formula.size());
    FETools::compute_projection_from_quadrature_points_matrix
    (history_fe,
     quadrature_formula, quadrature_formula,
     qpoint_to_dof_matrix);
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end(),
    dg_cell = history_dof_handler.begin_active();
    for (; cell!=endc; ++cell, ++dg_cell)
      if (cell->is_locally_owned())
        {
          PointHistory<dim> *local_quadrature_points_history
            = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          Assert (local_quadrature_points_history >=
                  &quadrature_point_history.front(),
                  ExcInternalError());
          Assert (local_quadrature_points_history <
                  &quadrature_point_history.back(),
                  ExcInternalError());
          for (unsigned int i=0; i<dim; i++)
            for (unsigned int j=0; j<dim; j++)
              {
                for (unsigned int q=0; q<quadrature_formula.size(); ++q)
                  {
                    local_history_stress_values_at_qpoints[i][j](q)
                      = local_quadrature_points_history[q].old_stress[i][j];

                    local_history_strain_values_at_qpoints[i][j](q)
                      = local_quadrature_points_history[q].old_strain[i][j];
                  }
                qpoint_to_dof_matrix.vmult (local_history_stress_fe_values[i][j],
                                            local_history_stress_values_at_qpoints[i][j]);
                dg_cell->set_dof_values (local_history_stress_fe_values[i][j],
                                         history_stress_field[i][j]);

                qpoint_to_dof_matrix.vmult (local_history_strain_fe_values[i][j],
                                            local_history_strain_values_at_qpoints[i][j]);
                dg_cell->set_dof_values (local_history_strain_fe_values[i][j],
                                         history_strain_field[i][j]);
              }
        }


    // ---------------------------------------------------------------
    // Refine the mesh
    if (refinement_strategy == RefinementStrategy::refine_global)
      {
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
          if (cell->is_locally_owned())
            cell->set_refine_flag ();
      }
    else
      {
        const double refine_fraction_cells = .3,
                     coarsen_fraction_cells = .03;
//        const double refine_fraction_cells = .1,
//                     coarsen_fraction_cells = .3;

        parallel::distributed::GridRefinement
        ::refine_and_coarsen_fixed_number(triangulation,
                                          estimated_error_per_cell,
                                          refine_fraction_cells, coarsen_fraction_cells);
      }

    triangulation.prepare_coarsening_and_refinement();

    parallel::distributed::SolutionTransfer<dim,
             TrilinosWrappers::MPI::Vector> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution);


    parallel::distributed::SolutionTransfer<dim,
             TrilinosWrappers::MPI::Vector> incremental_displacement_transfer(dof_handler);
    if (transfer_solution)
      incremental_displacement_transfer.prepare_for_coarsening_and_refinement(incremental_displacement);

    SolutionTransfer<dim, Vector<double> > history_stress_field_transfer0(history_dof_handler),
                     history_stress_field_transfer1(history_dof_handler),
                     history_stress_field_transfer2(history_dof_handler);
    history_stress_field_transfer0.prepare_for_coarsening_and_refinement(history_stress_field[0]);
    if ( dim > 1)
      {
        history_stress_field_transfer1.prepare_for_coarsening_and_refinement(history_stress_field[1]);
      }
    if ( dim == 3)
      {
        history_stress_field_transfer2.prepare_for_coarsening_and_refinement(history_stress_field[2]);
      }

    SolutionTransfer<dim, Vector<double> > history_strain_field_transfer0(history_dof_handler),
                     history_strain_field_transfer1(history_dof_handler),
                     history_strain_field_transfer2(history_dof_handler);
    history_strain_field_transfer0.prepare_for_coarsening_and_refinement(history_strain_field[0]);
    if ( dim > 1)
      {
        history_strain_field_transfer1.prepare_for_coarsening_and_refinement(history_strain_field[1]);
      }
    if ( dim == 3)
      {
        history_strain_field_transfer2.prepare_for_coarsening_and_refinement(history_strain_field[2]);
      }

    triangulation.execute_coarsening_and_refinement();
    pcout << "    Number of active cells:       "
          << triangulation.n_active_cells()
          << std::endl;

    setup_system();
    setup_quadrature_point_history ();


    TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
//    distributed_solution = solution;
    solution_transfer.interpolate(distributed_solution);
    solution = distributed_solution;

    if (transfer_solution)
      {
        TrilinosWrappers::MPI::Vector distributed_incremental_displacement(locally_owned_dofs, mpi_communicator);
//        distributed_incremental_displacement = incremental_displacement;
        incremental_displacement_transfer.interpolate(distributed_incremental_displacement);
        incremental_displacement = distributed_incremental_displacement;
//        compute_nonlinear_residual(incremental_displacement);
      }

    // ---------------------------------------------------
    history_dof_handler.distribute_dofs (history_fe);
    // stress
    std::vector< std::vector< Vector<double> > >
    distributed_history_stress_field (dim, std::vector< Vector<double> >(dim));
    for (unsigned int i=0; i<dim; i++)
      for (unsigned int j=0; j<dim; j++)
        {
          distributed_history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
        }

    history_stress_field_transfer0.interpolate(history_stress_field[0], distributed_history_stress_field[0]);
    if ( dim > 1)
      {
        history_stress_field_transfer1.interpolate(history_stress_field[1], distributed_history_stress_field[1]);
      }
    if ( dim == 3)
      {
        history_stress_field_transfer2.interpolate(history_stress_field[2], distributed_history_stress_field[2]);
      }

    history_stress_field = distributed_history_stress_field;

    // strain
    std::vector< std::vector< Vector<double> > >
    distributed_history_strain_field (dim, std::vector< Vector<double> >(dim));
    for (unsigned int i=0; i<dim; i++)
      for (unsigned int j=0; j<dim; j++)
        {
          distributed_history_strain_field[i][j].reinit(history_dof_handler.n_dofs());
        }

    history_strain_field_transfer0.interpolate(history_strain_field[0], distributed_history_strain_field[0]);
    if ( dim > 1)
      {
        history_strain_field_transfer1.interpolate(history_strain_field[1], distributed_history_strain_field[1]);
      }
    if ( dim == 3)
      {
        history_strain_field_transfer2.interpolate(history_strain_field[2], distributed_history_strain_field[2]);
      }

    history_strain_field = distributed_history_strain_field;

    // ---------------------------------------------------------------
    // Transfer the history data to the quadrature points of the new mesh
    // In a final step, we have to get the data back from the now
    // interpolated global field to the quadrature points on the
    // new mesh. The following code will do that:

    FullMatrix<double> dof_to_qpoint_matrix (quadrature_formula.size(),
                                             history_fe.dofs_per_cell);
    FETools::compute_interpolation_to_quadrature_points_matrix
    (history_fe,
     quadrature_formula,
     dof_to_qpoint_matrix);
    cell = dof_handler.begin_active();
    endc = dof_handler.end();
    dg_cell = history_dof_handler.begin_active();
    for (; cell != endc; ++cell, ++dg_cell)
      if (cell->is_locally_owned())
        {
          PointHistory<dim> *local_quadrature_points_history
            = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          Assert (local_quadrature_points_history >=
                  &quadrature_point_history.front(),
                  ExcInternalError());
          Assert (local_quadrature_points_history <
                  &quadrature_point_history.back(),
                  ExcInternalError());
          for (unsigned int i=0; i<dim; i++)
            for (unsigned int j=0; j<dim; j++)
              {
                dg_cell->get_dof_values (history_stress_field[i][j],
                                         local_history_stress_fe_values[i][j]);
                dof_to_qpoint_matrix.vmult (local_history_stress_values_at_qpoints[i][j],
                                            local_history_stress_fe_values[i][j]);

                dg_cell->get_dof_values (history_strain_field[i][j],
                                         local_history_strain_fe_values[i][j]);
                dof_to_qpoint_matrix.vmult (local_history_strain_values_at_qpoints[i][j],
                                            local_history_strain_fe_values[i][j]);
                for (unsigned int q=0; q<quadrature_formula.size(); ++q)
                  {
                    local_quadrature_points_history[q].old_stress[i][j]
                      = local_history_stress_values_at_qpoints[i][j](q);

                    local_quadrature_points_history[q].old_strain[i][j]
                      = local_history_strain_values_at_qpoints[i][j](q);
                  }
              }


        }
  }

  // @sect4{ElastoPlasticProblem::setup_quadrature_point_history}

  // At the beginning of our computations, we needed to set up initial values
  // of the history variables, such as the existing stresses in the material,
  // that we store in each quadrature point. As mentioned above, we use the
  // <code>user_pointer</code> for this that is available in each cell.
  //
  // To put this into larger perspective, we note that if we had previously
  // available stresses in our model (which we assume do not exist for the
  // purpose of this program), then we would need to interpolate the field of
  // preexisting stresses to the quadrature points. Likewise, if we were to
  // simulate elasto-plastic materials with hardening/softening, then we would
  // have to store additional history variables like the present yield stress
  // of the accumulated plastic strains in each quadrature
  // points. Pre-existing hardening or weakening would then be implemented by
  // interpolating these variables in the present function as well.
  template <int dim>
  void ElastoPlasticProblem<dim>::setup_quadrature_point_history ()
  {
    // What we need to do here is to first count how many quadrature points
    // are within the responsibility of this processor. This, of course,
    // equals the number of cells that belong to this processor times the
    // number of quadrature points our quadrature formula has on each cell.
    //
    // For good measure, we also set all user pointers of all cells, whether
    // ours of not, to the null pointer. This way, if we ever access the user
    // pointer of a cell which we should not have accessed, a segmentation
    // fault will let us know that this should not have happened:
    unsigned int our_cells = 0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned())
        ++our_cells;

    triangulation.clear_user_data();

    // Next, allocate as many quadrature objects as we need. Since the
    // <code>resize</code> function does not actually shrink the amount of
    // allocated memory if the requested new size is smaller than the old
    // size, we resort to a trick to first free all memory, and then
    // reallocate it: we declare an empty vector as a temporary variable and
    // then swap the contents of the old vector and this temporary
    // variable. This makes sure that the
    // <code>quadrature_point_history</code> is now really empty, and we can
    // let the temporary variable that now holds the previous contents of the
    // vector go out of scope and be destroyed. In the next step. we can then
    // re-allocate as many elements as we need, with the vector
    // default-initializing the <code>PointHistory</code> objects, which
    // includes setting the stress variables to zero.
    {
      std::vector<PointHistory<dim> > tmp;
      tmp.swap (quadrature_point_history);
    }
    quadrature_point_history.resize (our_cells *
                                     quadrature_formula.size());

    // Finally loop over all cells again and set the user pointers from the
    // cells that belong to the present processor to point to the first
    // quadrature point objects corresponding to this cell in the vector of
    // such objects:
    unsigned int history_index = 0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned())
        {
          cell->set_user_pointer (&quadrature_point_history[history_index]);
          history_index += quadrature_formula.size();
        }

    // At the end, for good measure make sure that our count of elements was
    // correct and that we have both used up all objects we allocated
    // previously, and not point to any objects beyond the end of the
    // vector. Such defensive programming strategies are always good checks to
    // avoid accidental errors and to guard against future changes to this
    // function that forget to update all uses of a variable at the same
    // time. Recall that constructs using the <code>Assert</code> macro are
    // optimized away in optimized mode, so do not affect the run time of
    // optimized runs:
    Assert (history_index == quadrature_point_history.size(),
            ExcInternalError());
  }

  // @sect4{ElastoPlasticProblem::update_quadrature_point_history}

  // At the end of each time step, we should have computed an incremental
  // displacement update so that the material in its new configuration
  // accommodates for the difference between the external body and boundary
  // forces applied during this time step minus the forces exerted through
  // preexisting internal stresses. In order to have the preexisting
  // stresses available at the next time step, we therefore have to update the
  // preexisting stresses with the stresses due to the incremental
  // displacement computed during the present time step. Ideally, the
  // resulting sum of internal stresses would exactly counter all external
  // forces. Indeed, a simple experiment can make sure that this is so: if we
  // choose boundary conditions and body forces to be time independent, then
  // the forcing terms (the sum of external forces and internal stresses)
  // should be exactly zero. If you make this experiment, you will realize
  // from the output of the norm of the right hand side in each time step that
  // this is almost the case: it is not exactly zero, since in the first time
  // step the incremental displacement and stress updates were computed
  // relative to the undeformed mesh, which was then deformed. In the second
  // time step, we again compute displacement and stress updates, but this
  // time in the deformed mesh -- there, the resulting updates are very small
  // but not quite zero. This can be iterated, and in each such iteration the
  // residual, i.e. the norm of the right hand side vector, is reduced; if one
  // makes this little experiment, one realizes that the norm of this residual
  // decays exponentially with the number of iterations, and after an initial
  // very rapid decline is reduced by roughly a factor of about 3.5 in each
  // iteration (for one testcase I looked at, other testcases, and other
  // numbers of unknowns change the factor, but not the exponential decay).

  // In a sense, this can then be considered as a quasi-timestepping scheme to
  // resolve the nonlinear problem of solving large-deformation elasticity on
  // a mesh that is moved along in a Lagrangian manner.
  //
  // Another complication is that the existing (old) stresses are defined on
  // the old mesh, which we will move around after updating the stresses. If
  // this mesh update involves rotations of the cell, then we need to also
  // rotate the updated stress, since it was computed relative to the
  // coordinate system of the old cell.
  //
  // Thus, what we need is the following: on each cell which the present
  // processor owns, we need to extract the old stress from the data stored
  // with each quadrature point, compute the stress update, add the two
  // together, and then rotate the result together with the incremental
  // rotation computed from the incremental displacement at the present
  // quadrature point. We will detail these steps below:
  template <int dim>
  void ElastoPlasticProblem<dim>::
  update_quadrature_point_history ()
  {
    // First, set up an <code>FEValues</code> object by which we will evaluate
    // the displacements and the gradients thereof at the
    // quadrature points, together with a vector that will hold this
    // information:
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values | update_gradients |
                             update_quadrature_points);

    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<SymmetricTensor<2, dim> > incremental_strain_tensor(n_q_points);
    SymmetricTensor<4, dim> stress_strain_tensor;


    // Then loop over all cells and do the job in the cells that belong to our
    // subdomain:

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    const FEValuesExtractors::Vector displacement(0);

    for (;  cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          // Next, get a pointer to the quadrature point history data local to
          // the present cell, and, as a defensive measure, make sure that
          // this pointer is within the bounds of the global array:
          PointHistory<dim> *local_quadrature_points_history
            = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          Assert (local_quadrature_points_history >=
                  &quadrature_point_history.front(),
                  ExcInternalError());
          Assert (local_quadrature_points_history <
                  &quadrature_point_history.back(),
                  ExcInternalError());

          // Then initialize the <code>FEValues</code> object on the present
          // cell, and extract the strains of the displacement at the
          // quadrature points
          fe_values.reinit (cell);
          fe_values[displacement].get_function_symmetric_gradients(incremental_displacement,
                                                                   incremental_strain_tensor);

          // Then loop over the quadrature points of this cell:
          for (unsigned int q=0; q<quadrature_formula.size(); ++q)
            {
              local_quadrature_points_history[q].old_strain +=
                incremental_strain_tensor[q];

              constitutive_law.get_stress_strain_tensor(local_quadrature_points_history[q].old_strain,
                                                        stress_strain_tensor);

              // The result of these operations is then written back into
              // the original place:
              local_quadrature_points_history[q].old_stress
                = stress_strain_tensor *  local_quadrature_points_history[q].old_strain;

              local_quadrature_points_history[q].point
                = fe_values.get_quadrature_points ()[q];
            }
        }
  }


  // @sect4{PlasticityContactProblem::move_mesh}

  // The remaining three functions before we get to <code>run()</code>
  // have to do with generating output. The following one is an attempt
  // at showing the deformed body in its deformed configuration. To this
  // end, this function takes a displacement vector field and moves every
  // vertex of the (local part) of the mesh by the previously computed
  // displacement. We will call this function with the current
  // displacement field before we generate graphical output, and we will
  // call it again after generating graphical output with the negative
  // displacement field to undo the changes to the mesh so made.
  //
  // The function itself is pretty straightforward. All we have to do
  // is keep track which vertices we have already touched, as we
  // encounter the same vertices multiple times as we loop over cells.
  template <int dim>
  void
  ElastoPlasticProblem<dim>::
  move_mesh (const TrilinosWrappers::MPI::Vector &displacement) const
  {
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);

    for (typename DoFHandler<dim>::active_cell_iterator cell =
           dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      if (cell->is_locally_owned())
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          if (vertex_touched[cell->vertex_index(v)] == false)
            {
              vertex_touched[cell->vertex_index(v)] = true;

              Point<dim> vertex_displacement;
              for (unsigned int d = 0; d < dim; ++d)
                vertex_displacement[d] = displacement(cell->vertex_dof_index(v, d));

              cell->vertex(v) += vertex_displacement;
            }
  }



  // @sect4{PlasticityContactProblem::output_results}

  // Next is the function we use to actually generate graphical output. The
  // function is a bit tedious, but not actually particularly complicated.
  // It moves the mesh at the top (and moves it back at the end), then
  // computes the contact forces along the contact surface. We can do
  // so (as shown in the accompanying paper) by taking the untreated
  // residual vector and identifying which degrees of freedom
  // correspond to those with contact by asking whether they have an
  // inhomogeneous constraints associated with them. As always, we need
  // to be mindful that we can only write into completely distributed
  // vectors (i.e., vectors without ghost elements) but that when we
  // want to generate output, we need vectors that do indeed have
  // ghost entries for all locally relevant degrees of freedom.
  template <int dim>
  void
  ElastoPlasticProblem<dim>::output_results (const std::string &filename_base)
  {
    TimerOutput::Scope t(computing_timer, "Graphical output");

    pcout << "      Writing graphical output... " << std::flush;

    TrilinosWrappers::MPI::Vector magnified_solution(solution);

    const double magnified_factor = 3;
    magnified_solution *= magnified_factor;

    move_mesh(magnified_solution);

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);

    //
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(solution,
                             std::vector<std::string> (dim, "displacement"),
                             DataOut<dim>::type_dof_data, data_component_interpretation);

    //
    std::vector<std::string> solution_names;

    switch (dim)
      {
      case 1:
        solution_names.push_back ("displacement");
        break;
      case 2:
        solution_names.push_back ("x_displacement");
        solution_names.push_back ("y_displacement");
        break;
      case 3:
        solution_names.push_back ("x_displacement");
        solution_names.push_back ("y_displacement");
        solution_names.push_back ("z_displacement");
        break;
      default:
        AssertThrow (false, ExcNotImplemented());
      }

    data_out.add_data_vector (solution, solution_names);


    //
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    //
    data_out.add_data_vector(fraction_of_plastic_q_points_per_cell,
                             "fraction_of_plastic_q_points");

    //
    data_out.build_patches();

    // In the remainder of the function, we generate one VTU file on
    // every processor, indexed by the subdomain id of this processor.
    // On the first processor, we then also create a <code>.pvtu</code>
    // file that indexes <i>all</i> of the VTU files so that the entire
    // set of output files can be read at once. These <code>.pvtu</code>
    // are used by Paraview to describe an entire parallel computation's
    // output files. We then do the same again for the competitor of
    // Paraview, the Visit visualization program, by creating a matching
    // <code>.visit</code> file.
    const std::string filename =
      (output_dir + filename_base + "-"
       + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));

    std::ofstream output_vtu((filename + ".vtu").c_str());
    data_out.write_vtu(output_vtu);
    pcout << output_dir + filename_base << ".pvtu" << std::endl;


    if (this_mpi_process == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < n_mpi_processes; ++i)
          filenames.push_back(filename_base + "-" +
                              Utilities::int_to_string(i, 4) +
                              ".vtu");

        std::ofstream pvtu_master_output((output_dir + filename_base + ".pvtu").c_str());
        data_out.write_pvtu_record(pvtu_master_output, filenames);

        std::ofstream visit_master_output((output_dir + filename_base + ".visit").c_str());
        data_out.write_visit_record(visit_master_output, filenames);

        // produce eps files for mesh illustration
        std::ofstream output_eps((filename + ".eps").c_str());
        GridOut grid_out;
        grid_out.write_eps(triangulation, output_eps);
      }

    // Extrapolate the stresses from Gauss point to the nodes
    SymmetricTensor<2, dim> stress_at_qpoint;

    FE_DGQ<dim> history_fe (1);
    DoFHandler<dim> history_dof_handler (triangulation);
    history_dof_handler.distribute_dofs (history_fe);
    std::vector< std::vector< Vector<double> > >
    history_stress_field (dim, std::vector< Vector<double> >(dim)),
                         local_history_stress_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
                         local_history_stress_fe_values (dim, std::vector< Vector<double> >(dim));
    for (unsigned int i=0; i<dim; i++)
      for (unsigned int j=0; j<dim; j++)
        {
          history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
          local_history_stress_values_at_qpoints[i][j].reinit(quadrature_formula.size());
          local_history_stress_fe_values[i][j].reinit(history_fe.dofs_per_cell);
        }

    Vector<double>  VM_stress_field (history_dof_handler.n_dofs()),
           local_VM_stress_values_at_qpoints (quadrature_formula.size()),
           local_VM_stress_fe_values (history_fe.dofs_per_cell);

    FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
                                             quadrature_formula.size());
    FETools::compute_projection_from_quadrature_points_matrix
    (history_fe,
     quadrature_formula, quadrature_formula,
     qpoint_to_dof_matrix);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end(),
    dg_cell = history_dof_handler.begin_active();

    const FEValuesExtractors::Vector displacement(0);

    for (; cell!=endc; ++cell, ++dg_cell)
      if (cell->is_locally_owned())
        {
          PointHistory<dim> *local_quadrature_points_history
            = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          Assert (local_quadrature_points_history >=
                  &quadrature_point_history.front(),
                  ExcInternalError());
          Assert (local_quadrature_points_history <
                  &quadrature_point_history.back(),
                  ExcInternalError());

          // Then loop over the quadrature points of this cell:
          for (unsigned int q=0; q<quadrature_formula.size(); ++q)
            {
              stress_at_qpoint = local_quadrature_points_history[q].old_stress;

              for (unsigned int i=0; i<dim; i++)
                for (unsigned int j=i; j<dim; j++)
                  {
                    local_history_stress_values_at_qpoints[i][j](q) = stress_at_qpoint[i][j];
                  }

              local_VM_stress_values_at_qpoints(q) = Evaluation::get_von_Mises_stress(stress_at_qpoint);

            }


          for (unsigned int i=0; i<dim; i++)
            for (unsigned int j=i; j<dim; j++)
              {
                qpoint_to_dof_matrix.vmult (local_history_stress_fe_values[i][j],
                                            local_history_stress_values_at_qpoints[i][j]);
                dg_cell->set_dof_values (local_history_stress_fe_values[i][j],
                                         history_stress_field[i][j]);
              }

          qpoint_to_dof_matrix.vmult (local_VM_stress_fe_values,
                                      local_VM_stress_values_at_qpoints);
          dg_cell->set_dof_values (local_VM_stress_fe_values,
                                   VM_stress_field);


        }

    // Save stresses on nodes by nodal averaging
    // construct a DoFHandler object based on FE_Q with 1 degree of freedom
    // in order to compute stresses on nodes (by applying nodal averaging)
    // Therefore, each vertex has one degree of freedom
    FE_Q<dim>          fe_1 (1);
    DoFHandler<dim>    dof_handler_1 (triangulation);
    dof_handler_1.distribute_dofs (fe_1);

    AssertThrow(dof_handler_1.n_dofs() == triangulation.n_vertices(),
                ExcDimensionMismatch(dof_handler_1.n_dofs(),triangulation.n_vertices()));

    std::vector< std::vector< Vector<double> > >
    history_stress_on_vertices (dim, std::vector< Vector<double> >(dim));
    for (unsigned int i=0; i<dim; i++)
      for (unsigned int j=0; j<dim; j++)
        {
          history_stress_on_vertices[i][j].reinit(dof_handler_1.n_dofs());
        }

    Vector<double>  VM_stress_on_vertices (dof_handler_1.n_dofs()),
           counter_on_vertices (dof_handler_1.n_dofs());
    VM_stress_on_vertices = 0;
    counter_on_vertices = 0;

    cell = dof_handler.begin_active();
    dg_cell = history_dof_handler.begin_active();
    typename DoFHandler<dim>::active_cell_iterator
    cell_1 = dof_handler_1.begin_active();
    for (; cell!=endc; ++cell, ++dg_cell, ++cell_1)
      if (cell->is_locally_owned())
        {
          dg_cell->get_dof_values (VM_stress_field,
                                   local_VM_stress_fe_values);

          for (unsigned int i=0; i<dim; i++)
            for (unsigned int j=0; j<dim; j++)
              {
                dg_cell->get_dof_values (history_stress_field[i][j],
                                         local_history_stress_fe_values[i][j]);
              }

          for  (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
            {
              types::global_dof_index dof_1_vertex = cell_1->vertex_dof_index(v, 0);

              // begin check
              //            Point<dim> point1, point2;
              //            point1 = cell_1->vertex(v);
              //            point2 = dg_cell->vertex(v);
              //            AssertThrow(point1.distance(point2) < cell->diameter()*1e-8, ExcInternalError());
              // end check

              counter_on_vertices (dof_1_vertex) += 1;

              VM_stress_on_vertices (dof_1_vertex) += local_VM_stress_fe_values (v);

              for (unsigned int i=0; i<dim; i++)
                for (unsigned int j=0; j<dim; j++)
                  {
                    history_stress_on_vertices[i][j](dof_1_vertex) +=
                      local_history_stress_fe_values[i][j](v);
                  }

            }
        }

    for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
      {
        VM_stress_on_vertices(id) /= counter_on_vertices(id);

        for (unsigned int i=0; i<dim; i++)
          for (unsigned int j=0; j<dim; j++)
            {
              history_stress_on_vertices[i][j](id) /= counter_on_vertices(id);
            }
      }

    // Save figures of stresses
    if (show_stresses)
      {
        {
          DataOut<dim>  data_out;
          data_out.attach_dof_handler (history_dof_handler);


          data_out.add_data_vector (history_stress_field[0][0], "stress_xx");
          data_out.add_data_vector (history_stress_field[1][1], "stress_yy");
          data_out.add_data_vector (history_stress_field[0][1], "stress_xy");
          data_out.add_data_vector (VM_stress_field, "Von_Mises_stress");

          if (dim == 3)
            {
              data_out.add_data_vector (history_stress_field[0][2], "stress_xz");
              data_out.add_data_vector (history_stress_field[1][2], "stress_yz");
              data_out.add_data_vector (history_stress_field[2][2], "stress_zz");
            }

          data_out.build_patches ();

          const std::string filename_base_stress = ("stress-" + filename_base);

          const std::string filename =
            (output_dir + filename_base_stress + "-"
             + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));

          std::ofstream output_vtu((filename + ".vtu").c_str());
          data_out.write_vtu(output_vtu);
          pcout << output_dir + filename_base_stress << ".pvtu" << std::endl;

          if (this_mpi_process == 0)
            {
              std::vector<std::string> filenames;
              for (unsigned int i = 0; i < n_mpi_processes; ++i)
                filenames.push_back(filename_base_stress + "-" +
                                    Utilities::int_to_string(i, 4) +
                                    ".vtu");

              std::ofstream pvtu_master_output((output_dir + filename_base_stress + ".pvtu").c_str());
              data_out.write_pvtu_record(pvtu_master_output, filenames);

              std::ofstream visit_master_output((output_dir + filename_base_stress + ".visit").c_str());
              data_out.write_visit_record(visit_master_output, filenames);
            }


        }

        {
          DataOut<dim>  data_out;
          data_out.attach_dof_handler (dof_handler_1);


          data_out.add_data_vector (history_stress_on_vertices[0][0], "stress_xx_averaged");
          data_out.add_data_vector (history_stress_on_vertices[1][1], "stress_yy_averaged");
          data_out.add_data_vector (history_stress_on_vertices[0][1], "stress_xy_averaged");
          data_out.add_data_vector (VM_stress_on_vertices, "Von_Mises_stress_averaged");

          if (dim == 3)
            {
              data_out.add_data_vector (history_stress_on_vertices[0][2], "stress_xz_averaged");
              data_out.add_data_vector (history_stress_on_vertices[1][2], "stress_yz_averaged");
              data_out.add_data_vector (history_stress_on_vertices[2][2], "stress_zz_averaged");
            }

          data_out.build_patches ();

          const std::string filename_base_stress = ("averaged-stress-" + filename_base);

          const std::string filename =
            (output_dir + filename_base_stress + "-"
             + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));

          std::ofstream output_vtu((filename + ".vtu").c_str());
          data_out.write_vtu(output_vtu);
          pcout << output_dir + filename_base_stress << ".pvtu" << std::endl;

          if (this_mpi_process == 0)
            {
              std::vector<std::string> filenames;
              for (unsigned int i = 0; i < n_mpi_processes; ++i)
                filenames.push_back(filename_base_stress + "-" +
                                    Utilities::int_to_string(i, 4) +
                                    ".vtu");

              std::ofstream pvtu_master_output((output_dir + filename_base_stress + ".pvtu").c_str());
              data_out.write_pvtu_record(pvtu_master_output, filenames);

              std::ofstream visit_master_output((output_dir + filename_base_stress + ".visit").c_str());
              data_out.write_visit_record(visit_master_output, filenames);
            }


        }
        // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      }

    magnified_solution *= -1;
    move_mesh(magnified_solution);

    // Timoshenko beam
    if (base_mesh == "Timoshenko beam")
      {
        const double length = .48,
                     depth  = .12;

        Point<dim> intersted_point(length, -depth/2);
        Point<dim> vertex_displacement;
        bool vertex_found = false;

        for (typename DoFHandler<dim>::active_cell_iterator cell =
               dof_handler.begin_active();
             cell != dof_handler.end(); ++cell)
          if (cell->is_locally_owned() && !vertex_found)
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
              if ( std::fabs(cell->vertex(v)[0] - intersted_point[0])<1e-6 &&
                   std::fabs(cell->vertex(v)[1] - intersted_point[1])<1e-6)
                {
                  vertex_found = true;

                  for (unsigned int d = 0; d < dim; ++d)
                    vertex_displacement[d] = solution(cell->vertex_dof_index(v, d));

                  break;
                }

        pcout << "   Number of active cells: "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        AssertThrow(vertex_found, ExcInternalError());
        std::cout << "Displacement at the point (" << intersted_point[0]
                  << ", " << intersted_point[1] << ") is "
                  << "(" << vertex_displacement[0]
                  << ", " << vertex_displacement[1] << ").\n";

        Vector<double> vertex_exact_displacement(dim);
        EquationData::IncrementalBoundaryValues<dim> incremental_boundary_values(present_time, end_time);
        incremental_boundary_values.vector_value (intersted_point, vertex_exact_displacement);

        std::cout << "Exact displacement at the point (" << intersted_point[0]
                  << ", " << intersted_point[1] << ") is "
                  << "(" << vertex_exact_displacement[0]
                  << ", " << vertex_exact_displacement[1] << ").\n\n";

      }
    else if (base_mesh == "Thick_tube_internal_pressure")
      {
        const double pressure (0.6*2.4e8),
              inner_radius (.1);
//      const double pressure (1.94e8),
//                   inner_radius (.1);


        // Plane stress
//      const double mu (((e_modulus*(1+2*nu)) / (std::pow((1+nu),2))) / (2 * (1 + (nu / (1+nu)))));
        // 3d and plane strain
        const double mu (e_modulus / (2 * (1 + nu)));

        const Point<dim> point_A(inner_radius, 0.);
        Vector<double>   disp_A(dim);

        // make a non-parallel copy of solution
        Vector<double> copy_solution(solution);

        typename Evaluation::PointValuesEvaluation<dim>::
        PointValuesEvaluation point_values_evaluation(point_A);

        point_values_evaluation.compute (dof_handler, copy_solution, disp_A);

        table_results.add_value("time step", timestep_no);
        table_results.add_value("Cells", triangulation.n_global_active_cells());
        table_results.add_value("DoFs", dof_handler.n_dofs());
        table_results.add_value("pressure/sigma_0", (pressure*present_time/end_time)/sigma_0);
        table_results.add_value("4*mu*u_A/(sigma_0*a)", 4*mu*disp_A(0)/(sigma_0*inner_radius));

        // Compute stresses in the POLAR coordinates, 1- save it on Gauss points,
        // 2- extrapolate them to nodes and taking their avarages (nodal avaraging)
        AssertThrow (dim == 2, ExcNotImplemented());

        // we define a rotation matrix to be able to transform the stress
        // from the Cartesian coordinate to the polar coordinate
        Tensor<2, dim> rotation_matrix; // [cos sin; -sin cos]    , sigma_r = rot * sigma * rot^T

        FEValues<dim> fe_values (fe, quadrature_formula, update_quadrature_points |
                                 update_values | update_gradients);

        const unsigned int n_q_points = quadrature_formula.size();

        std::vector<SymmetricTensor<2, dim> > strain_tensor(n_q_points);
        SymmetricTensor<4, dim> stress_strain_tensor;
        Tensor<2, dim>  stress_at_qpoint;

        FE_DGQ<dim> history_fe (1);
        DoFHandler<dim> history_dof_handler (triangulation);
        history_dof_handler.distribute_dofs (history_fe);
        std::vector< std::vector< Vector<double> > >
        history_stress_field (dim, std::vector< Vector<double> >(dim)),
                             local_history_stress_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
                             local_history_stress_fe_values (dim, std::vector< Vector<double> >(dim));
        for (unsigned int i=0; i<dim; i++)
          for (unsigned int j=0; j<dim; j++)
            {
              history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
              local_history_stress_values_at_qpoints[i][j].reinit(quadrature_formula.size());
              local_history_stress_fe_values[i][j].reinit(history_fe.dofs_per_cell);
            }

        FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
                                                 quadrature_formula.size());
        FETools::compute_projection_from_quadrature_points_matrix
        (history_fe,
         quadrature_formula, quadrature_formula,
         qpoint_to_dof_matrix);

        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end(),
        dg_cell = history_dof_handler.begin_active();

        const FEValuesExtractors::Vector displacement(0);

        for (; cell!=endc; ++cell, ++dg_cell)
          if (cell->is_locally_owned())
            {
              PointHistory<dim> *local_quadrature_points_history
                = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
              Assert (local_quadrature_points_history >=
                      &quadrature_point_history.front(),
                      ExcInternalError());
              Assert (local_quadrature_points_history <
                      &quadrature_point_history.back(),
                      ExcInternalError());

              // Then loop over the quadrature points of this cell:
              for (unsigned int q=0; q<quadrature_formula.size(); ++q)
                {
                  stress_at_qpoint = local_quadrature_points_history[q].old_stress;

                  // transform the stress from the Cartesian coordinate to the polar coordinate
                  const Point<dim> point = local_quadrature_points_history[q].point;
                  const double radius = point.norm ();
                  const double theta = std::atan2(point(1),point(0));

                  // rotation matrix
                  rotation_matrix[0][0] = std::cos(theta);
                  rotation_matrix[0][1] = std::sin(theta);
                  rotation_matrix[1][0] = -std::sin(theta);
                  rotation_matrix[1][1] = std::cos(theta);

                  // stress in polar coordinate
                  stress_at_qpoint = rotation_matrix * stress_at_qpoint * transpose(rotation_matrix);

                  for (unsigned int i=0; i<dim; i++)
                    for (unsigned int j=i; j<dim; j++)
                      {
                        local_history_stress_values_at_qpoints[i][j](q) = stress_at_qpoint[i][j];
                      }

                }


              for (unsigned int i=0; i<dim; i++)
                for (unsigned int j=i; j<dim; j++)
                  {
                    qpoint_to_dof_matrix.vmult (local_history_stress_fe_values[i][j],
                                                local_history_stress_values_at_qpoints[i][j]);
                    dg_cell->set_dof_values (local_history_stress_fe_values[i][j],
                                             history_stress_field[i][j]);
                  }

            }

        {
          DataOut<dim>  data_out;
          data_out.attach_dof_handler (history_dof_handler);


          data_out.add_data_vector (history_stress_field[0][0], "stress_rr");
          data_out.add_data_vector (history_stress_field[1][1], "stress_tt");
          data_out.add_data_vector (history_stress_field[0][1], "stress_rt");

          data_out.build_patches ();

          const std::string filename_base_stress = ("stress-polar-" + filename_base);

          const std::string filename =
            (output_dir + filename_base_stress + "-"
             + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));

          std::ofstream output_vtu((filename + ".vtu").c_str());
          data_out.write_vtu(output_vtu);
          pcout << output_dir + filename_base_stress << ".pvtu" << std::endl;

          if (this_mpi_process == 0)
            {
              std::vector<std::string> filenames;
              for (unsigned int i = 0; i < n_mpi_processes; ++i)
                filenames.push_back(filename_base_stress + "-" +
                                    Utilities::int_to_string(i, 4) +
                                    ".vtu");

              std::ofstream pvtu_master_output((output_dir + filename_base_stress + ".pvtu").c_str());
              data_out.write_pvtu_record(pvtu_master_output, filenames);

              std::ofstream visit_master_output((output_dir + filename_base_stress + ".visit").c_str());
              data_out.write_visit_record(visit_master_output, filenames);
            }


        }

        // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // construct a DoFHandler object based on FE_Q with 1 degree of freedom
        // in order to compute stresses on nodes (by applying nodal averaging)
        // Therefore, each vertex has one degree of freedom
        FE_Q<dim>          fe_1 (1);
        DoFHandler<dim>    dof_handler_1 (triangulation);
        dof_handler_1.distribute_dofs (fe_1);

        AssertThrow(dof_handler_1.n_dofs() == triangulation.n_vertices(),
                    ExcDimensionMismatch(dof_handler_1.n_dofs(),triangulation.n_vertices()));

        std::vector< std::vector< Vector<double> > >
        history_stress_on_vertices (dim, std::vector< Vector<double> >(dim));
        for (unsigned int i=0; i<dim; i++)
          for (unsigned int j=0; j<dim; j++)
            {
              history_stress_on_vertices[i][j].reinit(dof_handler_1.n_dofs());
            }

        Vector<double>  counter_on_vertices (dof_handler_1.n_dofs());
        counter_on_vertices = 0;

        cell = dof_handler.begin_active();
        dg_cell = history_dof_handler.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        cell_1 = dof_handler_1.begin_active();
        for (; cell!=endc; ++cell, ++dg_cell, ++cell_1)
          if (cell->is_locally_owned())
            {

              for (unsigned int i=0; i<dim; i++)
                for (unsigned int j=0; j<dim; j++)
                  {
                    dg_cell->get_dof_values (history_stress_field[i][j],
                                             local_history_stress_fe_values[i][j]);
                  }

              for  (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
                {
                  types::global_dof_index dof_1_vertex = cell_1->vertex_dof_index(v, 0);

                  // begin check
//            Point<dim> point1, point2;
//            point1 = cell_1->vertex(v);
//            point2 = dg_cell->vertex(v);
//            AssertThrow(point1.distance(point2) < cell->diameter()*1e-8, ExcInternalError());
                  // end check

                  counter_on_vertices (dof_1_vertex) += 1;

                  for (unsigned int i=0; i<dim; i++)
                    for (unsigned int j=0; j<dim; j++)
                      {
                        history_stress_on_vertices[i][j](dof_1_vertex) +=
                          local_history_stress_fe_values[i][j](v);
                      }

                }
            }

        for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
          {
            for (unsigned int i=0; i<dim; i++)
              for (unsigned int j=0; j<dim; j++)
                {
                  history_stress_on_vertices[i][j](id) /= counter_on_vertices(id);
                }
          }


        {
          DataOut<dim>  data_out;
          data_out.attach_dof_handler (dof_handler_1);


          data_out.add_data_vector (history_stress_on_vertices[0][0], "stress_rr_averaged");
          data_out.add_data_vector (history_stress_on_vertices[1][1], "stress_tt_averaged");
          data_out.add_data_vector (history_stress_on_vertices[0][1], "stress_rt_averaged");

          data_out.build_patches ();

          const std::string filename_base_stress = ("averaged-stress-polar-" + filename_base);

          const std::string filename =
            (output_dir + filename_base_stress + "-"
             + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));

          std::ofstream output_vtu((filename + ".vtu").c_str());
          data_out.write_vtu(output_vtu);
          pcout << output_dir + filename_base_stress << ".pvtu" << std::endl;

          if (this_mpi_process == 0)
            {
              std::vector<std::string> filenames;
              for (unsigned int i = 0; i < n_mpi_processes; ++i)
                filenames.push_back(filename_base_stress + "-" +
                                    Utilities::int_to_string(i, 4) +
                                    ".vtu");

              std::ofstream pvtu_master_output((output_dir + filename_base_stress + ".pvtu").c_str());
              data_out.write_pvtu_record(pvtu_master_output, filenames);

              std::ofstream visit_master_output((output_dir + filename_base_stress + ".visit").c_str());
              data_out.write_visit_record(visit_master_output, filenames);
            }


        }
        // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if ( std::abs( (present_time/end_time)*(pressure/sigma_0) - 0.6 ) <
             .501*(present_timestep/end_time)*(pressure/sigma_0) )
          {

            // table_results_2: presenting the stress_rr and stress_tt on the nodes of bottom edge
            const unsigned int face_id = 3;

            std::vector<bool> vertices_found (dof_handler_1.n_dofs(), false);

            bool evaluation_face_found = false;

            typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end(),
            cell_1 = dof_handler_1.begin_active();
            for (; cell!=endc; ++cell, ++cell_1)
              if (cell->is_locally_owned())
                {
                  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                    {
                      if (cell->face(face)->at_boundary()
                          &&
                          cell->face(face)->boundary_indicator() == face_id)
                        {
                          if (!evaluation_face_found)
                            {
                              evaluation_face_found = true;
                            }


                          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
                            {
                              types::global_dof_index dof_1_vertex =
                                cell_1->face(face)->vertex_dof_index(v, 0);
                              if (!vertices_found[dof_1_vertex])
                                {

                                  const Point<dim> vertex_coordinate = cell_1->face(face)->vertex(v);

                                  table_results_2.add_value("x coordinate", vertex_coordinate[0]);
                                  table_results_2.add_value("stress_rr", history_stress_on_vertices[0][0](dof_1_vertex));
                                  table_results_2.add_value("stress_tt", history_stress_on_vertices[1][1](dof_1_vertex));
                                  table_results_2.add_value("pressure/sigma_0", (pressure*present_time/end_time)/sigma_0);

                                  vertices_found[dof_1_vertex] = true;
                                }
                            }

                        }
                    }

                }

            AssertThrow(evaluation_face_found, ExcInternalError());

            // table_results_3: presenting the mean stress_rr of the nodes on the inner radius
            const unsigned int face_id_2 = 0;

            Tensor<2, dim> stress_node,
                   mean_stress_polar;
            mean_stress_polar = 0;

            std::vector<bool> vertices_found_2 (dof_handler_1.n_dofs(), false);
            unsigned int no_vertices_found = 0;

            evaluation_face_found = false;

            cell = dof_handler.begin_active(),
            endc = dof_handler.end(),
            cell_1 = dof_handler_1.begin_active();
            for (; cell!=endc; ++cell, ++cell_1)
              if (cell->is_locally_owned())
                {
                  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                    {
                      if (cell->face(face)->at_boundary()
                          &&
                          cell->face(face)->boundary_indicator() == face_id_2)
                        {
                          if (!evaluation_face_found)
                            {
                              evaluation_face_found = true;
                            }


                          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
                            {
                              types::global_dof_index dof_1_vertex =
                                cell_1->face(face)->vertex_dof_index(v, 0);
                              if (!vertices_found_2[dof_1_vertex])
                                {
                                  for (unsigned int ir=0; ir<dim; ++ir)
                                    for (unsigned int ic=0; ic<dim; ++ic)
                                      stress_node[ir][ic] = history_stress_on_vertices[ir][ic](dof_1_vertex);

                                  mean_stress_polar += stress_node;

                                  vertices_found_2[dof_1_vertex] = true;
                                  ++no_vertices_found;
                                }
                            }

                        }
                    }

                }

            AssertThrow(evaluation_face_found, ExcInternalError());

            mean_stress_polar /= no_vertices_found;

            table_results_3.add_value("time step", timestep_no);
            table_results_3.add_value("pressure/sigma_0", (pressure*present_time/end_time)/sigma_0);
            table_results_3.add_value("Cells", triangulation.n_global_active_cells());
            table_results_3.add_value("DoFs", dof_handler.n_dofs());
            table_results_3.add_value("radius", inner_radius);
            table_results_3.add_value("mean stress_rr", mean_stress_polar[0][0]);
            table_results_3.add_value("mean stress_tt", mean_stress_polar[1][1]);


          }


      }
    else if (base_mesh == "Perforated_strip_tension")
      {
        const double imposed_displacement (0.00055),
              inner_radius (0.05),
              height (0.18);

        // Plane stress
//      const double mu (((e_modulus*(1+2*nu)) / (std::pow((1+nu),2))) / (2 * (1 + (nu / (1+nu)))));
        // 3d and plane strain
        const double mu (e_modulus / (2 * (1 + nu)));

        // table_results: Demonstrates the result of displacement at the top left corner versus imposed tension
        /*
        {
          const Point<dim> point_C(0., height);
          Vector<double>   disp_C(dim);

          // make a non-parallel copy of solution
          Vector<double> copy_solution(solution);

          typename Evaluation::PointValuesEvaluation<dim>::
          PointValuesEvaluation point_values_evaluation(point_C);

          point_values_evaluation.compute (dof_handler, copy_solution, disp_C);

          table_results.add_value("time step", timestep_no);
          table_results.add_value("Cells", triangulation.n_global_active_cells());
          table_results.add_value("DoFs", dof_handler.n_dofs());
          table_results.add_value("4*mu*u_C/(sigma_0*r)", 4*mu*disp_C(1)/(sigma_0*inner_radius));
        }
        */

        // compute average sigma_yy on the bottom edge
        double stress_yy_av;
        {
          stress_yy_av = 0;
          const unsigned int face_id = 1;

          std::vector<bool> vertices_found (dof_handler_1.n_dofs(), false);
          unsigned int no_vertices_in_face = 0;

          bool evaluation_face_found = false;

          typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end(),
          cell_1 = dof_handler_1.begin_active();
          for (; cell!=endc; ++cell, ++cell_1)
            if (cell->is_locally_owned())
              {
                for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                  {
                    if (cell->face(face)->at_boundary()
                        &&
                        cell->face(face)->boundary_indicator() == face_id)
                      {
                        if (!evaluation_face_found)
                          {
                            evaluation_face_found = true;
                          }


                        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
                          {
                            types::global_dof_index dof_1_vertex =
                              cell_1->face(face)->vertex_dof_index(v, 0);
                            if (!vertices_found[dof_1_vertex])
                              {
                                stress_yy_av += history_stress_on_vertices[1][1](dof_1_vertex);
                                ++no_vertices_in_face;

                                vertices_found[dof_1_vertex] = true;
                              }
                          }

                      }
                  }

              }

          AssertThrow(evaluation_face_found, ExcInternalError());

          stress_yy_av /= no_vertices_in_face;

        }

        // table_results_2: Demonstrate the stress_yy on the nodes of bottom edge

//      if ( std::abs( (stress_yy_av/sigma_0) - .91 ) < .2 )
        if ( (timestep_no) % 19 == 0 )
//      if ( true )
          {
            const unsigned int face_id = 1;

            std::vector<bool> vertices_found (dof_handler_1.n_dofs(), false);

            bool evaluation_face_found = false;

            typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end(),
            cell_1 = dof_handler_1.begin_active();
            for (; cell!=endc; ++cell, ++cell_1)
              if (cell->is_locally_owned())
                {
                  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                    {
                      if (cell->face(face)->at_boundary()
                          &&
                          cell->face(face)->boundary_indicator() == face_id)
                        {
                          if (!evaluation_face_found)
                            {
                              evaluation_face_found = true;
                            }


                          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
                            {
                              types::global_dof_index dof_1_vertex =
                                cell_1->face(face)->vertex_dof_index(v, 0);

                              const Point<dim> vertex_coordinate = cell_1->face(face)->vertex(v);

                              if (!vertices_found[dof_1_vertex] && std::abs(vertex_coordinate[2])<1.e-8)
                                {
                                  table_results_2.add_value("x", vertex_coordinate[0]);
                                  table_results_2.add_value("x/r", vertex_coordinate[0]/inner_radius);
                                  table_results_2.add_value("stress_xx/sigma_0", history_stress_on_vertices[0][0](dof_1_vertex)/sigma_0);
                                  table_results_2.add_value("stress_yy/sigma_0", history_stress_on_vertices[1][1](dof_1_vertex)/sigma_0);
                                  table_results_2.add_value("stress_yy_av/sigma_0", stress_yy_av/sigma_0);
                                  table_results_2.add_value("Imposed u_y", (imposed_displacement*present_time/end_time));

                                  vertices_found[dof_1_vertex] = true;
                                }
                            }

                        }
                    }

                }

            AssertThrow(evaluation_face_found, ExcInternalError());

          }

        // table_results_3: Demonstrate the Stress_mean (average tensile stress)
        //  on the bottom edge versus epsilon_yy on the bottom left corner
        {
          double strain_yy_A;

          // compute strain_yy_A
          // Since the point A is the node on the bottom left corner,
          // we need to work just with one element
          {
            const Point<dim> point_A(inner_radius, 0, 0);

            Vector<double>  local_strain_yy_values_at_qpoints (quadrature_formula.size()),
                   local_strain_yy_fe_values (history_fe.dofs_per_cell);

            SymmetricTensor<2, dim> strain_at_qpoint;

            typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end(),
            dg_cell = history_dof_handler.begin_active();

            bool cell_found = false;

            for (; cell!=endc; ++cell, ++dg_cell)
              if (cell->is_locally_owned() && !cell_found)
                {
                  for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
                    if ( std::fabs(cell->vertex(v)[0] - point_A[0])<1e-6 &&
                         std::fabs(cell->vertex(v)[1] - point_A[1])<1e-6 &&
                         std::fabs(cell->vertex(v)[2] - point_A[2])<1e-6)
                      {
                        PointHistory<dim> *local_quadrature_points_history
                          = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
                        Assert (local_quadrature_points_history >=
                                &quadrature_point_history.front(),
                                ExcInternalError());
                        Assert (local_quadrature_points_history <
                                &quadrature_point_history.back(),
                                ExcInternalError());

                        // Then loop over the quadrature points of this cell:
                        for (unsigned int q=0; q<quadrature_formula.size(); ++q)
                          {
                            strain_at_qpoint = local_quadrature_points_history[q].old_strain;

                            local_strain_yy_values_at_qpoints(q) = strain_at_qpoint[1][1];
                          }

                        qpoint_to_dof_matrix.vmult (local_strain_yy_fe_values,
                                                    local_strain_yy_values_at_qpoints);

                        strain_yy_A = local_strain_yy_fe_values (v);

                        cell_found = true;
                        break;
                      }

                }

          }

          table_results_3.add_value("time step", timestep_no);
          table_results_3.add_value("Cells", triangulation.n_global_active_cells());
          table_results_3.add_value("DoFs", dof_handler.n_dofs());
          table_results_3.add_value("Imposed u_y", (imposed_displacement*present_time/end_time));
          table_results_3.add_value("mean_tensile_stress/sigma_0", stress_yy_av/sigma_0);
          table_results_3.add_value("E*strain_yy-A/sigma_0", e_modulus*strain_yy_A/sigma_0);

        }


        if (std::abs(present_time-end_time) < 1.e-7)
          {
            table_results_2.set_precision("Imposed u_y", 6);
            table_results_3.set_precision("Imposed u_y", 6);
          }

      }
    else if (base_mesh == "Cantiliver_beam_3d")
      {
        const double pressure (6e6),
              length (.7),
              height (200e-3);

        // table_results: Demonstrates the result of displacement at the top front point, Point A
        {
          // Quantity of interest:
          // displacement at Point A (x=0, y=height/2, z=length)

          const Point<dim> point_A(0, height/2, length);
          Vector<double>   disp_A(dim);

          // make a non-parallel copy of solution
          Vector<double> copy_solution(solution);

          typename Evaluation::PointValuesEvaluation<dim>::
          PointValuesEvaluation point_values_evaluation(point_A);

          point_values_evaluation.compute (dof_handler, copy_solution, disp_A);

          table_results.add_value("time step", timestep_no);
          table_results.add_value("Cells", triangulation.n_global_active_cells());
          table_results.add_value("DoFs", dof_handler.n_dofs());
          table_results.add_value("pressure", pressure*present_time/end_time);
          table_results.add_value("u_A", disp_A(1));
        }

        {
          // demonstrate the location and maximum von-Mises stress in the
          // specified domain close to the clamped face, z = 0
          // top domain: height/2 - thickness_flange <= y <= height/2
          //             0 <= z <= 2 * thickness_flange
          // bottom domain: -height/2 <= y <= -height/2 + thickness_flange
          //             0 <= z <= 2 * thickness_flange

          double VM_stress_max (0);
          Point<dim> point_max;

          SymmetricTensor<2, dim> stress_at_qpoint;

          typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();

          const FEValuesExtractors::Vector displacement(0);

          for (; cell!=endc; ++cell)
            if (cell->is_locally_owned())
              {
                PointHistory<dim> *local_quadrature_points_history
                  = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
                Assert (local_quadrature_points_history >=
                        &quadrature_point_history.front(),
                        ExcInternalError());
                Assert (local_quadrature_points_history <
                        &quadrature_point_history.back(),
                        ExcInternalError());

                // Then loop over the quadrature points of this cell:
                for (unsigned int q=0; q<quadrature_formula.size(); ++q)
                  {
                    stress_at_qpoint = local_quadrature_points_history[q].old_stress;

                    const double VM_stress = Evaluation::get_von_Mises_stress(stress_at_qpoint);
                    if (VM_stress > VM_stress_max)
                      {
                        VM_stress_max = VM_stress;
                        point_max = local_quadrature_points_history[q].point;
                      }

                  }
              }

          table_results.add_value("maximum von_Mises stress", VM_stress_max);
          table_results.add_value("x", point_max[0]);
          table_results.add_value("y", point_max[1]);
          table_results.add_value("z", point_max[2]);

        }

      }


  }


  // @sect4{PlasticityContactProblem::run}

  // As in all other tutorial programs, the <code>run()</code> function contains
  // the overall logic. There is not very much to it here: in essence, it
  // performs the loops over all mesh refinement cycles, and within each, hands
  // things over to the Newton solver in <code>solve_newton()</code> on the
  // current mesh and calls the function that creates graphical output for
  // the so-computed solution. It then outputs some statistics concerning both
  // run times and memory consumption that has been collected over the course of
  // computations on this mesh.
  template <int dim>
  void
  ElastoPlasticProblem<dim>::run ()
  {
    computing_timer.reset();

    present_time = 0;
    present_timestep = 1;
    end_time = 10;
    timestep_no = 0;

    make_grid();

    // ----------------------------------------------------------------
    //    base_mesh == "Thick_tube_internal_pressure"
    /*
    const Point<dim> center(0, 0);
    const double inner_radius = .1,
        outer_radius = .2;

    const HyperBallBoundary<dim> inner_boundary_description(center, inner_radius);
    triangulation.set_boundary (0, inner_boundary_description);

    const HyperBallBoundary<dim> outer_boundary_description(center, outer_radius);
    triangulation.set_boundary (1, outer_boundary_description);
    */
    // ----------------------------------------------------------------
    //    base_mesh == "Perforated_strip_tension"
    /*
    const double inner_radius = 0.05;

    const CylinderBoundary<dim> inner_boundary_description(inner_radius, 2);
    triangulation.set_boundary (10, inner_boundary_description);
    */
    // ----------------------------------------------------------------

    setup_quadrature_point_history ();

    while (present_time < end_time)
      {
        present_time += present_timestep;
        ++timestep_no;

        if (present_time > end_time)
          {
            present_timestep -= (present_time - end_time);
            present_time = end_time;
          }
        pcout << std::endl;
        pcout << "Time step " << timestep_no << " at time " << present_time
              << std::endl;

        relative_error = max_relative_error * 10;
        current_refinement_cycle = 0;

        setup_system();


        // ------------------------ Refinement based on the relative error -------------------------------

        while (relative_error >= max_relative_error)
          {
            solve_newton();
            compute_error();

            if ( (timestep_no > 1) && (current_refinement_cycle>0) && (relative_error >= max_relative_error) )
              {
                pcout << "The relative error, " << relative_error
                      << " , is still more than maximum relative error, "
                      << max_relative_error << ", but we move to the next increment.\n";
                relative_error = .1 * max_relative_error;
              }

            if (relative_error >= max_relative_error)
              {
                TimerOutput::Scope t(computing_timer, "Setup: refine mesh");
                ++current_refinement_cycle;
                refine_grid();
              }

          }

        // ------------------------ Refinement based on the number of refinement --------------------------
        /*
        bool continue_loop = true;
        while (continue_loop)
        {
          solve_newton();
          compute_error();

          if ( (timestep_no == 1) && (current_refinement_cycle < 1) )
          {
            TimerOutput::Scope t(computing_timer, "Setup: refine mesh");
            ++current_refinement_cycle;
            refine_grid();
          }else
          {
            continue_loop = false;
          }

        }
        */

        // -------------------------------------------------------------------------------------------------

        solution += incremental_displacement;

        update_quadrature_point_history ();

        output_results((std::string("solution-") +
                        Utilities::int_to_string(timestep_no, 4)).c_str());

        computing_timer.print_summary();
        computing_timer.reset();

        Utilities::System::MemoryStats stats;
        Utilities::System::get_memory_stats(stats);
        pcout << "Peak virtual memory used, resident in kB: " << stats.VmSize << " "
              << stats.VmRSS << std::endl;


        if (std::abs(present_time-end_time) < 1.e-7)
          {
            const std::string filename = (output_dir + "Results");

            std::ofstream output_txt((filename + ".txt").c_str());

            pcout << std::endl;
            table_results.write_text(output_txt);
            pcout << std::endl;
            table_results_2.write_text(output_txt);
            pcout << std::endl;
            table_results_3.write_text(output_txt);
            pcout << std::endl;
          }

      }

    if (base_mesh == "Thick_tube_internal_pressure")
      {
        triangulation.set_boundary (0);
        triangulation.set_boundary (1);
      }
    else if (base_mesh == "Perforated_strip_tension")
      {
        triangulation.set_boundary (10);
      }

  }
}

// @sect3{The <code>main</code> function}

// There really isn't much to the <code>main()</code> function. It looks
// like they always do:
int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace ElastoPlastic;

  try
    {
      deallog.depth_console(0);
      ParameterHandler prm;
      const int dim = 3;
      ElastoPlasticProblem<dim>::declare_parameters(prm);
      if (argc != 2)
        {
          std::cerr << "*** Call this program as <./elastoplastic input.prm>" << std::endl;
          return 1;
        }

      prm.read_input(argv[1]);
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
      {
        ElastoPlasticProblem<dim> problem(prm);
        problem.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
