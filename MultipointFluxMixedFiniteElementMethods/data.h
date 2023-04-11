/* ---------------------------------------------------------------------
 *
 * This file is part of the deal.II Code Gallery.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Ilona Ambartsumyan, Eldar Khattatov, University of Pittsburgh, 2018
 */

#ifndef MFMFE_DATA_H
#define MFMFE_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

// @sect3{Data and exact solution.}

// This file declares the classes for the given data, i.e.
// right-hand side, exact solution, permeability tensor and
// boundary conditions. For simplicity only 2d cases are
// provided, but 3d can be added straightforwardly.

namespace MFMFE
{
  using namespace dealii;

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int component = 0) const override;
  };

  template <int dim>
  double RightHandSide<dim>::value (const Point<dim>  &p,
                                    const unsigned int /*component*/) const
  {
    const double x = p[0];
    const double y = p[1];

    switch (dim)
      {
      case 2:
        return -(x*(y*y*y*y)*6.0-(y*y)*sin(x*y*2.0)*2.0+2.0)*(x*2.0+x*x+y*y+1.0)-sin(x*y)*(cos(x*y*2.0)+(x*x)*(y*y*y)*1.2E1
               -x*y*sin(x*y*2.0)*2.0)*2.0-(x*2.0+2.0)*(x*2.0+(x*x)*(y*y*y*y)*3.0+y*cos(x*y*2.0))+(x*x)*(sin(x*y*2.0)
                   -x*(y*y)*6.0)*pow(x+1.0,2.0)*2.0-x*cos(x*y)*(x*2.0+(x*x)*(y*y*y*y)*3.0+y*(pow(cos(x*y),2.0)*2.0-1.0))
               -x*y*cos(x*y)*((x*x)*(y*y*y)*4.0+pow(cos(x*y),2.0)*2.0-1.0);
      default:
        Assert(false, ExcMessage("The RHS data for dim != 2 is not provided"));
      }
  }



  template <int dim>
  class PressureBoundaryValues : public Function<dim>
  {
  public:
    PressureBoundaryValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int component = 0) const override;
  };

  template <int dim>
  double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                             const unsigned int /*component*/) const
  {
    const double x = p[0];
    const double y = p[1];

    switch (dim)
      {
      case 2:
        return (x*x*x)*(y*y*y*y)+cos(x*y)*sin(x*y)+x*x;
      default:
        Assert(false, ExcMessage("The BC data for dim != 2 is not provided"));
      }
  }



  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double> &value) const override;

    virtual void vector_gradient (const Point<dim> &p,
                                  std::vector<Tensor<1,dim,double>>  &grads) const override;
  };

  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    Assert (values.size() == dim+1,
            ExcDimensionMismatch (values.size(), dim+1));

    const double x = p[0];
    const double y = p[1];

    switch (dim)
      {
      case 2:
        values(0) = -(x*2.0+(x*x)*(y*y*y*y)*3.0+y*cos(x*y*2.0))*(x*2.0+x*x+y*y+1.0)-x*sin(x*y)*(cos(x*y*2.0)+(x*x)*(y*y*y)*4.0);
        values(1) = -sin(x*y)*(x*2.0+(x*x)*(y*y*y*y)*3.0+y*cos(x*y*2.0))-x*(cos(x*y*2.0)+(x*x)*(y*y*y)*4.0)*pow(x+1.0,2.0);
        values(2) = (x*x*x)*(y*y*y*y)+cos(x*y)*sin(x*y)+x*x;
        break;
      default:
        Assert(false, ExcMessage("The exact solution for dim != 2 is not provided"));
      }
  }

  template <int dim>
  void
  ExactSolution<dim>::vector_gradient (const Point<dim> &p,
                                       std::vector<Tensor<1,dim,double>> &grads) const
  {
    const double x = p[0];
    const double y = p[1];

    switch (dim)
      {
      case 2:
        grads[0][0] = -(x*(y*y*y*y)*6.0-(y*y)*sin(x*y*2.0)*2.0+2.0)*(x*2.0+x*x+y*y+1.0)-sin(x*y)*(cos(x*y*2.0)
                      +(x*x)*(y*y*y)*1.2E1-x*y*sin(x*y*2.0)*2.0)-(x*2.0+2.0)*(x*2.0+(x*x)*(y*y*y*y)*3.0
                          +y*cos(x*y*2.0))-x*y*cos(x*y)*((x*x)*(y*y*y)*4.0+pow(cos(x*y),2.0)*2.0-1.0);
        grads[0][1] = -(cos(x*y*2.0)+(x*x)*(y*y*y)*1.2E1-x*y*sin(x*y*2.0)*2.0)*(x*2.0+x*x+y*y+1.0)
                      -y*(x*2.0+(x*x)*(y*y*y*y)*3.0+y*cos(x*y*2.0))*2.0-(x*x)*cos(x*y)*((x*x)*(y*y*y)*4.0
                          +pow(cos(x*y),2.0)*2.0-1.0)+(x*x)*sin(x*y)*(sin(x*y*2.0)-x*(y*y)*6.0)*2.0;
        grads[1][0] = -sin(x*y)*(x*(y*y*y*y)*6.0-(y*y)*sin(x*y*2.0)*2.0+2.0)-pow(x+1.0,2.0)*(cos(x*y*2.0)
                      +(x*x)*(y*y*y)*1.2E1-x*y*sin(x*y*2.0)*2.0)-x*(cos(x*y*2.0)+(x*x)*(y*y*y)*4.0)*(x*2.0+2.0)
                      -y*cos(x*y)*(x*2.0+(x*x)*(y*y*y*y)*3.0+y*(pow(cos(x*y),2.0)*2.0-1.0));
        grads[1][1] = -sin(x*y)*(cos(x*y*2.0)+(x*x)*(y*y*y)*1.2E1-x*y*sin(x*y*2.0)*2.0)+(x*x)*(sin(x*y*2.0)
                      -x*(y*y)*6.0)*pow(x+1.0,2.0)*2.0-x*cos(x*y)*(x*2.0+(x*x)*(y*y*y*y)*3.0
                                                                   +y*(pow(cos(x*y),2.0)*2.0-1.0));
        grads[2] = 0;
        break;
      default:
        Assert(false, ExcMessage("The exact solution's gradient for dim != 2 is not provided"));
      }
  }



  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const override;
  };

  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));

    for (unsigned int p=0; p<points.size(); ++p)
      {
        values[p].clear ();

        const double x = points[p][0];
        const double y = points[p][1];

        switch (dim)
          {
          case 2:
            values[p][0][0] = pow(x+1.0,2.0)/(x*4.0+(x*x)*(y*y)-pow(sin(x*y),2.0)+x*(y*y)*2.0+(x*x)*6.0+(x*x*x)*4.0+x*x*x*x+y*y+1.0);
            values[p][0][1] = -sin(x*y)/(x*4.0+(x*x)*(y*y)-pow(sin(x*y),2.0)+x*(y*y)*2.0+(x*x)*6.0+(x*x*x)*4.0+x*x*x*x+y*y+1.0);
            values[p][1][0] = -sin(x*y)/(x*4.0+(x*x)*(y*y)-pow(sin(x*y),2.0)+x*(y*y)*2.0+(x*x)*6.0+(x*x*x)*4.0+x*x*x*x+y*y+1.0);
            values[p][1][1] = (x*2.0+x*x+y*y+1.0)/(x*4.0+(x*x)*(y*y)-pow(sin(x*y),2.0)+x*(y*y)*2.0+(x*x)*6.0+(x*x*x)*4.0+x*x*x*x+y*y+1.0);
            break;
          default:
            Assert(false, ExcMessage("The inverse of permeability tensor for dim != 2 is not provided"));
          }
      }
  }
}

#endif //MFMFE_DATA_H
