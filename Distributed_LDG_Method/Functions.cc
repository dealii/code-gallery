// @sect3{Functions.cc}
// In this file we keep right hand side function, Dirichlet boundary
// conditions and solution to our Poisson equation problem.  Since
// these classes and functions have been discussed extensively in
// the deal.ii tutorials we won't discuss them any further.
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>

#include <cmath>

using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>(1)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
class DirichletBoundaryValues : public Function<dim>
{
public:
  DirichletBoundaryValues()
    : Function<dim>(1)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
class TrueSolution : public Function<dim>
{
public:
  TrueSolution()
    : Function<dim>(dim + 1)
  {}

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &valuess) const;
};

template <int dim>
double
RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const
{
  const double x = p[0];
  const double y = p[1];
  return 4 * M_PI * M_PI * (cos(2 * M_PI * y) - sin(2 * M_PI * x));
}

template <int dim>
double
DirichletBoundaryValues<dim>::value(const Point<dim> &p,
                                    const unsigned int) const
{
  const double x = p[0];
  const double y = p[1];
  return cos(2 * M_PI * y) - sin(2 * M_PI * x) - x;
}


template <int dim>
void
TrueSolution<dim>::vector_value(const Point<dim> &p,
                                Vector<double> &  values) const
{
  Assert(values.size() == dim + 1,
         ExcDimensionMismatch(values.size(), dim + 1));

  double x = p[0];
  double y = p[1];

  values(0) = 1 + 2 * M_PI * cos(2 * M_PI * x);
  values(1) = 2 * M_PI * sin(2 * M_PI * y);

  values(2) = cos(2 * M_PI * y) - sin(2 * M_PI * x) - x;
}
