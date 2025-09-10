#include "SpectrumDecomposition.h" // use double quote
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/exceptions.h>


#include <iostream>

namespace usr_spectrum_decomposition
{
  void print()
  {
    Assert(3 >= 2, dealii::StandardExceptions::ExcNotImplemented());
    dealii::Tensor<1, 3> Br_tilde;
    std::cout << Br_tilde << std::endl;
    std::cout << "Hello world!" << std::endl;
  }

  double positive_ramp_function(const double x)
  {
    return std::fmax(x, 0.0);
  }

  double negative_ramp_function(const double x)
  {
    return std::fmin(x, 0.0);
  }

  double heaviside_function(const double x)
  {
    if (std::fabs(x) < 1.0e-16)
      return 0.5;

    if (x > 0)
      return 1.0;
    else
      return 0.0;
  }
}  
