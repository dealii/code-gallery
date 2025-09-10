#ifndef usrcodes_spectrum_decomposition_h
#define usrcodes_spectrum_decomposition_h
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/patterns.h>
#include <fstream>
#include <iostream>


namespace usr_spectrum_decomposition
{
  using namespace dealii;
  void print();

  double positive_ramp_function(const double x);

  double negative_ramp_function(const double x);

  double heaviside_function(const double x);

  // templated function has to be defined in the header file
  // perform a spectrum decomposition of a symmetric tensor
  // input: a symmetric tensor (SymmetricTensor<2, matrix_dimension>)
  // output: eigenvalues  (Vector<double>)
  //         eigenvectors (std::vector<Tensor<1, dim>>)
  template <int dim>
  void spectrum_decomposition(SymmetricTensor<2, dim> const & symmetric_tensor,
			      Vector<double> & myEigenvalues,
			      std::vector<Tensor<1, dim>> & myEigenvectors)
  {

    const std::array< std::pair< double, Tensor< 1, dim > >, dim >
      myEigenSystem = eigenvectors(symmetric_tensor);

    for (int i = 0; i < dim; i++)
      {
        myEigenvalues[i] = myEigenSystem[i].first;
        myEigenvectors[i] = myEigenSystem[i].second;
      }
  }

  template <int dim>
  SymmetricTensor<2, dim> positive_tensor(Vector<double> const & eigenvalues,
					  std::vector<Tensor<1, dim>> const & eigenvectors)
  {
    SymmetricTensor<2, dim> positive_part_tensor;
    positive_part_tensor = 0;
    for (int i = 0; i < dim; i++)
      positive_part_tensor += positive_ramp_function(eigenvalues[i])
                            * symmetrize(outer_product(eigenvectors[i],
                                                       eigenvectors[i]));
    return positive_part_tensor;
  }

  template <int dim>
  SymmetricTensor<2, dim> negative_tensor(Vector<double> const & eigenvalues,
					  std::vector<Tensor<1, dim>> const & eigenvectors)
  {
    SymmetricTensor<2, dim> negative_part_tensor;
    negative_part_tensor = 0;
    for (int i = 0; i < dim; i++)
      negative_part_tensor += negative_ramp_function(eigenvalues[i])
                            * symmetrize(outer_product(eigenvectors[i],
                                                       eigenvectors[i]));
    return negative_part_tensor;
  }

  template <int dim>
  void positive_negative_projectors(Vector<double> const & eigenvalues,
                                    std::vector<Tensor<1, dim>> const & eigenvectors,
			            SymmetricTensor<4, dim> & positive_projector,
				    SymmetricTensor<4, dim> & negative_projector)
  {
    Assert(dim <= 3,
	   ExcMessage("Project tensors only work for dim <= 3."));

    std::array<SymmetricTensor<2, dim>, dim> M;
    for (int a = 0; a < dim; a++)
      M[a] = symmetrize(outer_product(eigenvectors[a], eigenvectors[a]));

    std::array<SymmetricTensor<4, dim>, dim> Q;
    for (int a = 0; a < dim; a++)
      Q[a] = outer_product(M[a], M[a]);

    std::array<std::array<SymmetricTensor<4, dim>, dim>, dim> G;
    for (int a = 0; a < dim; a++)
      for (int b = 0; b < dim; b++)
	for (int i = 0; i < dim; i++)
	  for (int j = 0; j < dim; j++)
	    for (int k = 0; k < dim; k++)
              for (int l = 0; l < dim; l++)
        	G[a][b][i][j][k][l] = M[a][i][k] * M[b][j][l]
				    + M[a][i][l] * M[b][j][k];

    positive_projector = 0;
    for (int a = 0; a < dim; a++)
      {
	double lambda_a = eigenvalues[a];
	positive_projector += heaviside_function(lambda_a)
			    * Q[a];
	for (int b = 0; b < dim; b++)
	  {
	    if (b != a)
	      {
		double lambda_b = eigenvalues[b];
		double v_ab = 0.0;
		if (std::fabs(lambda_a - lambda_b) > 1.0e-12)
		  v_ab = (positive_ramp_function(lambda_a) - positive_ramp_function(lambda_b))
		       / (lambda_a - lambda_b);
		else
		  v_ab = 0.5 * (  heaviside_function(lambda_a)
		                + heaviside_function(lambda_b) );
		positive_projector += 0.5 * v_ab * 0.5 * (G[a][b] + G[b][a]);
	      }
	  }
      }

    negative_projector = 0;
    for (int a = 0; a < dim; a++)
      {
	double lambda_a = eigenvalues[a];
	negative_projector += heaviside_function(-lambda_a)
			    * Q[a];
	for (int b = 0; b < dim; b++)
	  {
	    if (b != a)
	      {
		double lambda_b = eigenvalues[b];
		double v_ab = 0.0;
		if (std::fabs(lambda_a - lambda_b) > 1.0e-12)
		  v_ab = (negative_ramp_function(lambda_a) - negative_ramp_function(lambda_b))
		       / (lambda_a - lambda_b);
		else
		  v_ab = 0.5 * (  heaviside_function(-lambda_a)
		                + heaviside_function(-lambda_b) );
		negative_projector += 0.5 * v_ab * 0.5 * (G[a][b] + G[b][a]);
	      }
	  }
      }

  }

}
#endif
