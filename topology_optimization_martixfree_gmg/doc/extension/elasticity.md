# Extension: Vector-Valued Linear Elasticity

The scalar heat-conduction example can be adapted to a static linear-elasticity topology-optimization problem. The main changes are to replace the scalar diffusion operator with an elasticity operator and to use a vector-valued finite-element space. The density filter, MMA update, and most of the optimization-loop infrastructure can otherwise be reused.

## Finite-element space and operator

For a displacement field with `dim` components, replace the scalar finite-element space

```cpp
FE_Q<dim>(degree)
```

with the vector-valued space

```cpp
FE_Q<dim>(degree)^dim
```

A matrix-free elasticity operator can be derived from `MatrixFreeOperators::Base`:

```cpp
template <int dim, int fe_degree, typename number>
class Elasticity_Matrix_Free
  : public MatrixFreeOperators::Base<
        dim,
        LinearAlgebra::distributed::Vector<number>>
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;

  Elasticity_Matrix_Free();

  void clear() override;

  void evaluate_coefficient(
    const LinearAlgebra::distributed::Vector<number> &simp_rho,
    number material_lambda,
    number material_mu);

  void compute_diagonal() override;

private:
  void apply_add(VectorType &dst,
                 const VectorType &src) const override;

  void local_apply(
    const MatrixFree<dim, number> &data,
    VectorType &dst,
    const VectorType &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  void local_compute_diagonal(
    FEEvaluation<dim, fe_degree, fe_degree + 1, dim, number> &fe_values) const;

  Table<1, VectorizedArray<number>> lambda;
  Table<1, VectorizedArray<number>> mu;
};
```

The `FEEvaluation` object has `dim` components. The local constitutive response is described by the Lamé parameters $\lambda$ and $\mu$, interpolated from the design density using the SIMP model. For example,

@f[
\lambda(\rho)=\lambda_{\mathrm{solid}}
\left[\varepsilon_{\min}+(1-\varepsilon_{\min})\rho^p\right],
@f]

@f[
\mu(\rho)=\mu_{\mathrm{solid}}
\left[\varepsilon_{\min}+(1-\varepsilon_{\min})\rho^p\right].
@f]

A vectorized implementation should preserve all SIMD lanes when reading the cell density. In particular, using only `get_dof_value(0)[0]` would apply the density from lane zero to every cell in the batch. Instead, retain the complete `VectorizedArray` returned by the matrix-free evaluation and evaluate the SIMP expression lane-wise.

A schematic coefficient update is:

```cpp
const auto rho = fe_values.get_dof_value(0);
const auto simp =
  make_vectorized_array<number>(epsimin) +
  rho * rho * rho *
    make_vectorized_array<number>(1.0 - epsimin);

lambda(cell) = simp * make_vectorized_array<number>(material_lambda);
mu(cell)     = simp * make_vectorized_array<number>(material_mu);
```

The exact accessor and coefficient-storage details depend on the finite-element and `MatrixFree` setup used by the application.

## Solver considerations

The elasticity state equation has rigid-body modes unless sufficient displacement boundary conditions are imposed. The multigrid hierarchy must represent the essential boundary conditions on every level; otherwise, coarse-grid operators can retain unconstrained rigid-body modes and degrade or invalidate the preconditioner. Boundary ids should be established in a way that is consistent across mesh refinement levels.

The extension is intended to be built alongside the main project sources; its exact source-file layout depends on which elasticity example files are included in a particular checkout.
