# Extension: Automatic Differentiation for Adjoint Sensitivities

This note describes an automatic-differentiation (AD) approach for forming adjoint systems for nonlinear or coupled physical models. The main idea is to regard the discrete physical residual as a function of the local degrees of freedom and use AD to compute its linearization at the converged state.

The discussion is inspired by deal.II steps 33 and 72 and by approaches used in libraries such as Trilinos Panzer and FEniCS.

## Discrete adjoint formulation

Let the discrete state equation be

@f[
\mathbf R(\mathbf u,\boldsymbol\rho)=\mathbf 0,
@f]

and let the objective be $J(\mathbf u,\boldsymbol\rho)$. Differentiating the state equation with respect to the design variables gives

@f[
\frac{\partial\mathbf R}{\partial\boldsymbol\rho}
+
\frac{\partial\mathbf R}{\partial\mathbf u}
\frac{\partial\mathbf u}{\partial\boldsymbol\rho}
=\mathbf 0.
@f]

The total derivative of the objective is

@f[
\frac{\mathrm dJ}{\mathrm d\boldsymbol\rho}
=
\frac{\partial J}{\partial\boldsymbol\rho}
+
\frac{\partial J}{\partial\mathbf u}
\frac{\partial\mathbf u}{\partial\boldsymbol\rho}.
@f]

Eliminating the state derivative by introducing an adjoint vector $\boldsymbol\lambda$ gives

@f[
\left(\frac{\partial\mathbf R}{\partial\mathbf u}\right)^T
\boldsymbol\lambda
=-\left(\frac{\partial J}{\partial\mathbf u}\right)^T,
@f]

and the total design sensitivity becomes

@f[
\boxed{
\frac{\mathrm dJ}{\mathrm d\boldsymbol\rho}
=
\frac{\partial J}{\partial\boldsymbol\rho}
+
\boldsymbol\lambda^T
\frac{\partial\mathbf R}{\partial\boldsymbol\rho}
}.
@f]

This sign convention corresponds to the Lagrangian

@f[
\mathcal L=J+\boldsymbol\lambda^T\mathbf R.
@f]

If the opposite sign convention is used for the Lagrangian, the adjoint right-hand side and the final sensitivity expression must be changed consistently.

The adjoint method avoids forming the generally dense state sensitivity matrix $\partial\mathbf u/\partial\boldsymbol\rho$. When the number of design variables is much larger than the number of scalar objectives, one adjoint solve per objective is typically preferable to solving one tangent problem per design variable.

## Why AD is useful for nonlinear and coupled problems

For a linear scalar heat equation, a linear elasticity problem, or another simple self-adjoint model, the tangent and adjoint operators can often be derived by hand. For nonlinear constitutive laws or coupled multiphysics systems, the residual Jacobian contains many cross-derivative blocks. For example, a thermomechanical-electrical model may involve displacement $\mathbf u$, temperature $T$, and electric potential $V$:

@f[
\begin{aligned}
R_{\mathbf u}(\mathbf u,T;\rho)
&=\int_\Omega\boldsymbol\sigma(\mathbf u,T):\boldsymbol\varepsilon(\mathbf v)\,\mathrm dx
-\int_\Omega\mathbf f\cdot\mathbf v\,\mathrm dx,\\
R_T(T,V;\rho)
&=\int_\Omega\kappa\nabla T\cdot\nabla w\,\mathrm dx
-\int_\Omega\sigma_e(T)|\nabla V|^2w\,\mathrm dx,\\
R_V(V,T;\rho)
&=\int_\Omega\sigma_e(T)\nabla V\cdot\nabla\psi\,\mathrm dx
-\int_{\Gamma_N^\varphi}\overline J_n\psi\,\mathrm ds.
\end{aligned}
@f]

The coupled state vector is $[\mathbf u,T,V]^T$, and its Jacobian contains diagonal and off-diagonal blocks such as

@f[
\frac{\partial R_{\mathbf u}}{\partial T},\qquad
\frac{\partial R_T}{\partial V},\qquad
\frac{\partial R_V}{\partial T}.
@f]

The transpose of the complete coupled Jacobian is required by the adjoint equation. Deriving every block manually is possible, but can be tedious and error-prone. AD provides a way to differentiate the assembled local residual expressions consistently, including their nonlinear and coupling terms.

## Residual linearization with `ResidualLinearization`

The deal.II AD interface `Differentiation::AD::ResidualLinearization` can be used to calculate the derivative of a local residual vector with respect to the local degrees of freedom.

For an element with $N$ local degrees of freedom, define $N$ independent variables and $N+1$ dependent variables:

@f[
\mathbf F(\mathbf u_e)=
\begin{bmatrix}
R_1(\mathbf u_e)\\
\vdots\\
R_N(\mathbf u_e)\\
J_e(\mathbf u_e)
\end{bmatrix}.
@f]

The final dependent variable is the element contribution to the objective. AD then returns a matrix of size $(N+1)\times N$:

@f[
\frac{\partial\mathbf F}{\partial\mathbf u_e}
=
\begin{bmatrix}
\partial R_1/\partial u_1 & \cdots & \partial R_1/\partial u_N\\
\vdots & & \vdots\\
\partial R_N/\partial u_1 & \cdots & \partial R_N/\partial u_N\\
\partial J_e/\partial u_1 & \cdots & \partial J_e/\partial u_N
\end{bmatrix}.
@f]

The first $N$ rows provide the local residual Jacobian; the last row provides the local objective gradient. Transposing the first $N$ rows and taking the negative of the last row gives the local contributions to

@f[
\left(\frac{\partial\mathbf R}{\partial\mathbf u}\right)^T
\boldsymbol\lambda
=-\left(\frac{\partial J}{\partial\mathbf u}\right)^T.
@f]

## Illustrative nonlinear scalar example: Bratu-type equation

Consider the nonlinear PDE

@f[
-\Delta u-\lambda e^u=0
\quad\text{in }\Omega,
\qquad u=0\quad\text{on }\partial\Omega.
@f]

Its element residual contribution is

@f[
R_i(u)=\int_{\Omega_e}
\left(\nabla\phi_i\cdot\nabla u
-\lambda\phi_i e^u\right)\,\mathrm dx,
@f]

and choose the illustrative objective

@f[
J(u)=\int_\Omega|\nabla u|^2\,\mathrm dx.
@f]

The corresponding derivatives are

@f[
\frac{\partial R_i}{\partial u_j}
=\int_{\Omega_e}
\left(\nabla\phi_i\cdot\nabla\phi_j
-\lambda\phi_i\phi_j e^u\right)\,\mathrm dx,
@f]

and

@f[
\frac{\partial J}{\partial u_i}
=2\int_\Omega\nabla u\cdot\nabla\phi_i\,\mathrm dx.
@f]

These expressions provide an analytical reference against which the AD results can be checked.

## Representative implementation

The following code shows the essential local assembly pattern. It assumes that `fe`, `dof_handler`, `constraints`, `adjoint_system_matrix`, and `adjoint_system_rhs` are members of the surrounding class. The example uses a scalar finite-element field and assembles contributions from locally owned cells.

```cpp
template <unsigned int dim, typename VectorType>
void assemble_adjoint_system(const VectorType &solution,
                             const double lambda)
{
  using ADHelper = Differentiation::AD::ResidualLinearization<
    Differentiation::AD::NumberTypes::sacado_dfad,
    double>;
  using ADNumberType = typename ADHelper::ad_type;

  QGauss<dim> quadrature_formula(quadrature_degree + 1);
  FEValues<dim> fe_values(
    fe,
    quadrature_formula,
    update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();
  const unsigned int n_independent_variables = dofs_per_cell;
  const unsigned int n_dependent_variables = dofs_per_cell + 1;

  ADHelper ad_helper(n_independent_variables, n_dependent_variables);

  FullMatrix<double> jacobian_matrix(
    n_dependent_variables, n_independent_variables);
  FullMatrix<double> adjoint_matrix(
    n_independent_variables, n_independent_variables);
  Vector<double> adjoint_rhs(n_independent_variables);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<ADNumberType> solution_values(fe_values.n_quadrature_points);
  std::vector<Tensor<1, dim, ADNumberType>>
    solution_gradients(fe_values.n_quadrature_points);

  const FEValuesExtractors::Scalar scalar(0);
  using std::exp;

  adjoint_system_matrix = 0;
  adjoint_system_rhs = 0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      cell->get_dof_indices(local_dof_indices);
      fe_values.reinit(cell);

      ad_helper.reset(n_independent_variables, n_dependent_variables);
      ad_helper.register_dof_values(solution, local_dof_indices);

      const std::vector<ADNumberType> &dof_values_ad =
        ad_helper.get_sensitive_dof_values();

      fe_values[scalar].get_function_values_from_local_dof_values(
        dof_values_ad, solution_values);
      fe_values[scalar].get_function_gradients_from_local_dof_values(
        dof_values_ad, solution_gradients);

      std::vector<ADNumberType> residual_ad(
        n_dependent_variables, ADNumberType(0.0));

      for (const unsigned int q : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            residual_ad[i] +=
              (fe_values[scalar].gradient(i, q) * solution_gradients[q]
               - lambda * fe_values[scalar].value(i, q)
                 * exp(solution_values[q]))
              * fe_values.JxW(q);

          residual_ad[n_dependent_variables - 1] +=
            solution_gradients[q] * solution_gradients[q]
            * fe_values.JxW(q);
        }

      ad_helper.register_residual_vector(residual_ad);
      ad_helper.compute_linearization(jacobian_matrix);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          adjoint_rhs[i] =
            -jacobian_matrix(n_dependent_variables - 1, i);

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            adjoint_matrix(i, j) = jacobian_matrix(j, i);
        }

      constraints.distribute_local_to_global(
        adjoint_matrix,
        adjoint_rhs,
        local_dof_indices,
        adjoint_system_matrix,
        adjoint_system_rhs);
    }

  // Required for distributed matrix/vector types.
  adjoint_system_matrix.compress(VectorOperation::add);
  adjoint_system_rhs.compress(VectorOperation::add);
}
```

For distributed solution vectors, update ghost values before registering the local DoF values. For serial matrix and vector types, the final `compress()` calls are not needed. Boundary constraints for the adjoint problem must be consistent with the state problem; in particular, homogeneous Dirichlet constraints apply to the adjoint field for the standard fixed-Dirichlet-state setting.

## Verification strategy

Before using AD in a coupled topology-optimization problem, validate it on a single cell or a small mesh:

1. Compare every entry of the AD residual Jacobian with the analytical Bratu Jacobian above.
2. Compare the AD objective gradient with $2\int_\Omega\nabla u\cdot\nabla\phi_i\,\mathrm dx$.
3. Check the assembled adjoint residual after solving the transpose system.
4. Verify the final design sensitivity against a finite-difference perturbation of one or more design variables, using a sufficiently converged state and adjoint solve.
5. For a coupled model, check selected off-diagonal Jacobian blocks independently before assembling the full block adjoint system.

The Bratu example is an illustration of the AD assembly pattern; it is not the governing equation used by the main thermal-compliance benchmark.
