#include "PhaseFieldSolver.h"

void PhaseFieldSolver::applying_bc(){
    FEValuesExtractors::Scalar phase_parameter(0);
    FEValuesExtractors::Scalar temperature(1);

    QGauss<2> quadrature_formula(fe.degree + 1);
    FEValues<2> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

    ComponentMask p_mask = fe.component_mask(phase_parameter);
    ComponentMask t_mask = fe.component_mask(temperature);

    std::map<types::global_dof_index,double> boundary_values;

    // Prescribing p=1 at the left face (this will be maintained in the subsequent iterations when zero BC is applied in the Newton-Raphson iterations)
    VectorTools::interpolate_boundary_values (dof_handler,
                                              1,
                                              Functions::ConstantFunction<2>(1., 2),
                                              boundary_values,p_mask);

    // To apply the boundary values only to the solution vector without the Jacobian Matrix and RHS Vector
    for (auto &boundary_value : boundary_values)
        old_solution(boundary_value.first) = boundary_value.second;

    old_solution.compress(VectorOperation::insert);

}
