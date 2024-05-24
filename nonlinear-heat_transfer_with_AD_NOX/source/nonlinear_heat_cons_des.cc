#include "nonlinear_heat.h"
/**
 * This is the constructor. All the variables defined in nonlinear_heat.h are
 * given values here.
 */
nonlinear_heat::nonlinear_heat ()
    :delta_t(0.1),
    alpha(0.5),
    tot_time(5),
    a(0.3),
    b(0.003),
    c(0),
    Cp(1),
    rho(1),
    dof_handler(triangulation),
    fe(FE_Q<2>(1), 1)
{}
/**
 * This is the destructor
 */
nonlinear_heat::~nonlinear_heat()
{
    dof_handler.clear();
}
    

