#include "nonlinear_heat.h"
/**
 * Returns the initial conditions.
 * @param p Point
 * @param comp component
 * @return
 */
double Initialcondition::value(const Point<2> & /*p*/, const unsigned int /*comp*/) const
{
    /**
     * In the current case, we asume that the initial conditions are zero everywhere.
     */
    return 0.0;
}
