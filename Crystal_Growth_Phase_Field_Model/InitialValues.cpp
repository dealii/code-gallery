#include "PhaseFieldSolver.h"

void InitialValues::vector_value(const Point<2> &p,
                                 Vector<double> & values) const
{
    values(0)= 0.0; //Initial p value of domain
    values(1)= 0.2; //Initial temperature of domain
}
