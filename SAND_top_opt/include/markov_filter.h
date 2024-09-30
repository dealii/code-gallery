//
// Created by justin on 2/17/21.
//

#ifndef SAND_MARKOV_FILTER_H
#define SAND_MARKOV_FILTER_H
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_direct.h>

using namespace dealii;

class MarkovFilter
{

public:
    void setup(const double objective_value, const double barrier_distance, const double feasibility, const double barrier_value);
    void add_point(const double objective_value, const double barrier_distance, const double feasibility);
    void update_barrier_value(const double barrier_value);
    bool check_filter(const double objective_value, const double barrier_distance,const double feasibility) const;

private:
    double objective_value;
    double barrier_distance;
    double feasibility;
    double barrier_value;
    double filter_barrier_function_value;
};


#endif //SAND_MARKOV_FILTER_H
