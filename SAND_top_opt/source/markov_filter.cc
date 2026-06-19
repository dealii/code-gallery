//
// Created by justin on 2/17/21.
//
#include "../include/markov_filter.h"

using namespace dealii;

///Initialized the markov filter with the initial values
void
MarkovFilter::setup(const double objective_value_input, const double barrier_distance_input,
                                        const double feasibility_input, const double barrier_value_input) {
    objective_value = objective_value_input;
    barrier_distance = barrier_distance_input;
    feasibility = feasibility_input;
    barrier_value = barrier_value_input;

    filter_barrier_function_value = feasibility + barrier_value * barrier_distance;
}

///Adds new information to the markov filter
void
MarkovFilter::add_point(const double objective_value_input, const  double barrier_distance_input,
                                    const double feasibility_input)
{
    objective_value = objective_value_input;
    barrier_distance = barrier_distance_input;
    feasibility = feasibility_input;

    filter_barrier_function_value = objective_value + barrier_value * barrier_distance;
}

///As the barrier always changes, this needs to be taken into account when accepting/rejecting a step.
/// This allows each point to be viewed in comparison to the current barrier value.
void
MarkovFilter::update_barrier_value(const double barrier_value_input)
{
    barrier_value = barrier_value_input;
    filter_barrier_function_value = objective_value + barrier_value * barrier_distance;
}

///Checks if a new point passes the filter.
bool
MarkovFilter::check_filter(const double objective_value_input, const  double barrier_distance_input,
                                       const double feasibility_input) const
{
    if ((objective_value_input + barrier_distance_input * barrier_value <= filter_barrier_function_value) ||
        (feasibility_input <= feasibility))
    {
        return true;
    }
    else
    {
        return false;
    }
}
