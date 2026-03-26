/*
 * TimeRateRequest.h
 *
 *  Created on: 27 Nov 2019
 *      Author: maien
 */

#ifndef TIMERATEREQUEST_H_
#define TIMERATEREQUEST_H_

#include <deal.II/base/tensor.h>

#include "Constants.h"
#include "TimeRateUpdateFlags.h"

namespace PlasticityLab {

  template <typename ValueType, int dim, typename Number=double>
  class TimeRateRequest {
   public:
    TimeRateRequest(TimeRateUpdateFlags);
    virtual ~TimeRateRequest();

    // Interface to be used by request client (FE system assembler)
    //   --request configuration stage--
    void set_value(const ValueType &value);
    void set_previous_value(const ValueType &previous_value);
    void set_second_previous_value(const ValueType &second_previous_value);

    void set_time_increment(const Number time_increment);
    void set_previous_time_increment(const Number previous_time_increment);

    void set_previous_partial_time_rate(const ValueType &previous_partial_time_rate);
    void set_previous_total_time_rate(const ValueType &previous_total_time_rate);

    void set_velocity(const dealii::Tensor<1, dim, Number> &velocity);

    // Interface to be used by request client (FE system assembler)
    //   --request response retrieval and interrogation stage--
    ValueType get_partial_time_rate() const;
    ValueType get_total_time_rate() const;
    ValueType get_partial_second_time_rate() const;
    ValueType get_total_second_time_rate() const;

    ValueType get_partial_time_rate_tangent(const ValueType &value_increment) const;
    ValueType get_partial_second_time_rate_tangent(const ValueType &value_increment) const;

    template<typename GradientType>
    ValueType get_total_time_rate_tangent(
      const ValueType &value_increment,
      const ValueType &gradient_increment) const;

    template<typename GradientType>
    ValueType get_total_second_time_rate_tangent(
      const ValueType &value_increment,
      const ValueType &gradient_increment) const;

    // interface used by constitutive model object to perform computation
    // TODO consider hiding this interface and exposing it through adapter
    TimeRateUpdateFlags get_update_flags() const;
    ValueType get_value() const;
    ValueType get_previous_value() const;
    ValueType get_second_previous_value() const;

    Number get_time_increment() const;
    Number get_previous_time_increment() const;

    ValueType get_previous_partial_time_rate() const;
    ValueType get_previous_total_time_rate() const;

    dealii::Tensor<1, dim, Number> get_velocity() const;

    // set the results
    void set_partial_time_rate(const ValueType &partial_time_rate);
    void set_total_time_rate(const ValueType &total_time_rate);
    void set_partial_second_time_rate(const ValueType &partial_second_time_rate);
    void set_total_second_time_rate(const ValueType &total_second_time_rate);

    void set_partial_time_rate_tangent(const ValueType &partial_time_rate_tangent);
    void set_total_time_rate_tangent(const ValueType &total_time_rate_tangent);
    void set_partial_second_time_rate_tangent(const ValueType &partial_second_time_rate_tangent);
    void set_total_second_time_rate_tangent(const ValueType &total_second_time_rate_tangent);

   protected:
    // inputs
    TimeRateUpdateFlags update_flags;

    ValueType value;
    ValueType previous_value;
    ValueType second_previous_value;

    Number time_increment;
    Number previous_time_increment;

    ValueType previous_partial_time_rate;
    ValueType previous_total_time_rate;

    dealii::Tensor<1, dim, Number> velocity;

    // outputs
    ValueType partial_time_rate;
    ValueType total_time_rate;
    ValueType partial_second_time_rate;
    ValueType total_second_time_rate;

    ValueType partial_time_rate_tangent;
    ValueType total_time_rate_tangent;
    ValueType partial_second_time_rate_tangent;
    ValueType total_second_time_rate_tangent;
  };


  template <typename ValueType, int dim, typename Number>
  TimeRateRequest<ValueType, dim, Number>::
  TimeRateRequest(TimeRateUpdateFlags update_flags):
    update_flags(update_flags) {
  }

  template <typename ValueType, int dim, typename Number>
  TimeRateRequest<ValueType, dim, Number>::~TimeRateRequest() { }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_value(const ValueType &value) {
    this->value = value;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_previous_value(const ValueType &previous_value) {
    this->previous_value = previous_value;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_second_previous_value(const ValueType &second_previous_value) {
    this->second_previous_value = second_previous_value;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_time_increment(const Number time_increment) {
    this->time_increment = time_increment;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_previous_time_increment(const Number previous_time_increment) {
    this->previous_time_increment = previous_time_increment;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_previous_partial_time_rate(const ValueType &previous_partial_time_rate) {
    this->previous_partial_time_rate = previous_partial_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_previous_total_time_rate(const ValueType &previous_total_time_rate) {
    this->previous_total_time_rate = previous_total_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_velocity(const dealii::Tensor<1, dim, Number> &velocity) {
    this->velocity = velocity;
  }




  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_partial_time_rate() const {
    return partial_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_total_time_rate() const {
    return total_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_partial_second_time_rate() const {
    return partial_second_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_total_second_time_rate() const {
    return total_second_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_partial_time_rate_tangent(const ValueType &value_increment) const {
    return partial_time_rate_tangent;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_partial_second_time_rate_tangent(const ValueType &value_increment) const {
    return partial_second_time_rate_tangent;
  }


  template <typename ValueType, int dim, typename Number>
  template<typename GradientType>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_total_time_rate_tangent(
    const ValueType &value_increment,
    const ValueType &gradient_increment) const {
    throw;
  }


  template <typename ValueType, int dim, typename Number>
  template<typename GradientType>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_total_second_time_rate_tangent(
    const ValueType &value_increment,
    const ValueType &gradient_increment) const {
    throw;
  }


  /*

  */
  template <typename ValueType, int dim, typename Number>
  TimeRateUpdateFlags TimeRateRequest<ValueType, dim, Number>::
  get_update_flags() const {
    return update_flags;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_value() const {
    return value;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_previous_value() const {
    return previous_value;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_second_previous_value() const {
    return second_previous_value;
  }


  template <typename ValueType, int dim, typename Number>
  Number TimeRateRequest<ValueType, dim, Number>::
  get_time_increment() const {
    return time_increment;
  }


  template <typename ValueType, int dim, typename Number>
  Number TimeRateRequest<ValueType, dim, Number>::
  get_previous_time_increment() const {
    return previous_time_increment;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_previous_partial_time_rate() const {
    return previous_partial_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  ValueType TimeRateRequest<ValueType, dim, Number>::
  get_previous_total_time_rate() const {
    return previous_total_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  dealii::Tensor<1, dim, Number> TimeRateRequest<ValueType, dim, Number>::
  get_velocity() const {
    return velocity;
  }


  /*

  */
  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_partial_time_rate(const ValueType &partial_time_rate) {
    this->partial_time_rate = partial_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_total_time_rate(const ValueType &total_time_rate) {
    this->total_time_rate = total_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_partial_second_time_rate(const ValueType &partial_second_time_rate) {
    this->partial_second_time_rate = partial_second_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_total_second_time_rate(const ValueType &total_second_time_rate) {
    this->total_second_time_rate = total_second_time_rate;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_partial_time_rate_tangent(const ValueType &partial_time_rate_tangent) {
    this->partial_time_rate_tangent = partial_time_rate_tangent;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_total_time_rate_tangent(const ValueType &total_time_rate_tangent) {
    this->total_time_rate_tangent = total_time_rate_tangent;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_partial_second_time_rate_tangent(const ValueType &partial_second_time_rate_tangent) {
    this->partial_second_time_rate_tangent = partial_second_time_rate_tangent;
  }


  template <typename ValueType, int dim, typename Number>
  void TimeRateRequest<ValueType, dim, Number>::
  set_total_second_time_rate_tangent(const ValueType &total_second_time_rate_tangent) {
    this->total_second_time_rate_tangent = total_second_time_rate_tangent;
  }


} /* namespace PlasticityLab */

#endif /* TIMERATEREQUEST_H_ */
