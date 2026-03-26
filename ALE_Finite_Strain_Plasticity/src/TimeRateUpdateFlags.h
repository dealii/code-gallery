/*
 * TimeRateUpdateFlags.h
 *
 *  Created on: 27 Nov 2019
 *      Author: maien
 */

#ifndef TIMERATEUPDATEFLAGS_H_
#define TIMERATEUPDATEFLAGS_H_

namespace PlasticityLab {

  enum TimeRateUpdateFlags {
    default_timerate_update_flags = 0x0000,
    update_partial_time_rate = 0x0001,
    update_total_time_rate = 0x0002,
    update_partial_second_time_rate=0x0004,
    update_total_second_time_rate=0x0008,
    update_partial_time_rate_tangent=0x0010,
    update_total_time_rate_tangent=0x0020,
    update_partial_second_time_rate_tangent=0x0040,
    update_total_second_time_rate_tangent=0x0080
  };

  inline
  TimeRateUpdateFlags
  operator | (TimeRateUpdateFlags f1, TimeRateUpdateFlags f2) {
    return static_cast<TimeRateUpdateFlags> (
             static_cast<unsigned int> (f1) |
             static_cast<unsigned int> (f2));
  }

  inline
  const TimeRateUpdateFlags &
  operator |= (TimeRateUpdateFlags &f1, TimeRateUpdateFlags f2) {
    f1 = f1 | f2;
    return f1;
  }

  inline
  TimeRateUpdateFlags
  operator & (TimeRateUpdateFlags f1, TimeRateUpdateFlags f2) {
    return static_cast<TimeRateUpdateFlags> (
             static_cast<unsigned int> (f1) &
             static_cast<unsigned int> (f2));
  }

  inline
  const TimeRateUpdateFlags &
  operator &= (TimeRateUpdateFlags &f1, TimeRateUpdateFlags f2) {
    f1 = f1 & f2;
    return f1;
  }

} /* namespace PlasticityLab */

#endif /*TIMERATEUPDATEFLAGS_H_*/
