/*
 * ConstitModelUpdateFlags.h
 *
 *  Created on: 04 Jan 2015
 *      Author: maien
 */

#ifndef CONSTITMODELUPDATEFLAGS_H_
#define CONSTITMODELUPDATEFLAGS_H_

namespace PlasticityLab {

  enum ConstitutiveModelUpdateFlags {
    update_default = 0x0000,
    update_pressure = 0x0001,
    update_pressure_tangent = 0x0002,
    update_stress_deviator = 0x0004,
    update_stress_deviator_tangent = 0x0008,
    update_heat_flux = 0x0010,
    update_heat_flux_tangent = 0x0020,
    update_elastic_entropy = 0x0040,
    update_elastic_entropy_tangent = 0x0080,
    update_mechanical_dissipation = 0x0100,
    update_mechanical_dissipation_tangent = 0x0200,
    update_thermoelastic_heating = 0x0400,
    update_thermoelastic_heating_tangent = 0x0800,
    update_stored_heat = 0x1000,
    update_stored_heat_tangent = 0x2000,
    update_material_point_history = 0x4000
  };

  inline
  ConstitutiveModelUpdateFlags
  operator | (ConstitutiveModelUpdateFlags f1, ConstitutiveModelUpdateFlags f2) {
    return static_cast<ConstitutiveModelUpdateFlags> (
             static_cast<unsigned int> (f1) |
             static_cast<unsigned int> (f2));
  }

  inline
  const ConstitutiveModelUpdateFlags &
  operator |= (ConstitutiveModelUpdateFlags &f1, ConstitutiveModelUpdateFlags f2) {
    f1 = f1 | f2;
    return f1;
  }

  inline
  ConstitutiveModelUpdateFlags
  operator & (ConstitutiveModelUpdateFlags f1, ConstitutiveModelUpdateFlags f2) {
    return static_cast<ConstitutiveModelUpdateFlags> (
             static_cast<unsigned int> (f1) &
             static_cast<unsigned int> (f2));
  }

  inline
  const ConstitutiveModelUpdateFlags &
  operator &= (ConstitutiveModelUpdateFlags &f1, ConstitutiveModelUpdateFlags f2) {
    f1 = f1 & f2;
    return f1;
  }

} /* namespace PlasticityLab */

#endif /*CONSTITMODELUPDATEFLAGS_H_*/
