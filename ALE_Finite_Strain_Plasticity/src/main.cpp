//============================================================================
// Name        : main.cpp
// Author      : Maien Hamed
// Version     :
// Copyright   :
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include "utilities.h"
#include "ThermoPlasticMaterial.h"
#include "ExponentialHardeningThermoviscoplasticYieldLaw.h"
#include "JohnsonCookThermoviscoplasticYieldLaw.h"
#include "ExponentialHardeningElastoplasticMaterial.h"
#include "PlasticityLabProg.h"

#define DIM 2

PlasticityLab::ThermoPlasticMaterial<DIM+1, PlasticityLab::ExponentialHardeningThermoviscoplasticYieldLaw<double>, double>
getExponentialHardeningThermoPlasticMaterial();

PlasticityLab::ThermoPlasticMaterial<DIM+1, PlasticityLab::JohnsonCookThermoviscoplasticYieldLaw<double>, double>
getJohnsonCookThermoPlasticMaterial();

int main(int argc, char **argv) {
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    {
      deallog.depth_console(0);
      auto material = getExponentialHardeningThermoPlasticMaterial();
      PlasticityLab::PlasticityLabProg<DIM> plasticityLab(material);
      plasticityLab.run();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Finished." << std::endl; // prints
    }
  } catch (std::exception &exc) {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}

PlasticityLab::ThermoPlasticMaterial<DIM+1, PlasticityLab::ExponentialHardeningThermoviscoplasticYieldLaw<double>, double>
getExponentialHardeningThermoPlasticMaterial() {
  double kappa (164206.0), // [MPa]
         mu (801938.0), // [MPa]
         thermal_expansion_coefficient(1.0e-5), // [K^-1]
         thermal_conductivity(4.5e-2), // [J/mm.K.s]
         heat_capacity(3.588e-3), // cp*rho: [J.mm^-3.K^-1]
         K_0(450.0/*std::numeric_limits<double>::max()*/), // [MPa]
         K_infty(715.0/*std::numeric_limits<double>::max()*/), // [MPa]
         delta(16.93), // dimensionless
         H_bar(129.24), // [MPa]
         beta(1.0), // dimensionless (1.0 for isotropic hardening, 0.0 for kinematic hardening)
         flow_stress_softening(0.002), // [K^-1]
         hardening_softening(0.002), // [K^-1]
         dissipation_factor(0.9); // dimensionless

  PlasticityLab::ExponentialHardeningThermoviscoplasticYieldLaw<double> thermo_viscoplastic_yield_law(
          K_0,
          K_infty,
          delta,
          H_bar,
          beta,
          flow_stress_softening,
          hardening_softening);

  return PlasticityLab::ThermoPlasticMaterial<DIM+1, PlasticityLab::ExponentialHardeningThermoviscoplasticYieldLaw<double>, double>
         (kappa,
          mu,
          thermal_expansion_coefficient,
          thermal_conductivity,
          heat_capacity,
          dissipation_factor,
          thermo_viscoplastic_yield_law);
}

PlasticityLab::ThermoPlasticMaterial<DIM+1, PlasticityLab::JohnsonCookThermoviscoplasticYieldLaw<double>, double>
getJohnsonCookThermoPlasticMaterial() {
  double kappa (103300.0), // [MPa]
         mu (47690.0), // [MPa]
         A(89.7), // [MPa]
         B(291.87), // [MPa]
         C(0.025), // dimensionless
         m(1.09), // dimensionless
         n(0.31), // dimensionless
         melting_temperature(1356), // [K]
         reference_temperature(293.15), // [K]
         reference_strain_rate(1.0), // [s^-1]
         beta(1.0), //dimensionless
         thermal_expansion_coefficient(1.0e-5), // [K^-1]
         thermal_conductivity(4.5e-2), // [J/mm.K.s] // http://www.matweb.com/search/datasheet_print.aspx?matguid=193434cf42e343fab880e1dabdb143ba
         heat_capacity(3.588e-3), // cp*rho: [J.mm^-3.K^-1]
         dissipation_factor(0.9); // dimensionless

  PlasticityLab::JohnsonCookThermoviscoplasticYieldLaw<double> thermo_viscoplastic_yield_law(
          mu, A, B, C,
          m, n,
          melting_temperature,
          reference_strain_rate,
          reference_temperature);

  return PlasticityLab::ThermoPlasticMaterial<DIM+1, PlasticityLab::JohnsonCookThermoviscoplasticYieldLaw<double>, double>
         (kappa,
          mu,
          thermal_expansion_coefficient,
          thermal_conductivity,
          heat_capacity,
          dissipation_factor,
          thermo_viscoplastic_yield_law);
}
