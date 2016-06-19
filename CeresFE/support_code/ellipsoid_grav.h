
/*
 Started 10/8/2012, R. R. Fu
 
 Reference Pohanka 2011, Contrib. Geophys. Geodes.
 
 */
#include <math.h>
#include <deal.II/base/point.h>
#include <fstream>
#include <iostream>

namespace A_Grav_namespace
{
	namespace system_parameters {
		double mantle_rho;
		double core_rho;
		double excess_rho;
		double r_eq;
		double r_polar;
		
		double r_core_eq; 
		double r_core_polar;
	}
	
template <int dim>
class AnalyticGravity 
{
  public:
    void setup_vars (std::vector<double> v);
	void get_gravity (const dealii::Point<dim> &p, std::vector<double> &g);

  private:	
	double ecc;
	double eV;
	double ke;
	double r00;
	double r01;
	double r11;
	double ecc_c;
	double eV_c;
	double ke_c;
	double r00_c;
	double r01_c;
	double r11_c;
	double g_coeff;
	double g_coeff_c;
};

template <int dim>
void AnalyticGravity<dim>::get_gravity (const dealii::Point<dim> &p, std::vector<double> &g)
	{
		double rsph = std::sqrt(p[0] * p[0] + p[1] * p[1]);
		double thetasph = std::atan2(p[0], p[1]);
		double costhetasph = std::cos(thetasph);
		
		//convert to elliptical coordinates for silicates
		double stemp = std::sqrt((rsph * rsph - eV * eV + std::sqrt((rsph * rsph - eV * eV) * (rsph * rsph - eV * eV)
								  + 4 * eV * eV * rsph * rsph * costhetasph *costhetasph)) / 2);
		double vout = stemp / system_parameters::r_eq / std::sqrt(1 - ecc * ecc);
		double eout = std::acos(rsph * costhetasph / stemp);
		
		//convert to elliptical coordinates for core correction
		double stemp_c = std::sqrt((rsph * rsph - eV_c * eV_c + std::sqrt((rsph * rsph - eV_c * eV_c) * (rsph * rsph - eV_c * eV_c)
																	+ 4 * eV_c * eV_c * rsph * rsph * costhetasph *costhetasph)) / 2);
		double vout_c = stemp_c / system_parameters::r_core_eq / std::sqrt(1 - ecc_c * ecc_c);
		double eout_c = std::acos(rsph * costhetasph / stemp_c);
		
		//shell contribution
		g[0] = g_coeff * r11 * std::sqrt((1 - ecc * ecc) * vout * vout + ecc * ecc) * std::sin(eout);
		g[1] = g_coeff * r01 * vout * std::cos(eout) / std::sqrt(1 - ecc * ecc);
		
		//core contribution		
		double expected_y = system_parameters::r_core_polar * std::sqrt(1 - 
							(p[0] * p[0] / system_parameters::r_core_eq / system_parameters::r_core_eq));

		
		if(p[1] <= expected_y)
		{
			g[0] += g_coeff_c * r11_c * std::sqrt((1 - ecc_c * ecc_c) * vout_c * vout_c + ecc_c * ecc_c) * std::sin(eout_c);
			g[1] += g_coeff_c * r01_c * vout_c * std::cos(eout_c) / std::sqrt(1 - ecc_c * ecc_c);
		}
		else
		{
			double g_coeff_co = - 2.795007963255562e-10 * system_parameters::excess_rho * system_parameters::r_core_eq
			/ vout_c / vout_c;
			double r00_co = 0;
			double r01_co = 0;
			double r11_co = 0;
			
			if(system_parameters::r_core_polar == system_parameters::r_core_eq)
			{
				r00_co = 1;
				r01_co = 1;
				r11_co = 1;
			}
			else
			{
				r00_co = ke_c * vout_c * std::atan2(1, ke_c * vout_c);
				double ke_co2 = ke_c * ke_c * vout_c * vout_c;
				r01_co = 3 * ke_co2 * (1 - r00_co);
				r11_co = 3 * ((ke_co2 + 1) * r00_co - ke_co2) / 2;
			}
			g[0] += g_coeff_co * vout_c * r11_co / std::sqrt((1 - ecc_c* ecc_c) * vout_c * vout_c + ecc_c * ecc_c) * std::sin(eout_c);
			g[1] += g_coeff_co * r01_co * std::cos(eout_c) / std::sqrt(1 - ecc_c * ecc_c);
		}
	}
	
template <int dim>
void AnalyticGravity<dim>::setup_vars (std::vector<double> v) 
{
	system_parameters::r_eq = v[0];
	system_parameters::r_polar = v[1];
	system_parameters::r_core_eq = v[2]; 
	system_parameters::r_core_polar = v[3];
	system_parameters::mantle_rho = v[4];
	system_parameters::core_rho = v[5];
	system_parameters::excess_rho = system_parameters::core_rho - system_parameters::mantle_rho;
		
		
	// Shell
	if (system_parameters::r_polar > system_parameters::r_eq)
	{
		//This makes the gravity field nearly that of a sphere in case the body becomes prolate
		std::cout << "\nWarning: The model body has become prolate. \n";
		ecc = 0.001;
	}
	else
	{
		ecc = std::sqrt(1 - (system_parameters::r_polar * system_parameters::r_polar / system_parameters::r_eq / system_parameters::r_eq));
	}
	
	eV = ecc * system_parameters::r_eq;
	ke = std::sqrt(1 - (ecc * ecc)) / ecc;
	r00 = ke * std::atan2(1, ke);
	double ke2 = ke * ke;
	r01 = 3 * ke2 * (1 - r00);
	r11 = 3 * ((ke2 + 1) * r00 - ke2) / 2;
	g_coeff = - 2.795007963255562e-10 * system_parameters::mantle_rho * system_parameters::r_eq;
	
	// Core
	if (system_parameters::r_core_polar > system_parameters::r_core_eq)
	{
		std::cout << "\nWarning: The model core has become prolate. \n";
		ecc_c = 0.001;
	}
	else
	{
		ecc_c = std::sqrt(1 - (system_parameters::r_core_polar * system_parameters::r_core_polar / system_parameters::r_core_eq / system_parameters::r_core_eq));
	}
	eV_c = ecc_c * system_parameters::r_core_eq;
	if(system_parameters::r_core_polar == system_parameters::r_core_eq)
	{
		ke_c = 1;
		r00_c = 1;
		r01_c = 1;
		r11_c = 1;
		g_coeff_c = - 2.795007963255562e-10 * system_parameters::excess_rho * system_parameters::r_core_eq;
	}
	else
	{
		ke_c = std::sqrt(1 - (ecc_c * ecc_c)) / ecc_c;
		r00_c = ke_c * std::atan2(1, ke_c);
		double ke2_c = ke_c * ke_c;
		r01_c = 3 * ke2_c * (1 - r00_c);
		r11_c = 3 * ((ke2_c + 1) * r00_c - ke2_c) / 2;
		g_coeff_c = - 2.795007963255562e-10 * system_parameters::excess_rho * system_parameters::r_core_eq;
	}
//			std::cout << "Loaded variables: ecc = " << ecc_c  << " ke = " << ke_c  << " r00 = " << r00_c  << " r01 = " << r01_c << " r11 = " << r11_c << "\n";	
}
}
