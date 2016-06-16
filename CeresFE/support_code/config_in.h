/*
 * config_in.h
 *
 *  Created on: Aug 17, 2015
 *      Author: antonermakov
 */


#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <libconfig.h++>

#include "local_math.h"

using namespace std;
using namespace libconfig;

namespace Step22 {
namespace system_parameters {

// Mesh file name
string mesh_filename;
string output_folder;

// Body parameters
double r_mean;
double period;
double omegasquared;
double beta;
double intercept;

// Rheology parameters
vector<double> depths_eta;
vector<double> eta_kinks;
vector<double> depths_rho;
vector<double> rho;
vector<int> material_id;
vector<double> G;

double eta_ceiling;
double eta_floor;
double eta_Ea;
bool lat_dependence;

unsigned int sizeof_depths_eta;
unsigned int sizeof_depths_rho;
unsigned int sizeof_rho;
unsigned int sizeof_eta_kinks;
unsigned int sizeof_material_id;
unsigned int sizeof_G;

double pressure_scale;
double q;
bool cylindrical;
bool continue_plastic_iterations;

// plasticity variables
bool plasticity_on;
unsigned int failure_criterion;
unsigned int max_plastic_iterations;
double smoothing_radius;

// viscoelasticity variables
unsigned int initial_elastic_iterations;
double elastic_time;
double viscous_time;
double initial_disp_target;
double final_disp_target;
double current_time_interval;

//mesh refinement variables
unsigned int global_refinement;
unsigned int small_r_refinement;
unsigned int crustal_refinement;
double crust_refine_region;
unsigned int surface_refinement;

//solver variables
int iteration_coefficient;
double tolerance_coefficient;

//time step variables
double present_time;
unsigned int present_timestep;
unsigned int total_viscous_steps;


// ellipse axes
vector<double> q_axes;
vector<double> p_axes;

}

class config_in
{
public:
	config_in(char*);
	
private:
	void write_config();
};

void config_in::write_config()
{
	std::ostringstream config_parameters;
	config_parameters << system_parameters::output_folder << "/run_parameters.txt";
	std::ofstream fout_config(config_parameters.str().c_str());

	// mesh filename
	fout_config << "mesh filename: " << system_parameters::mesh_filename << endl << endl;

	// body parameters
    fout_config << "r_mean = " << system_parameters::r_mean << endl;
    fout_config << "period = " << system_parameters::period << endl;
    fout_config << "omegasquared = " << system_parameters::omegasquared << endl;
    fout_config << "beta = " << system_parameters::beta << endl;
    fout_config << "intercept = " << system_parameters::intercept << endl;

    // rheology parameters

    for(unsigned int i=0; i<system_parameters::sizeof_depths_eta; i++)
        fout_config << "depths_eta[" << i << "] = " << system_parameters::depths_eta[i] << endl;

    for(unsigned int i=0; i<system_parameters::sizeof_eta_kinks; i++)
        fout_config << "eta_kinks[" << i << "] = " << system_parameters::eta_kinks[i] << endl;

    for(unsigned int i=0; i<system_parameters::sizeof_depths_rho; i++)
        fout_config << "depths_rho[" << i << "] = " << system_parameters::depths_rho[i] << endl;

    for(unsigned int i=0; i<system_parameters::sizeof_rho; i++)
        fout_config << "rho[" << i << "] = " << system_parameters::rho[i] << endl;

    for(unsigned int i=0; i<system_parameters::sizeof_material_id; i++)
        fout_config << "material_id[" << i << "] = " << system_parameters::material_id[i] << endl;

    for(unsigned int i=0; i<system_parameters::sizeof_G; i++)
        fout_config << "G[" << i << "] = " << system_parameters::G[i] << endl;

    fout_config << "eta_ceiling = " << system_parameters::eta_ceiling << endl;
    fout_config << "eta_floor = " << system_parameters::eta_floor << endl;
    fout_config << "eta_Ea = " << system_parameters::eta_Ea << endl;
    fout_config << "lat_dependence = " << system_parameters::lat_dependence << endl;
    fout_config << "pressure_scale = " << system_parameters::pressure_scale << endl;
    fout_config << "q = " << system_parameters::q << endl;
    fout_config << "cylindrical = " << system_parameters::cylindrical << endl;
    fout_config << "continue_plastic_iterations = " << system_parameters::continue_plastic_iterations << endl;

    // Plasticity parameters
    fout_config << "plasticity_on = " << system_parameters::plasticity_on << endl;
    fout_config << "failure_criterion = " << system_parameters::failure_criterion << endl;
    fout_config << "max_plastic_iterations = " << system_parameters::max_plastic_iterations << endl;
    fout_config << "smoothing_radius = " << system_parameters::smoothing_radius << endl;

    // Viscoelasticity parameters
    fout_config << "initial_elastic_iterations = " << system_parameters::initial_elastic_iterations << endl;
    fout_config << "elastic_time = " << system_parameters::elastic_time << endl;
    fout_config << "viscous_time = " << system_parameters::viscous_time << endl;
    fout_config << "initial_disp_target = " << system_parameters::initial_disp_target << endl;
    fout_config << "final_disp_target = " << system_parameters::final_disp_target << endl;
    fout_config << "current_time_interval = " << system_parameters::current_time_interval << endl;

    // Mesh refinement parameters
    fout_config << "global_refinement = " << system_parameters::global_refinement << endl;
    fout_config << "small_r_refinement = " << system_parameters::small_r_refinement << endl;
    fout_config << "crustal_refinement = " << system_parameters::crustal_refinement << endl;
    fout_config << "crust_refine_region = " << system_parameters::crust_refine_region << endl;
    fout_config << "surface_refinement = " << system_parameters::surface_refinement << endl;

    // Solver parameters
    fout_config << "iteration_coefficient = " << system_parameters::iteration_coefficient << endl;
	fout_config << "tolerance_coefficient = " << system_parameters::tolerance_coefficient << endl;

	// Time step parameters
	fout_config << "present_time = " << system_parameters::present_time << endl;
	fout_config << "present_timestep = " << system_parameters::present_timestep << endl;
	fout_config << "total_viscous_steps = " << system_parameters::total_viscous_steps << endl;
	
	fout_config.close();
}

config_in::config_in(char* filename)
{

	// This example reads the configuration file 'example.cfg' and displays
	// some of its contents.

	  Config cfg;

	  // Read the file. If there is an error, report it and exit.
	  try
	  {
	    cfg.readFile(filename);
	  }
	  catch(const FileIOException &fioex)
	  {
	    std::cerr << "I/O error while reading file:" << filename << std::endl;
	  }
	  catch(const ParseException &pex)
	  {
	    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
	              << " - " << pex.getError() << std::endl;
	  }

	  // Get mesh name.
	  try
	  {
        string msh = cfg.lookup("mesh_filename");
	    system_parameters::mesh_filename = msh;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
	    cerr << "No 'mesh_filename' setting in configuration file." << endl;
	  }

	  // get output folder

	  try
	  {
        string output = cfg.lookup("output_folder");
	    system_parameters::output_folder = output;

	    std::cout << "Writing to folder: " << output << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
	    cerr << "No 'output_folder' setting in configuration file." << endl;
	  }

	  // get radii

	  const Setting& root = cfg.getRoot();


	  // get body parameters
	  try
	  {
	    const Setting& body_parameters = root["body_parameters"];

	    body_parameters.lookupValue("period", system_parameters::period);
	    system_parameters::omegasquared = pow(TWOPI / 3600.0 / system_parameters::period, 2.0);
	    body_parameters.lookupValue("r_mean", system_parameters::r_mean);
	    body_parameters.lookupValue("beta", system_parameters::beta);
	    body_parameters.lookupValue("intercept", system_parameters::intercept);
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the body parameters block" << endl;

	  }

	// Rheology parameters

	  try
	  {
		  // get depths_eta ---------------------
	      const Setting& set_depths_eta = cfg.lookup("rheology_parameters.depths_eta");

		  unsigned int ndepths_eta = set_depths_eta.getLength();
		  system_parameters::sizeof_depths_eta = ndepths_eta;

		  for(unsigned int i=0; i<ndepths_eta; i++)
		  {
		      system_parameters::depths_eta.push_back(set_depths_eta[i]);
			    cout << "depth_eta[" << i << "] = " << system_parameters::depths_eta[i] << endl;
		  }

//		   get eta_kinks -------------------------
	      const Setting& set_eta_kinks = cfg.lookup("rheology_parameters.eta_kinks");

		  unsigned int neta_kinks = set_eta_kinks.getLength();
		  system_parameters::sizeof_eta_kinks = neta_kinks;

	//        cout << "Number of depth = " << ndepths << endl;

		  for(unsigned int i=0; i<neta_kinks; i++)
		  {
		      system_parameters::eta_kinks.push_back(set_eta_kinks[i]);
			  cout << "eta_kinks[" << i << "] = " << system_parameters::eta_kinks[i] << endl;
		  }

		  // get depths_rho -------------------------
	      const Setting& set_depths_rho = cfg.lookup("rheology_parameters.depths_rho");

		  unsigned int ndepths_rho = set_depths_rho.getLength();
		  system_parameters::sizeof_depths_rho = ndepths_rho;

	//        cout << "Number of depth = " << ndepths << endl;

		  for(unsigned int i=0; i<ndepths_rho; i++)
		  {
		      system_parameters::depths_rho.push_back(set_depths_rho[i]);
    		 cout << "depths_rho[" << i << "] = " << system_parameters::depths_rho[i] << endl;
		  }

		  // get rho -------------------------
	      const Setting& set_rho = cfg.lookup("rheology_parameters.rho");

		  unsigned int nrho = set_rho.getLength();
		  system_parameters::sizeof_rho = nrho;

	//        cout << "Number of depth = " << ndepths << endl;

		  for(unsigned int i=0; i<nrho; i++)
		  {
		      system_parameters::rho.push_back(set_rho[i]);
   		      cout << "rho[" << i << "] = " << system_parameters::rho[i] << endl;
		  }

		  // get material_id -------------------------
	      const Setting& set_material_id = cfg.lookup("rheology_parameters.material_id");

		  unsigned int nmaterial_id = set_material_id.getLength();
		  system_parameters::sizeof_material_id = nmaterial_id;

	//        cout << "Number of depth = " << ndepths << endl;

		  for(unsigned int i=0; i<nmaterial_id; i++)
		  {
		      system_parameters::material_id.push_back(set_material_id[i]);
   		      cout << "material_id[" << i << "] = " << system_parameters::material_id[i] << endl;
		  }

		  // get G -------------------------
	      const Setting& set_G = cfg.lookup("rheology_parameters.G");

		  unsigned int nG = set_G.getLength();
		  system_parameters::sizeof_G = nG;

	//        cout << "Number of depth = " << ndepths << endl;

		  for(unsigned int i=0; i<nG; i++)
		  {
		      system_parameters::G.push_back(set_G[i]);
			    cout << "G[" << i << "] = " << system_parameters::G[i] << endl;
		  }

	    const Setting& rheology_parameters = root["rheology_parameters"];
	    rheology_parameters.lookupValue("eta_ceiling", system_parameters::eta_ceiling);
	    rheology_parameters.lookupValue("eta_floor", system_parameters::eta_floor);
	    rheology_parameters.lookupValue("eta_Ea", system_parameters::eta_Ea);
	    rheology_parameters.lookupValue("lat_dependence", system_parameters::lat_dependence);
	    rheology_parameters.lookupValue("pressure_scale", system_parameters::pressure_scale);
	    rheology_parameters.lookupValue("q", system_parameters::q);
	    rheology_parameters.lookupValue("cylindrical", system_parameters::cylindrical);
	    rheology_parameters.lookupValue("continue_plastic_iterations", system_parameters::continue_plastic_iterations);
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the rheology parameters block" << endl;
	  }

	  // Plasticity parameters
	  try
	  {

	    const Setting& plasticity_parameters = root["plasticity_parameters"];
	    plasticity_parameters.lookupValue("plasticity_on", system_parameters::plasticity_on);
	    plasticity_parameters.lookupValue("failure_criterion", system_parameters::failure_criterion);
	    plasticity_parameters.lookupValue("max_plastic_iterations", system_parameters::max_plastic_iterations);
	    plasticity_parameters.lookupValue("smoothing_radius", system_parameters::smoothing_radius);
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the plasticity parameters block" << endl;
	  }

	// Viscoelasticity parameters

	  try
	  {

	    const Setting& viscoelasticity_parameters = root["viscoelasticity_parameters"];
	    viscoelasticity_parameters.lookupValue("initial_elastic_iterations", system_parameters::initial_elastic_iterations);
	    viscoelasticity_parameters.lookupValue("elastic_time", system_parameters::elastic_time);
	    viscoelasticity_parameters.lookupValue("viscous_time", system_parameters::viscous_time);
	    viscoelasticity_parameters.lookupValue("initial_disp_target", system_parameters::initial_disp_target);
	    viscoelasticity_parameters.lookupValue("final_disp_target", system_parameters::final_disp_target);
	    viscoelasticity_parameters.lookupValue("current_time_interval", system_parameters::current_time_interval);

	    system_parameters::viscous_time *= SECSINYEAR;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the viscoelasticity parameters block" << endl;
	  }

	// Mesh refinement parameters
	  try
	  {

	    const Setting& mesh_refinement_parameters = root["mesh_refinement_parameters"];
	    mesh_refinement_parameters.lookupValue("global_refinement", system_parameters::global_refinement);
	    mesh_refinement_parameters.lookupValue("small_r_refinement", system_parameters::small_r_refinement);
	    mesh_refinement_parameters.lookupValue("crustal_refinement", system_parameters::crustal_refinement);
	    mesh_refinement_parameters.lookupValue("crust_refine_region", system_parameters::crust_refine_region);
	    mesh_refinement_parameters.lookupValue("surface_refinement", system_parameters::surface_refinement);
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		 cerr << "We've got a problem in the mesh refinement parameters block" << endl;
	  }

	  // Solver parameters
	  try
	  {
	    const Setting& solve_parameters = root["solve_parameters"];
	    solve_parameters.lookupValue("iteration_coefficient", system_parameters::iteration_coefficient);
	    solve_parameters.lookupValue("tolerance_coefficient", system_parameters::tolerance_coefficient);


	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the solver parameters block" << endl;
	  }

	  // Time step parameters
	  try
	  {
	    const Setting& time_step_parameters = root["time_step_parameters"];
	    time_step_parameters.lookupValue("present_time", system_parameters::present_time);
	    time_step_parameters.lookupValue("present_timestep", system_parameters::present_timestep);
	    time_step_parameters.lookupValue("total_viscous_steps", system_parameters::total_viscous_steps);
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the time step parameters block" << endl;
	  }

	  write_config();
}
}





