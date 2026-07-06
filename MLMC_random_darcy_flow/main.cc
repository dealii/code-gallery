#include "include/random_permeability.h"
#include "include/random_darcy.h"
#include "include/mlmc.h"

#include <fstream>
#include <iostream>

int main()
{
    using namespace dealii;

    std::ofstream outfile("mlmc_results.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open mlmc_results.txt for writing!" << std::endl;
        return 1;
    }

    // Run parameters
    double domain_length = 10.0;

    /* Describes how strongly different points are correlated. 
    A higher correlation length means the random field is more "spread out." 
    A shorter correlation length increases local variance, which requires more KL terms 
    to capture the higher frequencies. This is similar to a Fourier series, where a 
    function with high-frequency oscillations requires more terms for a sufficient approximation. */ 
    double correlation_length = 5.0;

    // The number of terms chosen for 1D. For higher dimensions, we use oneD_samples^dim.
    unsigned int oneD_samples = 20;

    // The number of levels was not chosen arbitrarily. Convergence is usually so fast 
    // that using more than 4–5 levels is rarely necessary.
    unsigned int levels = 5;

    // We used a constant mean here. While this was a specific choice for this case, 
    // in practice, most mean functions are spatially varying rather than constant.
    double mu = 1.0;

    /* This tolerance should not be confused with the tolerances seen in standard numerical 
    convergence studies or linear solvers. Those low tolerances (on the order of 1e-8) 
    would be realistically unachievable here; instead, we use a more moderate tolerance 
    that keeps the error "small enough." A typical value used in research is 1e-2 to 1e-4 
    at most. A larger value was chosen here for demonstration purposes, but the results 
    are still quite acceptable. Note that this value is squared later, so the actual 
    tolerance used to abort the runs is smaller. */
    double tolerance = 1e-1; 

    // In general, these values should be determined by a pilot run. 
    // I chose these specific values to keep the implementation simple.
    std::vector<unsigned int> runs_per_level{20000, 10000, 5000, 2500, 1000, 250};

    MultilevelMonteCarlo::MLMC<2> mlmc(oneD_samples);
    std::vector<double> first_sample{};

    /* The construction of this class also computes the KL expansion. 
    We compute the expansion once and then reuse it by swapping out the samples. */
    RandomField::RandomPermeability<2> permeability(first_sample, oneD_samples, domain_length, correlation_length, mu);

    Discretization::RandomDarcy<2> random_darcy;

    random_darcy.generate_mesh(domain_length);

    double global_mean = 0.0;



    for(unsigned int i = 0; i<=levels; i++)
    {
        std::cout << "Level:" << i << std::endl;
        for(unsigned int j = 0; j<runs_per_level[i]; j++)
        {
            permeability.overwrite_samples(mlmc.generate_samples());
            
            //coarse run
            random_darcy.set_tria(false);
            random_darcy.setup_system();
            random_darcy.assemble_system(permeability);
            random_darcy.solve();
            double Keff_coarse = random_darcy.compute_Keff(permeability);
            if( i == 0)
            {
                mlmc.add_sample(Keff_coarse);
            }
            

            // fine run
            if(i!=0)
            {
                random_darcy.set_tria(true);
                random_darcy.setup_system();
                random_darcy.assemble_system(permeability);
                random_darcy.solve();
                double Keff_fine = random_darcy.compute_Keff(permeability);

                mlmc.add_sample(Keff_fine-Keff_coarse);

            }

            if( j >=2)
            {
                double var = mlmc.compute_variance();
                
                if (var / (j+1) < tolerance * tolerance)
                {
                    global_mean+= mlmc.compute_mean();
                    std::cout << "number of samples for level" << i << ":" << j << std::endl;
                    random_darcy.output_results(i==0, i);

                    outfile << "FINAL_STATS Level:" << i << " Mean:" << mlmc.compute_mean() 
                            << " Var:" << mlmc.compute_variance() << " Samples:" << j << std::endl;
                    break;
                }
            }
            

        }
        random_darcy.refine_grid(i==0);
        mlmc.clear_samples();
        std::cout <<"global mean" << global_mean << std::endl;
    }

    outfile.close();

    
    return 0;
}

