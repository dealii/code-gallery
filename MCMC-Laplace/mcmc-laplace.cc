/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 by the deal.II authors and Wolfgang Bangerth.
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Colorado State University, 2019.
 */



#include <deal.II/base/timer.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <random>

#include <deal.II/base/logstream.h>

using namespace dealii;


// The following is a namespace in which we define the solver of the PDE.
// The main class implements an abstract `Interface` class declared at
// the top, which provides for an `evaluate()` function that, given
// a coefficient vector, solves the PDE discussed in the Readme file
// and then evaluates the solution at the 169 mentioned points.
//
// The solver follows the basic layout of step-4, though it precomputes
// a number of things in the `setup_system()` function, such as the
// evaluation of the matrix that corresponds to the point evaluations,
// as well as the local contributions to matrix and right hand side.
//
// Rather than commenting on everything in detail, in the following
// we will only document those things that are not already clear from
// step-4 and a small number of other tutorial programs.
namespace ForwardSimulator
{
  class Interface
  {
  public:
    virtual Vector<double> evaluate(const Vector<double> &coefficients) = 0;

    virtual ~Interface() = default;
  };



  template <int dim>
  class PoissonSolver : public Interface
  {
  public:
    PoissonSolver(const unsigned int global_refinements,
                  const unsigned int fe_degree,
                  const std::string &dataset_name);
    virtual Vector<double>
    evaluate(const Vector<double> &coefficients) override;

  private:
    void make_grid(const unsigned int global_refinements);
    void setup_system();
    void assemble_system(const Vector<double> &coefficients);
    void solve();
    void output_results(const Vector<double> &coefficients) const;

    Triangulation<dim>        triangulation;
    FE_Q<dim>                 fe;
    DoFHandler<dim>           dof_handler;

    FullMatrix<double>        cell_matrix;
    Vector<double>            cell_rhs;
    std::map<types::global_dof_index,double> boundary_values;

    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;

    Vector<double>            solution;
    Vector<double>            system_rhs;

    std::vector<Point<dim>>   measurement_points;

    SparsityPattern           measurement_sparsity;
    SparseMatrix<double>      measurement_matrix;

    TimerOutput  timer;
    unsigned int nth_evaluation;

    const std::string &dataset_name;
  };



  template <int dim>
  PoissonSolver<dim>::PoissonSolver(const unsigned int global_refinements,
                                    const unsigned int fe_degree,
                                    const std::string &dataset_name)
    : fe(fe_degree)
    , dof_handler(triangulation)
    , timer(std::cout, TimerOutput::summary, TimerOutput::cpu_times)
    , nth_evaluation(0)
    , dataset_name(dataset_name)
  {
    make_grid(global_refinements);
    setup_system();
  }



  template <int dim>
  void PoissonSolver<dim>::make_grid(const unsigned int global_refinements)
  {
    Assert(global_refinements >= 3,
           ExcMessage("This program makes the assumption that the mesh for the "
                      "solution of the PDE is at least as fine as the one used "
                      "in the definition of the coefficient."));
    GridGenerator::hyper_cube(triangulation, 0, 1);
    triangulation.refine_global(global_refinements);

    std::cout << "   Number of active cells: " << triangulation.n_active_cells()
              << std::endl;
  }



  template <int dim>
  void PoissonSolver<dim>::setup_system()
  {
    // First define the finite element space:
    dof_handler.distribute_dofs(fe);

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    // Then set up the main data structures that will hold the discrete problem:
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp);
      sparsity_pattern.copy_from(dsp);

      system_matrix.reinit(sparsity_pattern);

      solution.reinit(dof_handler.n_dofs());
      system_rhs.reinit(dof_handler.n_dofs());
    }

    // And then define the tools to do point evaluation. We choose
    // a set of 13x13 points evenly distributed across the domain:
    {
      const unsigned int n_points_per_direction = 13;
      const double       dx = 1. / (n_points_per_direction + 1);

      for (unsigned int x = 1; x <= n_points_per_direction; ++x)
        for (unsigned int y = 1; y <= n_points_per_direction; ++y)
          measurement_points.emplace_back(x * dx, y * dx);

      // First build a full matrix of the evaluation process. We do this
      // even though the matrix is really sparse -- but we don't know
      // which entries are nonzero. Later, the `copy_from()` function
      // calls build a sparsity pattern and a sparse matrix from
      // the dense matrix.
      Vector<double>     weights(dof_handler.n_dofs());
      FullMatrix<double> full_measurement_matrix(n_points_per_direction *
                                                   n_points_per_direction,
                                                 dof_handler.n_dofs());

      for (unsigned int index = 0; index < measurement_points.size(); ++index)
        {
          VectorTools::create_point_source_vector(dof_handler,
                                                  measurement_points[index],
                                                  weights);
          for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
            full_measurement_matrix(index, i) = weights(i);
        }

      measurement_sparsity.copy_from(full_measurement_matrix);
      measurement_matrix.reinit(measurement_sparsity);
      measurement_matrix.copy_from(full_measurement_matrix);
    }

    // Next build the mapping from cell to the index in the 64-element
    // coefficient vector:
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        const unsigned int i = std::floor(cell->center()[0] * 8);
        const unsigned int j = std::floor(cell->center()[1] * 8);

        const unsigned int index = i + 8 * j;

        cell->set_user_index(index);
      }

    // Finally prebuild the building blocks of the linear system as
    // discussed in the Readme file:
    {
      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      const QGauss<dim>  quadrature_formula(fe.degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values(fe,
                              quadrature_formula,
                              update_values | update_gradients |
                                update_JxW_values);

      fe_values.reinit(dof_handler.begin_active());

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            10.0 *                              // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }

      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               ZeroFunction<dim>(),
                                               boundary_values);
    }
  }



  // Given that we have pre-built the matrix and right hand side contributions
  // for a (representative) cell, the function that assembles the matrix is
  // pretty short and straightforward:
  template <int dim>
  void PoissonSolver<dim>::assemble_system(const Vector<double> &coefficients)
  {
    Assert(coefficients.size() == 64, ExcInternalError());

    system_matrix = 0;
    system_rhs    = 0;

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const double coefficient = coefficients(cell->user_index());
        
        cell->get_dof_indices(local_dof_indices);        
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              system_matrix.add(local_dof_indices[i],
                                local_dof_indices[j],
                                coefficient * cell_matrix(i, j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }
    
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
  }


  // The same is true for the function that solves the linear system:
  template <int dim>
  void PoissonSolver<dim>::solve()
  {
    SparseDirectUMFPACK solver;
    solver.factorize(system_matrix);
    solver.vmult(solution, system_rhs);
  }



  // The following function outputs graphical data for the most recently
  // used coefficient and corresponding solution of the PDE. Collecting
  // the coefficient values requires translating from the 64-element
  // coefficient vector and the cells that correspond to each of these
  // elements. The rest remains pretty obvious, with the exception
  // of including the number of the current sample into the file name.
  template <int dim>
  void
  PoissonSolver<dim>::output_results(const Vector<double> &coefficients) const
  {
    Vector<float> coefficient_values(triangulation.n_active_cells());
    for (const auto &cell : triangulation.active_cell_iterators())
      coefficient_values[cell->active_cell_index()] =
        coefficients(cell->user_index());

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(coefficient_values, "coefficient");

    data_out.build_patches();

    std::ofstream output("solution-" +
                         Utilities::int_to_string(nth_evaluation, 10) + ".vtu");
    data_out.write_vtu(output);
  }



  // The following is the main function of this class: Given a coefficient
  // vector, it assembles the linear system, solves it, and then evaluates
  // the solution at the measurement points by applying the measurement
  // matrix to the solution vector. That vector of "measured" values
  // is then returned.
  //
  // The function will also output the solution in a graphical format
  // if you un-comment the corresponding statement in the third
  // code block. However, you may end up with a very large amount
  // of data: This code is producing, at the minimum, 10,000 samples
  // and creating output for each one of them is surely more data
  // than you ever want to see!
  //
  // At the end of the function, we output some timing information
  // every 10,000 samples.
  template <int dim>
  Vector<double>
  PoissonSolver<dim>::evaluate(const Vector<double> &coefficients)
  {
    {
      TimerOutput::Scope section(timer, "Building linear systems");
      assemble_system(coefficients);
    }

    {
      TimerOutput::Scope section(timer, "Solving linear systems");
      solve();
    }

    Vector<double> measurements(measurement_matrix.m());
    {
      TimerOutput::Scope section(timer, "Postprocessing");

      measurement_matrix.vmult(measurements, solution);
      Assert(measurements.size() == measurement_points.size(),
             ExcInternalError());

      /*  output_results(coefficients);  */
    }

    ++nth_evaluation;
    if (nth_evaluation % 10000 == 0)
      timer.print_summary();

    return std::move(measurements);
  }
} // namespace ForwardSimulator


// The following namespaces define the statistical properties of the Bayesian
// inverse problem. The first is about the definition of the measurement
// statistics (the "likelihood"), which we here assume to be a normal
// distribution $N(\mu,\sigma I)$ with mean value $\mu$ given by the
// actual measurement vector (passed as an argument to the constructor
// of the `Gaussian` class and standard deviation $\sigma$.
//
// For reasons of numerical accuracy, it is useful to not return the
// actual likelihood, but its logarithm. This is because these
// values can be very small, occasionally on the order of $e^{-100}$,
// for which it becomes very difficult to compute accurate
// values.
namespace LogLikelihood
{
  class Interface
  {
  public:
    virtual double log_likelihood(const Vector<double> &x) const = 0;

    virtual ~Interface() = default;
  };


  class Gaussian : public Interface
  {
  public:
    Gaussian(const Vector<double> &mu, const double sigma);

    virtual double log_likelihood(const Vector<double> &x) const override;

  private:
    const Vector<double> mu;
    const double         sigma;
  };

  Gaussian::Gaussian(const Vector<double> &mu, const double sigma)
    : mu(mu)
    , sigma(sigma)
  {}


  double Gaussian::log_likelihood(const Vector<double> &x) const
  {
    Vector<double> x_minus_mu = x;
    x_minus_mu -= mu;

    return -x_minus_mu.norm_sqr() / (2 * sigma * sigma);
  }
} // namespace LogLikelihood


// Next up is the "prior" imposed on the coefficients. We assume
// that the logarithms of the entries of the coefficient vector
// are all distributed as a Gaussian with given mean and standard
// deviation. If the logarithms of the coefficients are normally
// distributed, then this implies in particular that the coefficients
// can only be positive, which is a useful property to ensure the
// well-posedness of the forward problem.
//
// For the same reasons as for the likelihood above, the interface
// for the prior asks for returning the *logarithm* of the prior,
// instead of the prior probability itself.
namespace LogPrior
{
  class Interface
  {
  public:
    virtual double log_prior(const Vector<double> &x) const = 0;

    virtual ~Interface() = default;
  };


  class LogGaussian : public Interface
  {
  public:
    LogGaussian(const double mu, const double sigma);

    virtual double log_prior(const Vector<double> &x) const override;

  private:
    const double mu;
    const double sigma;
  };

  LogGaussian::LogGaussian(const double mu, const double sigma)
    : mu(mu)
    , sigma(sigma)
  {}


  double LogGaussian::log_prior(const Vector<double> &x) const
  {
    double log_of_product = 0;

    for (const auto &el : x)
      log_of_product +=
        -(std::log(el) - mu) * (std::log(el) - mu) / (2 * sigma * sigma);

    return log_of_product;
  }
} // namespace LogPrior



// The Metropolis-Hastings algorithm requires a method to create a new sample
// given a previous sample. We do this by perturbing the current (coefficient)
// sample randomly using a Gaussian distribution centered at the current
// sample. To ensure that the samples' individual entries all remain
// positive, we use a Gaussian distribution in logarithm space -- in other
// words, instead of *adding* a small perturbation with mean value zero,
// we *multiply* the entries of the current sample by a factor that
// is the exponential of a random number with mean zero. (Because the
// exponential of zero is one, this means that the most likely factors
// to multiply the existing sample entries by are close to one. And
// because the exponential of a number is always positive, we never
// get negative samples this way.)
//
// But the Metropolis-Hastings sampler doesn't just need a perturbed
// sample $y$ location given the current sample location $x$. It also
// needs to know the ratio of the probability of reaching $y$ from
// $x$, divided by the probability of reaching $x$ from $y$. If we
// were to use a symmetric proposal distribution (e.g., a Gaussian
// distribution centered at $x$ with a width independent of $x$), then
// these two probabilities would be the same, and the ratio one. But
// that's not the case for the Gaussian in log space. It's not
// terribly difficult to verify that in that case, for a single
// component the ratio of these probabilities is $y_i/x_i$, and
// consequently for all components of the vector together, the
// probability is the product of these ratios.
namespace ProposalGenerator
{
  class Interface
  {
  public:
    virtual
    std::pair<Vector<double>,double>
    perturb(const Vector<double> &current_sample) const = 0;

    virtual ~Interface() = default;
  };


  class LogGaussian : public Interface
  {
  public:
    LogGaussian(const unsigned int random_seed, const double log_sigma);

    virtual
    std::pair<Vector<double>,double>
    perturb(const Vector<double> &current_sample) const;

  private:
    const double         log_sigma;
    mutable std::mt19937 random_number_generator;
  };



  LogGaussian::LogGaussian(const unsigned int random_seed,
                           const double       log_sigma)
    : log_sigma(log_sigma)
  {
    random_number_generator.seed(random_seed);
  }


  std::pair<Vector<double>,double>
  LogGaussian::perturb(const Vector<double> &current_sample) const
  {
    Vector<double> new_sample = current_sample;
    double         product_of_ratios = 1;
    for (auto &x : new_sample)
      {
        const double rnd = std::normal_distribution<>(0, log_sigma)(random_number_generator);
        const double exp_rnd = std::exp(rnd);
        x *= exp_rnd;
        product_of_ratios *= exp_rnd;
      }

    return {new_sample, product_of_ratios};
  }

} // namespace ProposalGenerator


// The last main class is the Metropolis-Hastings sampler itself.
// If you understand the algorithm behind this method, then
// the following implementation should not be too difficult
// to read. The only thing of relevance is that descriptions
// of the algorithm typically ask whether the *ratio* of two
// probabilities (the "posterior" probabilities of the current
// and the previous samples, where the "posterior" is the product of the
// likelihood and the prior probability) is larger or smaller than a
// randomly drawn number. But because our interfaces return the
// *logarithms* of these probabilities, we now need to take
// the ratio of appropriate exponentials -- which is made numerically
// more stable by considering the exponential of the difference of
// the log probabilities. The only other slight complication is that
// we need to multiply this ratio by the ratio of proposal probabilities
// since we use a non-symmetric proposal distribution.
//
// Finally, we note that the output is generated with 7 digits of
// accuracy. (The C++ default is 6 digits.) We do this because,
// as shown in the paper, we can determine the mean value of the
// probability distribution we are sampling here to at least six
// digits of accuracy, and do not want to be limited by the precision
// of the output.
namespace Sampler
{
  class MetropolisHastings
  {
  public:
    MetropolisHastings(ForwardSimulator::Interface &       simulator,
                       const LogLikelihood::Interface &    likelihood,
                       const LogPrior::Interface &         prior,
                       const ProposalGenerator::Interface &proposal_generator,
                       const unsigned int                  random_seed,
                       const std::string &                 dataset_name);

    void sample(const Vector<double> &starting_guess,
                const unsigned int    n_samples);

  private:
    ForwardSimulator::Interface &       simulator;
    const LogLikelihood::Interface &    likelihood;
    const LogPrior::Interface &         prior;
    const ProposalGenerator::Interface &proposal_generator;

    std::mt19937 random_number_generator;

    unsigned int sample_number;
    unsigned int accepted_sample_number;

    std::ofstream output_file;

    void write_sample(const Vector<double> &current_sample,
                      const double          current_log_likelihood);
  };


  MetropolisHastings::MetropolisHastings(
    ForwardSimulator::Interface &       simulator,
    const LogLikelihood::Interface &    likelihood,
    const LogPrior::Interface &         prior,
    const ProposalGenerator::Interface &proposal_generator,
    const unsigned int                  random_seed,
    const std::string &                 dataset_name)
    : simulator(simulator)
    , likelihood(likelihood)
    , prior(prior)
    , proposal_generator(proposal_generator)
    , sample_number(0)
    , accepted_sample_number(0)
  {
    output_file.open("samples-" + dataset_name + ".txt");
    output_file.precision(7);

    random_number_generator.seed(random_seed);
  }


  void MetropolisHastings::sample(const Vector<double> &starting_guess,
                                  const unsigned int    n_samples)
  {
    std::uniform_real_distribution<> uniform_distribution(0, 1);

    Vector<double> current_sample = starting_guess;
    double         current_log_posterior =
      (likelihood.log_likelihood(simulator.evaluate(current_sample)) +
       prior.log_prior(current_sample));

    ++sample_number;
    ++accepted_sample_number;
    write_sample(current_sample, current_log_posterior);

    for (unsigned int k = 1; k < n_samples; ++k, ++sample_number)
      {
        std::pair<Vector<double>,double>
          perturbation = proposal_generator.perturb(current_sample);
        const Vector<double> trial_sample                   = std::move (perturbation.first);
        const double         perturbation_probability_ratio = perturbation.second;

        const double trial_log_posterior =
          (likelihood.log_likelihood(simulator.evaluate(trial_sample)) +
           prior.log_prior(trial_sample));

        if (std::exp(trial_log_posterior - current_log_posterior) * perturbation_probability_ratio
            >=
            uniform_distribution(random_number_generator))
          {
            current_sample        = trial_sample;
            current_log_posterior = trial_log_posterior;

            ++accepted_sample_number;
          }

        write_sample(current_sample, current_log_posterior);
      }
  }



  void MetropolisHastings::write_sample(const Vector<double> &current_sample,
                                        const double current_log_posterior)
  {
    output_file << current_log_posterior << '\t';
    output_file << accepted_sample_number << '\t';
    for (const auto &x : current_sample)
      output_file << x << ' ';
    output_file << '\n';

    output_file.flush();
  }
} // namespace Sampler


// The final function is `main()`, which simply puts all of these pieces
// together into one. The "exact solution", i.e., the "measurement values"
// we use for this program are tabulated to make it easier for other
// people to use in their own implementations of this benchmark. These
// values created using the same main class above, but using 8 mesh
// refinements and using a Q3 element -- i.e., using a much more accurate
// method than the one we use in the forward simulator for generating
// samples below (which uses 5 global mesh refinement steps and a Q1
// element). If you wanted to regenerate this set of numbers, then
// the following code snippet would do that:
// @code
//  /* Set the exact coefficient: */
//  Vector<double> exact_coefficients(64);
//  for (auto &el : exact_coefficients)
//    el = 1.;
//  exact_coefficients(9) = exact_coefficients(10) = exact_coefficients(17) =
//    exact_coefficients(18)                       = 0.1;
//  exact_coefficients(45) = exact_coefficients(46) = exact_coefficients(53) =
//    exact_coefficients(54)                        = 10.;
//
//  /* Compute the "correct" solution vector: */
//  const Vector<double> exact_solution =
//    ForwardSimulator::PoissonSolver<2>(/* global_refinements = */ 8,
//                                       /* fe_degree = */ 3,
//                                       /* prefix = */ "exact")
//      .evaluate(exact_coefficients);
// @endcode
int main()
{
  const bool testing = true;

  // Run with one thread, so as to not step on other processes
  // doing the same at the same time. It turns out that the problem
  // is also so small that running with more than one thread
  // *increases* the runtime.
  MultithreadInfo::set_thread_limit(1);

  const unsigned int random_seed  = (testing ? 1U : std::random_device()());
  const std::string  dataset_name = Utilities::to_string(random_seed, 10);

  const Vector<double> exact_solution(
    {   0.06076511762259369, 0.09601910120848481, 
        0.1238852517838584,  0.1495184117375201, 
        0.1841596127549784,  0.2174525028261122, 
        0.2250996160898698,  0.2197954769002993, 
        0.2074695698370926,  0.1889996477663016, 
        0.1632722532153726,  0.1276782480038186, 
        0.07711845915789312, 0.09601910120848552, 
        0.2000589533367983,  0.3385592591951766, 
        0.3934300024647806,  0.4040223892461541, 
        0.4122329537843092,  0.4100480091545554, 
        0.3949151637189968,  0.3697873264791232, 
        0.33401826235924,    0.2850397806663382, 
        0.2184260032478671,  0.1271121156350957, 
        0.1238852517838611,  0.3385592591951819, 
        0.7119285162766475,  0.8175712861756428, 
        0.6836254116578105,  0.5779452419831157, 
        0.5555615956136897,  0.5285181561736719, 
        0.491439702849224,   0.4409367494853282, 
        0.3730060082060772,  0.2821694983395214, 
        0.1610176733857739,  0.1495184117375257, 
        0.3934300024647929,  0.8175712861756562, 
        0.9439154625527653,  0.8015904115095128, 
        0.6859683749254024,  0.6561235366960599, 
        0.6213197201867315,  0.5753611315000049, 
        0.5140091754526823,  0.4325325506354165, 
        0.3248315148915482,  0.1834600412730086, 
        0.1841596127549917,  0.4040223892461832, 
        0.6836254116578439,  0.8015904115095396, 
        0.7870119561144977,  0.7373108331395808, 
        0.7116558878070463,  0.6745179049094283, 
        0.6235300574156917,  0.5559332704045935, 
        0.4670304994474178,  0.3499809143811, 
        0.19688263746294,    0.2174525028261253, 
        0.4122329537843404,  0.5779452419831566, 
        0.6859683749254372,  0.7373108331396063, 
        0.7458811983178246,  0.7278968022406559, 
        0.6904793535357751,  0.6369176452710288, 
        0.5677443693743215,  0.4784738764865867, 
        0.3602190632823262,  0.2031792054737325, 
        0.2250996160898818,  0.4100480091545787, 
        0.5555615956137137,  0.6561235366960938, 
        0.7116558878070715,  0.727896802240657, 
        0.7121928678670187,  0.6712187391428729, 
        0.6139157775591492,  0.5478251665295381, 
        0.4677122687599031,  0.3587654911000848, 
        0.2050734291675918,  0.2197954769003094, 
        0.3949151637190157,  0.5285181561736911, 
        0.6213197201867471,  0.6745179049094407, 
        0.690479353535786,   0.6712187391428787, 
        0.6178408289359514,  0.5453605027237883, 
        0.489575966490909,   0.4341716881061278, 
        0.3534389974779456,  0.2083227496961347, 
        0.207469569837099,   0.3697873264791366, 
        0.4914397028492412,  0.5753611315000203, 
        0.6235300574157017,  0.6369176452710497, 
        0.6139157775591579,  0.5453605027237935, 
        0.4336604929612851,  0.4109641743019312, 
        0.3881864790111245,  0.3642640090182592, 
        0.2179599909280145,  0.1889996477663011, 
        0.3340182623592461,  0.4409367494853381, 
        0.5140091754526943,  0.5559332704045969, 
        0.5677443693743304,  0.5478251665295453, 
        0.4895759664908982,  0.4109641743019171, 
        0.395727260284338,   0.3778949322004734, 
        0.3596268271857124,  0.2191250268948948, 
        0.1632722532153683,  0.2850397806663325, 
        0.373006008206081,   0.4325325506354207, 
        0.4670304994474315,  0.4784738764866023, 
        0.4677122687599041,  0.4341716881061055, 
        0.388186479011099,   0.3778949322004602, 
        0.3633362567187364,  0.3464457261905399, 
        0.2096362321365655,  0.1276782480038148, 
        0.2184260032478634,  0.2821694983395252, 
        0.3248315148915535,  0.3499809143811097, 
        0.3602190632823333,  0.3587654911000799, 
        0.3534389974779268,  0.3642640090182283, 
        0.35962682718569,    0.3464457261905295, 
        0.3260728953424643,  0.180670595355394, 
        0.07711845915789244, 0.1271121156350963, 
        0.1610176733857757,  0.1834600412730144, 
        0.1968826374629443,  0.2031792054737354, 
        0.2050734291675885,  0.2083227496961245, 
        0.2179599909279998,  0.2191250268948822, 
        0.2096362321365551,  0.1806705953553887, 
        0.1067965550010013                         });

  // Now run the forward simulator for samples:
  ForwardSimulator::PoissonSolver<2> laplace_problem(
    /* global_refinements = */ 5,
    /* fe_degree = */ 1,
    dataset_name);
  LogLikelihood::Gaussian        log_likelihood(exact_solution, 0.05);
  LogPrior::LogGaussian          log_prior(0, 2);
  ProposalGenerator::LogGaussian proposal_generator(
    random_seed, 0.09); /* so that the acceptance ratio is ~0.24 */
  Sampler::MetropolisHastings sampler(laplace_problem,
                                      log_likelihood,
                                      log_prior,
                                      proposal_generator,
                                      random_seed,
                                      dataset_name);

  Vector<double> starting_coefficients(64);
  for (auto &el : starting_coefficients)
    el = 1.;
  sampler.sample(starting_coefficients,
                 (testing ? 250 * 40 /* takes 40 seconds */
                            :
                            100000000 /* takes 6 days */
                  ));
}
