/////////////////////////////////////////////////////
//////////////////// INITIAL PHI ////////////////////
/////////////////////////////////////////////////////
template <int dim>
class InitialPhi : public Function <dim>
{
public:
  InitialPhi (unsigned int PROBLEM, double sharpness=0.005) : Function<dim>(),
    sharpness(sharpness),
    PROBLEM(PROBLEM) {}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  double sharpness;
  unsigned int PROBLEM;
};
template <int dim>
double InitialPhi<dim>::value (const Point<dim> &p,
                               const unsigned int) const
{
  double x = p[0];
  double y = p[1];
  double pi=numbers::PI;

  if (PROBLEM==FILLING_TANK)
    return 0.5*(-std::tanh((y-0.3)/sharpness)*std::tanh((y-0.35)/sharpness)+1)
           *(-std::tanh((x-0.02)/sharpness)+1)-1;
  else if (PROBLEM==BREAKING_DAM)
    return 0.5*(-std::tanh((x-0.35)/sharpness)*std::tanh((x-0.65)/sharpness)+1)
           *(1-std::tanh((y-0.35)/sharpness))-1;
  else if (PROBLEM==FALLING_DROP)
    {
      double x0=0.15;
      double y0=0.75;
      double r0=0.1;
      double r = std::sqrt(std::pow(x-x0,2)+std::pow(y-y0,2));
      return 1-(std::tanh((r-r0)/sharpness)+std::tanh((y-0.3)/sharpness));
    }
  else if (PROBLEM==SMALL_WAVE_PERTURBATION)
    {
      double wave = 0.1*std::sin(pi*x)+0.25;
      return -std::tanh((y-wave)/sharpness);
    }
  else
    {
      std::cout << "Error in type of PROBLEM" << std::endl;
      abort();
    }
}

///////////////////////////////////////////////////////
//////////////////// FORCE TERMS ///// ////////////////
///////////////////////////////////////////////////////
template <int dim>
class ForceTerms : public ConstantFunction <dim>
{
public:
  ForceTerms (const std::vector<double> values) : ConstantFunction<dim>(values) {}
};

/////////////////////////////////////////////////////
//////////////////// BOUNDARY PHI ///////////////////
/////////////////////////////////////////////////////
template <int dim>
class BoundaryPhi : public ConstantFunction <dim>
{
public:
  BoundaryPhi (const double value, const unsigned int n_components=1) : ConstantFunction<dim>(value,n_components) {}
};

//////////////////////////////////////////////////////////
//////////////////// BOUNDARY VELOCITY ///////////////////
//////////////////////////////////////////////////////////
template <int dim>
class BoundaryU : public Function <dim>
{
public:
  BoundaryU (unsigned int PROBLEM, double t=0) : Function<dim>(), PROBLEM(PROBLEM) {this->set_time(t);}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  unsigned PROBLEM;
};
template <int dim>
double BoundaryU<dim>::value (const Point<dim> &p, const unsigned int) const
{
  //////////////////////
  // FILLING THE TANK //
  //////////////////////
  // boundary for filling the tank (inlet)
  double x = p[0];
  double y = p[1];

  if (PROBLEM==FILLING_TANK)
    {
      if (x==0 && y>=0.3 && y<=0.35)
        return 0.25;
      else
        return 0.0;
    }
  else
    {
      std::cout << "Error in PROBLEM definition" << std::endl;
      abort();
    }
}

template <int dim>
class BoundaryV : public Function <dim>
{
public:
  BoundaryV (unsigned int PROBLEM, double t=0) : Function<dim>(), PROBLEM(PROBLEM) {this->set_time(t);}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  unsigned int PROBLEM;
};
template <int dim>
double BoundaryV<dim>::value (const Point<dim> &p, const unsigned int) const
{
  // boundary for filling the tank (outlet)
  double x = p[0];
  double y = p[1];
  double return_value = 0;

  if (PROBLEM==FILLING_TANK)
    {
      if (y==0.4 && x>=0.3 && x<=0.35)
        return_value = 0.25;
    }
  return return_value;
}

///////////////////////////////////////////////////////
/////////////////// POST-PROCESSING ///////////////////
///////////////////////////////////////////////////////
template <int dim>
class Postprocessor : public DataPostprocessorScalar <dim>
{
public:
  Postprocessor(double eps, double rho_air, double rho_fluid)
    :
    DataPostprocessorScalar<dim>("Density",update_values)
  {
    this->eps=eps;
    this->rho_air=rho_air;
    this->rho_fluid=rho_fluid;
  }

  virtual
  void
  evaluate_scalar_field (const DataPostprocessorInputs::Scalar<dim> &input_data,
                         std::vector<Vector<double> >               &computed_quantities) const;

  double eps;
  double rho_air;
  double rho_fluid;
};


template <int dim>
void
Postprocessor<dim>::
evaluate_scalar_field (const DataPostprocessorInputs::Scalar<dim> &input_data,
                       std::vector<Vector<double> >               &computed_quantities) const
{
  const unsigned int n_quadrature_points = input_data.solution_values.size();
  for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      double H;
      double rho_value;
      double phi_value=input_data.solution_values[q];
      if (phi_value > eps)
        H=1;
      else if (phi_value < -eps)
        H=-1;
      else
        H=phi_value/eps;
      rho_value = rho_fluid*(1+H)/2. + rho_air*(1-H)/2.;
      computed_quantities[q] = rho_value;
    }
}

