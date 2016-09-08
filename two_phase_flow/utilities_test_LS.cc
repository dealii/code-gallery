///////////////////////////////////////////////////////
//////////////////// INITIAL PHI ////////////////////
///////////////////////////////////////////////////////
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
  double x = p[0]; double y = p[1];
  double return_value = -1.;

  if (PROBLEM==CIRCULAR_ROTATION)
    {
      double x0=0.5; double y0=0.75;
      double r0=0.15;
      double r = std::sqrt(std::pow(x-x0,2)+std::pow(y-y0,2));
      return_value = -std::tanh((r-r0)/sharpness);
    }
  else // (PROBLEM==DIAGONAL_ADVECTION)
    {
      double x0=0.25; double y0=0.25;
      double r0=0.15;
      double r=0;
      if (dim==2)
	r = std::sqrt(std::pow(x-x0,2)+std::pow(y-y0,2));	
      else
	{
	  double z0=0.25;
	  double z=p[2];
	  r = std::sqrt(std::pow(x-x0,2)+std::pow(y-y0,2)+std::pow(z-z0,2));
	}
      return_value = -std::tanh((r-r0)/sharpness);
    }
  return return_value;
}

/////////////////////////////////////////////////////
//////////////////// BOUNDARY PHI ///////////////////
/////////////////////////////////////////////////////
template <int dim>
class BoundaryPhi : public Function <dim>
{
public:
  BoundaryPhi (double t=0) 
    : 
    Function<dim>() 
  {this->set_time(t);}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
};

template <int dim>
double BoundaryPhi<dim>::value (const Point<dim> &p, const unsigned int) const
{
  return -1.0;
}

///////////////////////////////////////////////////////
//////////////////// EXACT VELOCITY ///////////////////
///////////////////////////////////////////////////////
template <int dim>
class ExactU : public Function <dim>
{
public:
  ExactU (unsigned int PROBLEM, double time=0) : Function<dim>(), PROBLEM(PROBLEM), time(time) {}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  void set_time(double time){this->time=time;};
  unsigned PROBLEM;
  double time;
};

template <int dim>
double ExactU<dim>::value (const Point<dim> &p, const unsigned int) const
{
  if (PROBLEM==CIRCULAR_ROTATION)
    return -2*numbers::PI*(p[1]-0.5);
  else // (PROBLEM==DIAGONAL_ADVECTION)
    return 1.0;
}

template <int dim>
class ExactV : public Function <dim>
{
public:
  ExactV (unsigned int PROBLEM, double time=0) : Function<dim>(), PROBLEM(PROBLEM), time(time) {}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  void set_time(double time){this->time=time;};
  unsigned int PROBLEM;
  double time;
};

template <int dim>
double ExactV<dim>::value (const Point<dim> &p, const unsigned int) const
{
  if (PROBLEM==CIRCULAR_ROTATION)
    return 2*numbers::PI*(p[0]-0.5);
  else // (PROBLEM==DIAGONAL_ADVECTION)
    return 1.0;
}

template <int dim>
class ExactW : public Function <dim>
{
public:
  ExactW (unsigned int PROBLEM, double time=0) : Function<dim>(), PROBLEM(PROBLEM), time(time) {}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  void set_time(double time){this->time=time;};
  unsigned int PROBLEM;
  double time;
};

template <int dim>
double ExactW<dim>::value (const Point<dim> &p, const unsigned int) const
{
  // PROBLEM = 3D_DIAGONAL_ADVECTION
  return 1.0;
}

