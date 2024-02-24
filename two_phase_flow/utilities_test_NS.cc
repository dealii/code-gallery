/* -----------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2016 Manuel Quezada de Luna
 *
 * This file is part of the deal.II code gallery.
 *
 * -----------------------------------------------------------------------------
 */

///////////////////////////////////////////////////////
//////////// EXACT SOLUTION RHO TO TEST NS ////////////
///////////////////////////////////////////////////////
template <int dim>
class RhoFunction : public Function <dim>
{
public:
  RhoFunction (double t=0) : Function<dim>() {this->set_time(t);}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
};
template <int dim>
double RhoFunction<dim>::value (const Point<dim> &p,
                                const unsigned int) const
{
  double t = this->get_time();
  double return_value = 0;
  if (dim==2)
    return_value = std::pow(std::sin(p[0]+p[1]+t),2)+1;
  else //dim=3
    return_value = std::pow(std::sin(p[0]+p[1]+p[2]+t),2)+1;
  return return_value;
}

template <int dim>
class NuFunction : public Function <dim>
{
public:
  NuFunction (double t=0) : Function<dim>() {this->set_time(t);}
  virtual double value (const Point<dim>   &p, const unsigned int component=0) const;
};
template <int dim>
double NuFunction<dim>::value (const Point<dim> &, const unsigned int) const
{
  return 1.;
}

//////////////////////////////////////////////////////////////////
/////////////////// EXACT SOLUTION U to TEST NS //////////////////
//////////////////////////////////////////////////////////////////
template <int dim>
class ExactSolution_and_BC_U : public Function <dim>
{
public:
  ExactSolution_and_BC_U (double t=0, int field=0)
    :
    Function<dim>(),
    field(field)
  {
    this->set_time(t);
  }
  virtual double value (const Point<dim> &p, const unsigned int  component=1) const;
  virtual Tensor<1,dim> gradient (const Point<dim> &p, const unsigned int component=1) const;
  virtual void set_field(int field) {this->field=field;}
  int field;
  unsigned int type_simulation;
};
template <int dim>
double ExactSolution_and_BC_U<dim>::value (const Point<dim> &p,
                                           const unsigned int) const
{
  double t = this->get_time();
  double return_value = 0;
  double Pi = numbers::PI;
  double x = p[0];
  double y = p[1];
  double z = 0;

  if (dim == 2)
    if (field == 0)
      return_value = std::sin(x)*std::sin(y+t);
    else
      return_value = std::cos(x)*std::cos(y+t);
  else //dim=3
    {
      z = p[2];
      if (field == 0)
        return_value = std::cos(t)*std::cos(Pi*y)*std::cos(Pi*z)*std::sin(Pi*x);
      else if (field == 1)
        return_value = std::cos(t)*std::cos(Pi*x)*std::cos(Pi*z)*std::sin(Pi*y);
      else
        return_value = -2*std::cos(t)*std::cos(Pi*x)*std::cos(Pi*y)*std::sin(Pi*z);
    }
  return return_value;
}
template <int dim>
Tensor<1,dim> ExactSolution_and_BC_U<dim>::gradient (const Point<dim> &p,
                                                     const unsigned int) const
{
  // THIS IS USED JUST FOR TESTING NS
  Tensor<1,dim> return_value;
  double t = this->get_time();
  double Pi = numbers::PI;
  double x = p[0];
  double y = p[1];
  double z = 0;
  if (dim == 2)
    if (field == 0)
      {
        return_value[0] = std::cos(x)*std::sin(y+t);
        return_value[1] = std::sin(x)*std::cos(y+t);
      }
    else
      {
        return_value[0] = -std::sin(x)*std::cos(y+t);
        return_value[1] = -std::cos(x)*std::sin(y+t);
      }
  else //dim=3
    {
      z=p[2];
      if (field == 0)
        {
          return_value[0] = Pi*std::cos(t)*std::cos(Pi*x)*std::cos(Pi*y)*std::cos(Pi*z);
          return_value[1] = -(Pi*std::cos(t)*std::cos(Pi*z)*std::sin(Pi*x)*std::sin(Pi*y));
          return_value[2] = -(Pi*std::cos(t)*std::cos(Pi*y)*std::sin(Pi*x)*std::sin(Pi*z));
        }
      else if (field == 1)
        {
          return_value[0] = -(Pi*std::cos(t)*std::cos(Pi*z)*std::sin(Pi*x)*std::sin(Pi*y));
          return_value[1] = Pi*std::cos(t)*std::cos(Pi*x)*std::cos(Pi*y)*std::cos(Pi*z);
          return_value[2] = -(Pi*std::cos(t)*std::cos(Pi*x)*std::sin(Pi*y)*std::sin(Pi*z));
        }
      else
        {
          return_value[0] = 2*Pi*std::cos(t)*std::cos(Pi*y)*std::sin(Pi*x)*std::sin(Pi*z);
          return_value[1] = 2*Pi*std::cos(t)*std::cos(Pi*x)*std::sin(Pi*y)*std::sin(Pi*z);
          return_value[2] = -2*Pi*std::cos(t)*std::cos(Pi*x)*std::cos(Pi*y)*std::cos(Pi*z);
        }
    }
  return return_value;
}

///////////////////////////////////////////////////////
/////////// EXACT SOLUTION FOR p TO TEST NS ///////////
///////////////////////////////////////////////////////
template <int dim>
class ExactSolution_p : public Function <dim>
{
public:
  ExactSolution_p (double t=0) : Function<dim>() {this->set_time(t);}
  virtual double value (const Point<dim> &p, const unsigned int  component=0) const;
  virtual Tensor<1,dim> gradient (const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
double ExactSolution_p<dim>::value (const Point<dim> &p, const unsigned int) const
{
  double t = this->get_time();
  double return_value = 0;
  if (dim == 2)
    return_value = std::cos(p[0])*std::sin(p[1]+t);
  else //dim=3
    return_value = std::sin(p[0]+p[1]+p[2]+t);
  return return_value;
}

template <int dim>
Tensor<1,dim> ExactSolution_p<dim>::gradient (const Point<dim> &p, const unsigned int) const
{
  Tensor<1,dim> return_value;
  double t = this->get_time();
  if (dim == 2)
    {
      return_value[0] = -std::sin(p[0])*std::sin(p[1]+t);
      return_value[1] = std::cos(p[0])*std::cos(p[1]+t);
    }
  else //dim=3
    {
      return_value[0] = std::cos(t+p[0]+p[1]+p[2]);
      return_value[1] = std::cos(t+p[0]+p[1]+p[2]);
      return_value[2] = std::cos(t+p[0]+p[1]+p[2]);
    }
  return return_value;
}

//////////////////////////////////////////////////////////////////
//////////////////// FORCE TERMS to TEST NS //////////////////////
//////////////////////////////////////////////////////////////////
template <int dim>
class ForceTerms : public Function <dim>
{
public:
  ForceTerms (double t=0)
    :
    Function<dim>()
  {
    this->set_time(t);
    nu = 1.;
  }
  virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;
  double nu;
};

template <int dim>
void ForceTerms<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
{
  double x = p[0];
  double y = p[1];
  double z = 0;
  double t = this->get_time();
  double Pi = numbers::PI;

  if (dim == 2)
    {
      // force in x
      values[0] = std::cos(t+y)*std::sin(x)*(1+std::pow(std::sin(t+x+y),2)) // time derivative
                  +2*nu*std::sin(x)*std::sin(t+y) // viscosity
                  +std::cos(x)*std::sin(x)*(1+std::pow(std::sin(t+x+y),2)) // non-linearity
                  -std::sin(x)*std::sin(y+t); // pressure
      // force in y
      values[1] = -(std::cos(x)*std::sin(t+y)*(1+std::pow(std::sin(t+x+y),2))) // time derivative
                  +2*nu*std::cos(x)*std::cos(t+y) // viscosity
                  -(std::sin(2*(t+y))*(1+std::pow(std::sin(t+x+y),2)))/2. // non-linearity
                  +std::cos(x)*std::cos(y+t); // pressure
    }
  else //3D
    {
      z = p[2];
      // force in x
      values[0]=
        -(std::cos(Pi*y)*std::cos(Pi*z)*std::sin(t)*std::sin(Pi*x)*(1+std::pow(std::sin(t+x+y+z),2))) //time der.
        +3*std::pow(Pi,2)*std::cos(t)*std::cos(Pi*y)*std::cos(Pi*z)*std::sin(Pi*x) //viscosity
        -(Pi*std::pow(std::cos(t),2)*(-3+std::cos(2*(t+x+y+z)))*std::sin(2*Pi*x)*(std::cos(2*Pi*y)+std::pow(std::sin(Pi*z),2)))/4. //NL
        +std::cos(t+x+y+z); // pressure
      values[1]=
        -(std::cos(Pi*x)*std::cos(Pi*z)*std::sin(t)*std::sin(Pi*y)*(1+std::pow(std::sin(t+x+y+z),2))) //time der
        +3*std::pow(Pi,2)*std::cos(t)*std::cos(Pi*x)*std::cos(Pi*z)*std::sin(Pi*y) //viscosity
        -(Pi*std::pow(std::cos(t),2)*(-3+std::cos(2*(t+x+y+z)))*std::sin(2*Pi*y)*(std::cos(2*Pi*x)+std::pow(std::sin(Pi*z),2)))/4. //NL
        +std::cos(t+x+y+z); // pressure
      values[2]=
        2*std::cos(Pi*x)*std::cos(Pi*y)*std::sin(t)*std::sin(Pi*z)*(1+std::pow(std::sin(t+x+y+z),2)) //time der
        -6*std::pow(Pi,2)*std::cos(t)*std::cos(Pi*x)*std::cos(Pi*y)*std::sin(Pi*z) //viscosity
        -(Pi*std::pow(std::cos(t),2)*(2+std::cos(2*Pi*x)+std::cos(2*Pi*y))*(-3+std::cos(2*(t+x+y+z)))*std::sin(2*Pi*z))/4. //NL
        +std::cos(t+x+y+z); // pressure
    }
}
