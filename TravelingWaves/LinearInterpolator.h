#ifndef LINEAR_INTERPOLATOR
#define LINEAR_INTERPOLATOR

#include <cmath>
#include <algorithm>
#include <vector>

// Linear interpolation class
template <typename Number_Type>
class LinearInterpolator
{
public:
  LinearInterpolator(const std::vector<Number_Type> &ix_points, const std::vector<Number_Type> &iy_points);
  Number_Type value(const Number_Type x) const;

private:
  const std::vector<Number_Type> x_points; 	// Must be an increasing sequence, i.e. x[i] < x[i+1]
  const std::vector<Number_Type> y_points;
};

template <typename Number_Type>
LinearInterpolator<Number_Type>::LinearInterpolator(const std::vector<Number_Type> &ix_points, const std::vector<Number_Type> &iy_points)
  : x_points(ix_points)
  , y_points(iy_points)
{}

template <typename Number_Type>
Number_Type LinearInterpolator<Number_Type>::value(const Number_Type x) const
{
  Number_Type res = 0.;

  auto lower = std::lower_bound(x_points.begin(), x_points.end(), x);
  unsigned int right_index = 0;
  unsigned int left_index = 0;
  if (lower == x_points.begin())
  {
    res = y_points[0];
  }
  else if (lower == x_points.end())
  {
    res = y_points[x_points.size()-1];
  }
  else
  {
    right_index = lower - x_points.begin();
    left_index = right_index - 1;

    Number_Type y_2 = y_points[right_index];
    Number_Type y_1 = y_points[left_index];
    Number_Type x_2 = x_points[right_index];
    Number_Type x_1 = x_points[left_index];

    res = (y_2 - y_1) / (x_2 - x_1) * (x - x_1) + y_1;
  }

  return res;
}

#endif