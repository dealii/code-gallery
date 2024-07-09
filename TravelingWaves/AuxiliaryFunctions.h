#ifndef AUXILIARY_FUNCTIONS
#define AUXILIARY_FUNCTIONS

#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <cstring>

// Comparison of numbers with a given tolerance.
template <typename T>
bool isapprox(const T &a, const T &b, const double tol = 1e-10)
{
  return (std::abs( a - b ) < tol);
}

// Fill the std::vector with the values from the range [interval_begin, interval_end].
template <typename T>
void linspace(T interval_begin, T interval_end, std::vector<T> &arr) 
{
  const size_t SIZE = arr.size();
  const T step = (interval_end - interval_begin) / static_cast<T>(SIZE - 1);
  for (size_t i = 0; i < SIZE; ++i) 
  {
    arr[i] = interval_begin + i * step;
  }
}

// Check the file existence.
inline bool file_exists(const std::string &filename) 
{
  std::ifstream f(filename.c_str());
  return f.good();
}

#endif
