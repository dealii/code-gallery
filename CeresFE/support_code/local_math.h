/*
 * File:   localmath.h
 * Author: antonermakov
 *
 * Created on September 21, 2013, 7:14 PM
 */

#ifndef LOCAL_MATH_
#define LOCAL_MATH_

#define PI 3.14159265358979323846
#define TWOPI 6.283185307179586476925287
#define SECSINYEAR 3.155692608e+07
#define MAX(x,y) ((x) > (y)) ? (x) : (y)
#define MIN(x,y) ((x) < (y)) ? (x) : (y)
//#define ABS(a) ((a) < 0 ? -(a) : (a))

//double factorial(int n)
//{
// if(n == 0) {
//   return(1.);
// } else if(n == 1) {
//   return(1.);
// } else if(n == 2) {
//   return(2.);
// } else if(n == 3) {
//   return(6.);
// } else if(n == 4) {
//   return(24.);
// } else {
//   exit(-1);
// }
//}

//double fudge(int m)
//{
// if(m == 0) {
//   return(1.0);
// } else {
//   return(2.0);
// }
//}


//double sign(double x)
//{
// if(x > 0) {
//   return(1.0);
// } else if(x < 0.0) {
//   return(-1.0);
// } else {
//   return(0.0);
// }
//}

//double pv0(double x)
//{
// double ans;
//
// ans = x - TWOPI*floor(x/TWOPI);
// if(ans > TWOPI/2.0) {
//   ans = ans - TWOPI;
// }
//
//return(ans);
//}

/* assumes l=2 */
//double System::Plm(int m, double x)
//{
// if(m == 0) {
//   return(1.5*x*x - 0.5);
// } else if(m == 1) {
//   return(3.0*x*sqrt(1.0 - x*x));
// } else if(m == 2) {
//   return(3.0 - 3.0*x*x);
// } else {
//   exit(-1);
// }
//}

//double System::DP(int m, double x)
//{
// if(m == 0) {
//   return(3.0*x);
// } else if(m == 1) {
//   return((3.0 - 6.0*x*x)/sqrt(1.0 - x*x));
// } else if(m == 2) {
//   return(- 6.0*x);
// } else {
//   exit(-1);
// }
//}


#endif	/* LOCALMATH_H */

