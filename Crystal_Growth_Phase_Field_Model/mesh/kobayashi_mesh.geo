// Gmsh project created on Wed Dec 16 04:57:42 2020
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {12, 0, 0, 1.0};
//+
Point(3) = {12, 3, 0, 1.0};
//+
Point(4) = {0, 3, 0, 1.0};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Curve("1") = {1};
//+
Physical Curve("2") = {2};
//+
Physical Curve("3") = {3};
//+
Physical Curve("4") = {4};
//+
Physical Surface("5") = {1};
//+
Transfinite Surface {1} = {4, 1, 2, 3};
//+
Transfinite Curve {1, 3} = 100 Using Progression 1;
//+
Transfinite Curve {4, 2} = 400 Using Progression 1;
