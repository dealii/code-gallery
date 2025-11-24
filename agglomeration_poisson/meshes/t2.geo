// -----------------------------------------------------------------------------
//
//  Gmsh GEO tutorial 11
//
//  Unstructured quadrangular meshes
//
// -----------------------------------------------------------------------------

// We have seen in tutorials `t3.geo' and `t6.geo' that extruded and transfinite
// meshes can be "recombined" into quads, prisms or hexahedra by using the
// "Recombine" keyword. Unstructured meshes can be recombined in the same
// way. Let's define a simple geometry with an analytical mesh size field:

Point(1) = {-1.25, -.5, 0}; Point(2) = {1.25, -.5, 0};
Point(3) = {1.25, 1.25, 0};  Point(4) = {-1.25, 1.25, 0};

Line(1) = {1, 2}; Line(2) = {2, 3};
Line(3) = {3, 4}; Line(4) = {4, 1};

Curve Loop(4) = {1, 2, 3, 4}; Plane Surface(100) = {4};

// Field[1] = MathEval;
// Field[1].F = "0.01*(1.0+30.*(y-x*x)*(y-x*x) + (1-x)*(1-x))";
Background Field = 1;

// To generate quadrangles instead of triangles, we can simply add

Recombine Surface{100};