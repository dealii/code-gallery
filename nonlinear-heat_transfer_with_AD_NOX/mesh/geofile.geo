//+
Point(1) = {-0.5, -0.8, 0, 1.0};
Point(2) = {0.5, -0.8, 0, 1.0};
Point(3) = {0.5, 0.8, 0, 1.0};
Point(4) = {-0.5, 0.8, 0, 1.0};

Point(5) = {-0.1, -0.4, 0, 1.0};
Point(6) = {0.1, -0.4, 0, 1.0};
Point(7) = {0.1, 0.4, 0, 1.0};
Point(8) = {-0.1, 0.4, 0, 1.0};


Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};


Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1,2};

Physical Line("Left", 1) = {4};
Physical Line("Bottom", 2) = {1};
Physical Line("Right", 3) = {2};
Physical Line("Top", 4) = {3};


Physical Surface(1) = {1};





