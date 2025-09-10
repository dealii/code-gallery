/////////////////////////////////////////////////////////////////////////////////
/////////////////////      Variable Instructions        /////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/********************************************************************************
      y ^
        |
        +---------------------------------+ - - - - - - - - - - - 
        |                                 |               ^
        |                                 |               |
        |                                 |               |
        |                                 |               |
        |                                 |               |
        |                                 |               |
        |                |                |               |
        |<----- X1 ----->|<----- X2 ----->|               |
        |                |                |               |
        |                +- - - - - - - - | - - - -       H            
        |               / \               |    ^          |
        |              /   \              |    |          |
        |             /     \             |    |          |
        |            /       \            |    Y1         |
        |           /         \           |    |          |
        |          /           \          |    |          |
        |         /             \         |    |          |
        |        /               \        |    v          v
        +-------+                 +-------+---------------------->  x
      O         |<-- 2 * delta -->|
 ********************************************************************************/


/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Geometry Set-up        ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


X1 = 4;         // the x-coordinate of crack tip
X2 = 4;         // the distance between the crack tip and the right edge
Y1 = 0.4;       // crack length
H = 2;          // the height of the plate

lc = 10e-2;      // element size

delta = 0.1;    // 1/2 of crack opening at the bottom edge

/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Geometry Generation    ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

L = X1+X2;    // the length of the plate
Y2 = H-Y1;    // the distance between the crack tip and the top edge



nCrackEleX1 = Floor(X1/lc)+1;       // the number of element along X1
nCrackEleX2 = Floor(X2/lc)+1;       // the number of element along X2
nCrackEleY1 = Floor(Y1/lc)+1;       // the number of element along Y1
nCrackEleY2 = Floor(Y2/lc)+1;       // the number of element along Y2

Point(1) = {0, 0, 0, lc};
Point(2) = {X1-delta, 0,  0, lc};
Point(3) = {X1+delta, 0,  0, lc};
Point(4) = {L, 0,  0, lc};
Point(5) = {L, Y1, 0, lc};
Point(6) = {L, H,  0, lc};
Point(7) = {X1, H,  0, lc};
Point(8) = {0,  H,  0, lc};
Point(9) = {0, Y1,  0, lc};
Point(10) = {X1, Y1,  0, lc};

Line(1) = {1, 2};
Line(2) = {2, 10};
Line(3) = {10, 3};
Line(4) = {3, 4};
Line(5) = {4, 5};
Line(6) = {5, 6};
Line(7) = {6, 7};
Line(8) = {7, 8};
Line(9) = {8, 9};
Line(10) = {9, 1};
Line(11) = {9, 10};
Line(12) = {10, 5};
Line(13) = {10, 7};


Curve Loop(1) = {1, 2, -11, 10};
Plane Surface(1) = {1};
Transfinite Surface{1} = {1, 2, 10, 9};

Curve Loop(2) = {3, 4, 5, -12};
Plane Surface(2) = {2};
Transfinite Surface{2} = {10, 3, 4, 5};

Curve Loop(3) = {12, 6, 7, -13};
Plane Surface(3) = {3};
Transfinite Surface{3} = {10, 5, 6, 7};

Curve Loop(4) = {13, 8, 9, 11};
Plane Surface(4) = {4};
Transfinite Surface{4} = {10, 7, 8, 9};


Transfinite Curve {1, 8, 11} = nCrackEleX1 Using Progression 1;
Transfinite Curve {4, 7, 12} = nCrackEleX2 Using Progression 1;
Transfinite Curve {2, 3, 5, 10} = nCrackEleY1 Using Progression 1;
Transfinite Curve {6, 9, 13} = nCrackEleY2 Using Progression 1;


Recombine Surface{1:4};


/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Boundary conditions    ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/* */
//Physical Curve("upper_edge", 14)  = {7, 8};
//Physical Curve("left_edge", 15)   = {9, 10};
//Physical Curve("crack_edge", 16)  = {2, 3};
//Physical Curve("bottom_edge", 17) = {1, 4};
//Physical Curve("right_edge", 18)  = {5, 6};
/* */
/////////////////////////////////////////////////////////////////////////////////
////////////////////////       Material set-up       ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/* */
Physical Surface(1) = {1:4};
/* */
