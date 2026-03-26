/////////////////////////////////////////////////////////////////////////////////
/////////////////////      Variable Instructions        /////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/********************************************************************************
      y ^
        |
        +--------------------------------+  
        |                 ^              |  
        |                 |              |  
        |                 Y2             |  
        |                 |              |  
        |                 |              |  
        |<-----X1---->|<--+--- X2 ------>|   
        |             |   |              |   
        +-----------  +   v              |    
2*delta              >+ ------           |    
        +-----------  +   ^              |    
        |                 |              |    
        |                 Y1             |    
        |                 |              |    
        |                 v              |    
        +--------------------------------+------------->  x
      O
 ********************************************************************************/


/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Geometry Set-up        ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


X1 = 0.5;         // crack length
X2 = 0.5;         // uncracked length
Y1 = 0.5;         // height from bottom to the crack opening 
Y2 = 0.5;         // height from the crack opening to the top

lc = 4e-2;      // element size

delta = 0.001;    // 1/2 of crack opening at the left edge

/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Geometry Generation    ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

nCrackEleX1 = Floor(X1/lc)+1;       // the number of element along X1
nCrackEleX2 = Floor(X2/lc)+1;       // the number of element along X2
nCrackEleY1 = Floor(Y1/lc)+1;       // the number of element along Y1
nCrackEleY2 = Floor(Y2/lc)+1;       // the number of element along Y2

L = X1+X2;    // the length of the plate
H = Y1+Y2;    // the height of the plate

Point(1) = {0, 0, 0, lc};
Point(2) = {X1, 0,  0, lc};
Point(3) = {L, 0,  0, lc};
Point(4) = {L, Y1,  0, lc};
Point(5) = {L, H,  0, lc};
Point(6) = {X1, H,  0, lc};
Point(7) = {0, H,  0, lc};
Point(8) = {0, Y1+delta,  0, lc};
Point(9) = {X1, Y1,  0, lc};
Point(10) = {0, Y1-delta,  0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 1};
Line(11) = {6, 9};
Line(12) = {9, 4};
Line(13) = {2, 9};


Curve Loop(1) = {1, 13, 9, 10};
Plane Surface(1) = {1};
Transfinite Surface{1} = {1, 2, 9, 10};

Curve Loop(2) = {2, 3, -12, -13};
Plane Surface(2) = {2};
Transfinite Surface{2} = {2, 3, 4, 9};

Curve Loop(3) = {4, 5, 11, 12};
Plane Surface(3) = {3};
Transfinite Surface{3} = {4, 5, 6, 9};

Curve Loop(4) = {6, 7, 8, -11};
Plane Surface(4) = {4};
Transfinite Surface{4} = {6, 7, 8, 9};


Transfinite Curve {1, 8, 9, 6} = nCrackEleX1 Using Progression 1;
Transfinite Curve {2, 5,   12} = nCrackEleX2 Using Progression 1;
Transfinite Curve {3, 10,  13} = nCrackEleY1 Using Progression 1;
Transfinite Curve {4, 7, 11} = nCrackEleY2 Using Progression 1;


Recombine Surface{1:4};


/////////////////////////////////////////////////////////////////////////////////
////////////////////////      Boundary conditions    ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//Physical Curve("upper_edge", 14)  = {6, 5};
//Physical Curve("left_edge", 15)   = {7, 10};
//Physical Curve("crack_edge", 16)  = {8, 9};
//Physical Curve("bottom_edge", 17) = {1, 2};
//Physical Curve("right_edge", 18)  = {3, 4};
//Physical Curve(20) = {11, 13, 12};

/////////////////////////////////////////////////////////////////////////////////
////////////////////////       Material set-up       ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


Physical Surface(1) = {4, 3, 2, 1};
