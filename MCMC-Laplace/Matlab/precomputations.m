%% -----------------------------------------------------------------------------
%%
%% SPDX-License-Identifier: LGPL-2.1-or-later
%% Copyright (C) 2022 by Wolfgang Bangerth
%%
%% This file is part of the deal.II code gallery.
%%
%% -----------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% do all precomputations necessary for MCMC simulations %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define mesh width
h = 1/32;   

%define characteristic function of unit square
S = @(x,y) heaviside(x).*heaviside(y) ...
           .*(1-heaviside(x-h)).*(1-heaviside(y-h));

%define tent function on the domain [-h,h]x[-h,h]
phi = @(x,y) ((x+h).*(y+h).*S(x+h,y+h) + (h-x).*(h-y).*S(x,y) ... 
          + (x+h).*(h-y).*S(x+h,y) + (h-x).*(y+h).*S(x,y+h))/h^2;

%define function that converts from (i,j) to dof, and its inverse
lbl = @(i,j) 33*j+i+1;
inv_lbl = @(k) [k-1-33*floor((k-1)/33),floor((k-1)/33)];

%construct measurement matrix, M
xs = 1/14:1/14:13/14;   %measurement points
M = zeros(13,13,33^2);
for k=1:33^2
    c = inv_lbl(k);
    for i=1:13
        for j=1:13
            M(i,j,k) = phi(xs(i)-h*c(1),xs(j)-h*c(2));
        end
    end
end
M = reshape(M,[13^2 33^2]);

%construct exact coefficient matrix, theta_hat
theta_hat = ones(8,8);
theta_hat(2:3,2:3) = 0.1;
theta_hat(6:7,6:7) = 10;

%construct local overlap matrix, A_loc, and identity matrix Id
A_loc = [2/3  -1/6  -1/3  -1/6;
          -1/6  2/3  -1/6  -1/3;
          -1/3 -1/6   2/3  -1/6;
          -1/6 -1/3  -1/6   2/3];
Id = eye(33^2);

%locate boundary labels
boundaries = [lbl(0:1:32,0),lbl(0:1:32,32),lbl(0,1:1:31),lbl(32,1:1:31)];

%define RHS of FEM linear system, AU = b
b = ones(33^2,1)*10*h^2;
b(boundaries) = zeros(128,1);    %enforce boundary conditions on b

%load exact z_hat values
exact_values

%set global parameters and functions for simulation
sig = 0.05;           %likelihood standard deviation
sig_pr = 2;           %prior (log) standard deviation
sig_prop = 0.0725;    %proposal (log) standard deviation
theta0 = ones(8,8);   %initial theta values
forward_solver_ = @(theta) ... 
                  forward_solver(theta,lbl,A_loc,Id,boundaries,b,M);
log_probability_ = @(theta,z) log_probability(theta,z,z_hat,sig,sig_pr);

%find initial z values
z0 = forward_solver_(theta0);

save precomputations.mat
