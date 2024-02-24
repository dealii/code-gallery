%% -----------------------------------------------------------------------------
%%
%% SPDX-License-Identifier: LGPL-2.1-or-later
%% Copyright (C) 2022 by Wolfgang Bangerth
%%
%% This file is part of the deal.II code gallery.
%%
%% -----------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% compute log probability, log pi %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS: 
%theta = current 8x8 parameter matrix
%z = current vector of measurements
%z_hat = vector of "exact" measurements
%sig = standard deviation parameter in likelihood
%sig_pr = standard deviation parameter in prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OUTPUTS:
%log_pi = logarithm of posterior probability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function log_pi = log_probability(theta,z,z_hat,sig,sig_pr)

%compute log likelihood
log_L = -sum((z-z_hat).^2)/(2*sig^2);

%compute log prior
log_pi_pr = -sum(log(theta).^2,'all')/(2*sig_pr^2);

%compute log posterior
log_pi = log_L+log_pi_pr;
