%% -----------------------------------------------------------------------------
%%
%% SPDX-License-Identifier: LGPL-2.1-or-later
%% Copyright (C) 2022 by Wolfgang Bangerth
%%
%% This file is part of the deal.II code gallery.
%%
%% -----------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% compute statistics on data set %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS: 
%data = tensor of theta samples from each lag time and chain
%theta_means = means of theta over each independent chain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OUTPUTS:
%theta_mean = overall mean of chains
%covars = covariance matrices of each independent chain
%autocovar = mean of autocovariance matrix over all the chains
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [theta_mean,covars,autocovar] = get_statistics(data,theta_means);

%compute overall mean of data, and get size of data matrix
theta_mean = mean(theta_means,3);
[~,~,L,N] = size(data);

%initialize covariance matrices and mean autocovariance matrix
covars = zeros(64,64,N);
autocovar = zeros(64,64,2*L-1);

%compute covariance matrices and mean autocovariance matrix
for n=1:N   %loop over independent Markov chains
    
    %get data from chain n
    data_ = reshape(permute(data(:,:,:,n),[3 2 1]),[L 64]);

    %compute autocovariance matrix of chain n
    mat = xcov(data_,'unbiased');

    %store covariance matrix of chain n
    covars(:,:,n) = reshape(mat(L,:),64,64);

    %update mean autocovariance matrix
    autocovar = autocovar + reshape(mat',[64 64 2*L-1]);
end

%compute mean of autocovariance matrix
autocovar = autocovar(1:64,1:64,L:2*L-1)/N;
