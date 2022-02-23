%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Run MCMC sampler to estimate posterior distribution %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define number of chains, chain length, and lag time
N = input('number of independent Markov chains: ');
N_L = input('length of each Markov chain: ');
lag = input('lag time for measurements: ');
workers = input('number of parallel workers: ');
L = N_L/lag;

%open Matlab parallel pool
parpool(workers)

%load precomputations
load precomputations.mat

%define lag time and data matrix
data = zeros(8,8,L,N);   %data matrix of samples at lag times
theta_means = zeros(8,8,N);   %overall mean of theta

tic

parfor n=1:N

    %set initial theta, theta mean, and z values of chain
    theta = theta0;
    theta_mean = 0;
    z = z0;

    for m=1:L

        for l=1:lag

            %define proposal, theta_tilde
            xi = normrnd(0,sig_prop,[8 8]);
            theta_tilde = theta.*exp(xi);
        
            %compute new z values
            z_tilde = forward_solver_(theta_tilde);
        
            %compute posterior log probability of theta_tilde
            log_pi_tilde = log_probability_(theta_tilde,z_tilde);
            log_pi = log_probability_(theta,z);
        
            %compute acceptance probability; accept proposal appropriately
            accept = exp(log_pi_tilde-log_pi) ... 
                    *prod(theta_tilde./theta,'all');
            if rand<accept
                theta = theta_tilde;   %accept new theta values
                z = z_tilde;           %record associated measurements
            end
            
            %update mean of theta
            theta_mean = theta_mean + theta;
        
        end
    
        %update data matrix
        data(:,:,m,n) = theta;

    end

    %update theta means
    theta_means(:,:,n) = theta_mean/N_L;
    
end

toc

%compute statistics on data set
[theta_mean,covars,autocovar] = get_statistics(data,theta_means);

%save data to Matlab workspace, labeled by N and N_L
save (['data_N_' num2str(N) '_N_L_ ' num2str(N_L) '.mat'])
