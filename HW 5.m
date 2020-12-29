% Part I
% 2
close;
clear all;
clc;

lambda = 100; %poisson rate
mu =.1; %log normal mean
sigma = .4; %log normal std dev

mean_N = lambda;
mean_X = exp(mu+sigma^2/2);
var_N = lambda;
var_X = exp(2*mu+sigma^2)*(exp(sigma^2)-1);
mean_X_cube=exp(3*mu+9*sigma^2/2);
mean_X_sq = exp(2*mu+2*sigma^2);
mean_SN = mean_N*mean_X;
var_SN = mean_N*var_X + (mean_X)^2*var_N;
skew_SN = (mean_X_cube)*(lambda*(mean_X_sq)^3)^(-1/2);

gamma_alpha = 4*(skew_SN)^(-2);
gamma_beta = (gamma_alpha/(lambda*mean_X_sq))^(1/2);
gamma_k = lambda*mean_X - gamma_alpha/gamma_beta;

norm_approx = @(x) normcdf(x,mean_SN,(var_SN)^(1/2));
gamma_approx = @(x) gamcdf(x-gamma_k,gamma_alpha,(gamma_beta)^(-1));

num_values = 100000;

poisson_rnd=poissrnd(lambda,[num_values,1]);
comp_poisson_rnd=zeros(num_values,1);

for j=1:num_values
    if poisson_rnd(j,1)~= 0
        comp_poisson_rnd(j,1)=sum(lognrnd(mu,sigma,[poisson_rnd(j,1),1]));
    end
end

sort_comp_poisson_rnd = sort(comp_poisson_rnd,'ascend');
alpha_low = .95;
alpha_high = .99999;
low_val = sort_comp_poisson_rnd(ceil(num_values*alpha_low),1);
high_val = sort_comp_poisson_rnd(ceil(num_values*alpha_high),1);

num_plot_pts = 1000;
cdf_values = low_val:(high_val-low_val)/num_plot_pts:high_val;

norm_cdf_tail = 1-norm_approx(cdf_values(1,:));
gamma_cdf_tail = 1-gamma_approx(cdf_values(1,:));

[emp_cdf emp_cdf_x]=ecdf(comp_poisson_rnd);
emp_cdf_tail=zeros(1,num_plot_pts+1);
for j=1:num_plot_pts+1
    emp_cdf_tail(1,j)=1-max(emp_cdf(emp_cdf_x<=cdf_values(1,j)));
end

loglog(cdf_values,norm_cdf_tail,cdf_values,gamma_cdf_tail,cdf_values,emp_cdf_tail);
legend('normal','gamma','empirical');

% 3
close;
clear all;
clc;

mu= 0;
sigma= .4/(252^(.5));
nu = .002;
zeta = .0008;
stock_px = 59;
shares = 100;
k_mult = 3;
num_trials = 100000;
alpha = .99;

losses = zeros(num_trials,1);
normal_samples = randn(num_trials,2);
for m=1:num_trials
    losses(m,1) = -shares*stock_px*(exp(mu + sigma*normal_samples(m,1))...
        *(1-(1/2)*(nu+zeta*normal_samples(m,2)))-1);
end
sort_losses(:,1) = sort(losses(:,1),'ascend');
sim_liq_var = sort_losses(ceil(num_trials*alpha),1);

th_var = shares*stock_px*(1-exp(mu + sigma*norminv(1-alpha,0,1)));
th_lc = (1/2)*shares*stock_px*(nu+k_mult*zeta);

fprintf('Confidence: %10.3f\n', alpha);
fprintf('Simulated Liquidty VaR: %10.2f\n', sim_liq_var);
fprintf('Theoretical VaR: %10.2f\n', th_var);
fprintf('Simulated Liquidity Cost: %10.2f\n', sim_liq_var-th_var);
fprintf('Simulated Percentage Liquidity VaR Increase: %10.2f\n', 100*((sim_liq_var/th_var)-1));
fprintf('-----\n');
fprintf('Industry Approximate Liquidity VaR: %10.2f\n', th_var + th_lc);
fprintf('Theoretical VaR: %10.2f\n', th_var);
fprintf('Industry Approximate Liquidity Cost: %10.2f\n', th_lc);
fprintf('Industry Approximate Percentage Liquidity VaR Increase: %10.2f\n', 100*(th_lc/th_var));

% Part II
clear;
close all;
clc;

data_file = 'AAPL_Data.csv';
return_data = csvread(data_file);

N = length(return_data(:,1))-1;
price_data = return_data(:,2);

for n=1:N
    log_return_data(n,1) = log(price_data(n+1,1)) - log(price_data(n,1));
end
mean_log_return=mean(log_return_data(:,1));
var_log_return=var(log_return_data(:,1));
port_value = 1000000;
port_loss = -port_value*(exp(log_return_data)-1);
sort_port_loss = sort(port_loss(:,1),'ascend');
alpha = .97;
emp_var = sort_port_loss(ceil(N*alpha),1);
sample_var = port_value*(1-exp(mean_log_return+(var_log_return)^(.5)*norminv(1-alpha,0,1)));
beta = .02;
low_chi_val = chi2inv(beta/2,N-1);
high_chi_val = chi2inv(1-beta/2,N-1);
low_sigma_val = ((N-1)*var_log_return/high_chi_val)^(.5);
high_sigma_val = ((N-1)*var_log_return/low_chi_val)^(.5);
low_VaR_val = port_value*(1-exp(mean_log_return+low_sigma_val*norminv(1-alpha,0,1)));
high_VaR_val = port_value*(1-exp(mean_log_return + high_sigma_val*norminv(1-alpha,0,1)));

M=125000;
VaR_est=zeros(M,1);

for m=1:M
    chi_square_value=chi2rnd(N-1);
    sigma_est = ((N-1)*var_log_return/chi_square_value)^(.5);
    mu_est = mean_log_return;
    VaR_est(m,1) = port_value*(1-exp(mu_est+sigma_est*norminv(1-alpha,0,1)));
end

VaR_est_sim_mean = mean(VaR_est(:,1));
VaR_est_sim_var = var(VaR_est(:,1));

sort_VaR_est = sort(VaR_est(:,1),'ascend');
low_VaR_val_sim = sort_VaR_est(ceil(M*beta/2),1);
high_VaR_val_sim = sort_VaR_est(ceil(M*(1-beta/2)),1);

fprintf('VaR Confidence: %0.3f\n', alpha);
fprintf('Empirical VaR: %10.2f\n', emp_var);
fprintf('VaR results assuming mean is known \n');
fprintf('VaR based on sample mean/variance: %10.2f\n', sample_var);
fprintf('Confidence Interval Beta: %0.3f\n', beta);
fprintf('VaR Confidence Interval High Point: %10.2f\n', high_VaR_val);
fprintf('VaR Confidence Interval Low Point: %10.2f\n', low_VaR_val);
fprintf('VaR results assuming mean is not known \n');
fprintf('VaR %10.2f\n', VaR_est_sim_mean);
fprintf('VaR Confidence Interval High Point: %10.2f\n', high_VaR_val_sim);
fprintf('VaR Confidence Interval Low Point: %10.2f\n', low_VaR_val_sim);
