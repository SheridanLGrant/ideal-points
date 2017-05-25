// ideal point model identified with fixed legislator ideal points
data {
  // number of items
  int K;
  // number of individuals
  int N;
  // observed votes
  int<lower = 0, upper = N * K> Y_obs;
  int y_idx_leg[Y_obs];
  int y_idx_vote[Y_obs];
  int y[Y_obs];
  // ideal points
  // for identification, some ideal points are fixed
  int<lower = 0, upper = N> N_obs;
  int<lower = 0, upper = N> N_param;
  int<lower = 1, upper = N> theta_obs_idx[N_obs];
  int theta_obs[N_obs];
  int<lower = 1, upper = N> theta_param_idx[N_param];
  // priors
  vector[K] alpha_loc;
  vector<lower = 0.>[K] alpha_scale;
  vector[K] lambda_loc;
  vector<lower = 0.>[K] lambda_scale;
}
parameters {
  // item difficulties
  vector[K] alpha;
  // item cutpoints
  vector[K] lambda;
  // unknown ideal points
  vector[N_param] theta_param;
  // prior on ideal points
  real xi;
  real<lower = 0.> tau;
}
transformed parameters {
  // create theta from observed and parameter ideal points
  vector[N] theta;
  vector[Y_obs] mu;
  for (k in 1:N_param) {
    theta[theta_param_idx[k]] = theta_param[k];
  }
  for (k in 1:N_obs) {
    theta[theta_obs_idx[k]] = theta_obs[k];
  }
  for (i in 1:Y_obs) {
    mu[i] = alpha[y_idx_vote[i]] + lambda[y_idx_vote[i]] * theta[y_idx_leg[i]];
  }
}
model {
  alpha ~ normal(alpha_loc, alpha_scale);
  lambda ~ normal(lambda_loc, lambda_scale);
  theta_param ~ normal(xi, tau);
  xi ~ normal(0., 1.);
  tau ~ normal(0., 1.);
  y ~ binomial_logit(1, mu);
}
generated quantities {
  vector[Y_obs] log_lik;
  // int y_rep[Y_obs];
  for (i in 1:Y_obs) {
    log_lik[i] = binomial_logit_lpmf(y[i] | 1, mu[i]);
    // y_rep[i] = binomial_rng(1, inv_logit(mu[i]));
  }
}
