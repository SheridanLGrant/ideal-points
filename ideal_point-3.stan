// ideal point model without identification except priors
data {
  // number of individuals
  int N;
  // number of items
  int K;
  // observed votes
  int<lower = 0, upper = N * K> Y_obs;
  int y_idx_leg[Y_obs];
  int y_idx_vote[Y_obs];
  int y[Y_obs];
  // priors
  real alpha_loc;
  real<lower = 0.> alpha_scale;
  vector[K] lambda_loc;
  vector<lower = 0.>[K] lambda_scale;
  vector[K] lambda_alpha;
}
parameters {
  // item difficulties
  vector[K] alpha;
  // item discrimination
  vector[K] lambda;
  // unknown ideal points
  vector[N] theta_raw;
}
transformed parameters {
  // create theta from observed and parameter ideal points
  vector[Y_obs] mu;
  vector[N] theta;
  theta = (theta_raw - mean(theta_raw)) ./ sd(theta_raw);
  for (i in 1:Y_obs) {
    mu[i] = alpha[y_idx_vote[i]] + lambda[y_idx_vote[i]] * theta[y_idx_leg[i]];
  }
}
model {
  alpha ~ normal(alpha_loc, alpha_scale);
  lambda ~ skew_normal(lambda_loc, lambda_scale, lambda_alpha);
  theta_raw ~ normal(0., 1.);
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
