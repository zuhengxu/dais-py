import jax.numpy as np
import numpyro.distributions as npdist
import jax


def encode_params(mean, logdiag):
	return {'mean': mean, 'logdiag': logdiag}

def decode_params(params):
	mean, logdiag = params['mean'], params['logdiag']
	return mean, logdiag

def to_scale(logdiag):
	return np.exp(logdiag)
	# return (logdiag + np.sqrt(4. + logdiag * logdiag)) / 2.

def initialize(dim):
	mean = np.zeros(dim)
	logdiag = np.zeros(dim)
	return encode_params(mean, logdiag)

def build(params):
	mean, logdiag = decode_params(params)
	return npdist.Independent(npdist.Normal(loc = mean, scale = to_scale(logdiag)), 1)

def log_prob(z, params):
	dist = build(params)
	return dist.log_prob(z)

def log_prob_frozen(z, params):
	dist = build(jax.lax.stop_gradient(params))
	return dist.log_prob(z)

def entropy(params):
	# mean, logdiag = decode_params(params)
	# dim = mean.shape[0]
	# return dim * (1. + np.log(2. * np.pi)) / 2. + np.sum(logdiag)
	dist = build(params)
	return dist.entropy()

def reparameterize(params, eps):
	mean, logdiag = decode_params(params)
	return to_scale(logdiag) * eps + mean

def sample_eps(rng_key, dim):
	return jax.random.normal(rng_key, shape = (dim,))

def sample_rep(rng_key, params):
	mean, _ = decode_params(params)
	dim = mean.shape[0]
	eps = sample_eps(rng_key, dim)
	return reparameterize(params, eps)





