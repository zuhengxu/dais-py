import jax
import jax.example_libraries.stax as stax
import jax.random as random
from jax.nn import sigmoid
from jax.example_libraries.stax import Dense, serial, Softplus, FanInSum, FanOut, Identity, parallel, Relu
import jax.numpy as np



def initialize_embedding(rng, nbridges, emb_dim, factor=0.05):
	return jax.random.normal(rng, shape = (nbridges, emb_dim)) * factor


def initialize_mcd_network(x_dim, emb_dim, nbridges, rho_dim=0, nlayers=4):
	in_dim = x_dim + rho_dim + emb_dim

	layers = ([
		serial(FanOut(2), parallel(Identity, serial(Dense(in_dim), Softplus)), FanInSum),
		serial(FanOut(2), parallel(Identity, serial(Dense(in_dim), Softplus)), FanInSum),
		Dense(x_dim)
		])
	
	init_fun_nn, apply_fun_nn = serial(*layers)

	def init_fun(rng, input_shape):
		params = {}
		output_shape, params_nn = init_fun_nn(rng, (in_dim,))
		params["nn"] = params_nn
		rng, _ = jax.random.split(rng)
		params["emb"] = initialize_embedding(rng, nbridges, emb_dim)
		params["factor_sn"] = np.array(0.)
		return output_shape, params
	
	def apply_fun(params, inputs, i, **kwargs):
		# inputs has size (x_dim)
		emb = params["emb"][i, :] # (emb_dim,)
		input_all = np.concatenate([inputs, emb])
		return apply_fun_nn(params["nn"], input_all) * params["factor_sn"] # (x_dim,)

	return init_fun, apply_fun


def ScaledSigmoid(scale = 1.0):
    """Returns a sigmoid function scaled by a factor provided in kwargs."""
    def init_fun(rng, input_shape):
        return input_shape, ()
    
    def apply_fun(params, inputs, **kwargs):
        return scale * sigmoid(inputs)
    
    return init_fun, apply_fun

def amortize_eps_nn(outputdim, b):
    init_fun, apply_fun = stax.serial(
        Dense(32),  # First layer with 32 hidden units
        Relu,     # ReLU activation
        Dense(outputdim),  # last layer with outputdim hidden units
        ScaledSigmoid(b)  # Custom sigmoid activation scaled by b
    )
    return init_fun, apply_fun



# # Usage example:
# b = 0.01  # Upper bound for epsilon(t)
# outdim = 5 # Output dimension of the network
# key = random.PRNGKey(0)  # Random key for initialization
# init_fun, apply_fun = amortize_eps_nn(outdim, b)
# output_shape, params = init_fun(key, (-1, 1))  # Initialize with input shape (-1, 1)

# ngridb = 32
# mgridref_y = np.ones(ngridb + 1) * 1.
# gridref_y = np.cumsum(mgridref_y) / np.sum(mgridref_y)
# gridref_y = np.concatenate([np.array([0.]), gridref_y])
# gridref_y[2]

# apply_fun(params, np.array([gridref_y[2]]))  # Apply the network to some input data

# # Apply the network to some input data
# x = np.array(0.1)  # Example input
# outputs = apply_fun(params, x)
# outputs
# print(outputs)
