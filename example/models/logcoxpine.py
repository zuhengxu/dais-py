import jax
import jax.numpy as np

# update jax config to enable 64-bit precision
jax.config.update("jax_enable_x64", True)

dim = 32
jax.random.normal(jax.random.PRNGKey(0), (dim,), dtype=jax.numpy.float64)
