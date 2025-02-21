"""From PDD
"""

import abc

from check_shapes import check_shapes

import jax
import jax.numpy as jnp
import jax.scipy.linalg as slinalg
from jax.scipy.special import logsumexp
import numpy as np
import numpyro
from jax.flatten_util import ravel_pytree
from models.data_utils import load_data

from jaxtyping import Float as f, PRNGKeyArray, Array, install_import_hook
import typing as tp

from chex import assert_axis_dimension, assert_shape, assert_equal_shape

# with install_import_hook("pdds", "typeguard.typechecked"):
import models.cp_utils as cp_utils

Key = PRNGKeyArray


class Distribution(metaclass=abc.ABCMeta):
    def __init__(self, dim: int, is_target: bool):
        self.dim = dim
        self.is_target = (
            is_target  # marks whether or not to incrememt density_state on call
        )

    @abc.abstractmethod
    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        """Produce samples from the specified distribution.

        Args:
            key: Jax key
            num_samples: int, number of samples to draw.
        Returns:
            Array of shape (num_samples, dim) containing samples from the distribution.
        """

    @abc.abstractmethod
    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        """Evaluate the log likelihood of the specified distribution.

        Args:
            x: Samples.
            density_state: int, tracks number of density evaluations.
        Returns:
            Array of shape (batch_size,) containing values of log densities.
            Int containing updated density state.
        """


class WhitenedDistributionWrapper(Distribution):
    """Reparametrizes the target distribution based on the
    learnt variational reference measure."""

    @check_shapes("vi_means: [d]", "vi_scales: [d]")
    def __init__(
        self,
        target: Distribution,
        vi_means: Array,
        vi_scales: Array,
        is_target: bool = False,
    ):
        super().__init__(dim=target.dim, is_target=is_target)
        self.target = target
        self.vi_means = vi_means
        self.vi_scales = vi_scales
        # for GaussianRatioAnalyticPotential (only works when dim=1)
        if target.dim == 1:
            self._mean = vi_means[0]
            self._scale = vi_scales[0]

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        transformed_x = self.vi_means + x * self.vi_scales
        out, density_state = self.target.evaluate_log_density(
            transformed_x, density_state
        )
        out += jnp.log(jnp.prod(self.vi_scales))
        return out, density_state

    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        original_samples = self.target.sample(key, num_samples)
        return (original_samples - self.vi_means) / self.vi_scales


class NormalDistribution(Distribution):
    """Multivariate normal distribution with shared diagonal covariance only. Scale
    is a scalar value giving the shared scale for the diagonal covariance."""

    @check_shapes("mean: [b, d]")
    def __init__(
        self,
        mean: Array,
        scale: tp.Union[float, f[Array, ""]],
        dim: int = 1,
        is_target: bool = False,
    ):
        super().__init__(dim, is_target)
        assert_axis_dimension(mean, 1, dim)
        self._mean = mean
        self._scale = scale

    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        batched_sample_shape = (num_samples,) + (self.dim,)
        self._cov_matrix = self._scale**2 * jnp.eye(self.dim)
        samples = jax.random.multivariate_normal(
            key=key, mean=self._mean, cov=self._cov_matrix, shape=(num_samples,)
        )
        assert_shape(samples, batched_sample_shape)
        return samples

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        out = jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=self._mean, cov=self._scale**2
        )
        density_state += self.is_target * x.shape[0]
        return out, density_state


class BatchedNormalDistribution(Distribution):
    """Same as NormalDistribution above except that the scale is also batched as well as the mean, i.e.
    scale is a [b] dimensional vector with the unique shared scale for each sample in the batch.
    """

    @check_shapes("means: [b, d]", "scales: [b]")
    def __init__(
        self, means: Array, scales: Array, dim: int = 1, is_target: bool = False
    ):
        super().__init__(dim, is_target)
        assert_axis_dimension(means, 1, dim)
        self._mean = means
        self._scale = scales

    @check_shapes("return: [..., d]")
    def sample(self, key: Key, num_samples: tp.Tuple) -> Array:
        batched_sample_shape = (*num_samples,) + (self.dim,)
        self._cov_matrix = self._scale[..., None, None] ** 2 * jnp.tile(
            jnp.eye(self.dim)[None, ...], (*num_samples, 1, 1)
        )
        samples = jax.random.multivariate_normal(
            key=key, mean=self._mean, cov=self._cov_matrix, shape=(*num_samples,)
        )
        assert_shape(samples, batched_sample_shape)
        return samples

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        self._cov_matrix = self._scale[..., None, None] ** 2 * jnp.tile(
            jnp.eye(self.dim)[None, ...], (x.shape[0], 1, 1)
        )
        out = jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=self._mean, cov=self._cov_matrix
        )
        density_state += self.is_target * x.shape[0]
        return out, density_state


class MeanFieldNormalDistribution(Distribution):
    """Multivariate normal distribution with diagonal covariance (non-isotropic). Scales
    is a vector value giving the scales which go on the diagonal of the covariance."""

    @check_shapes("mean: [d]", "scales: [d]")
    def __init__(
        self, mean: Array, scales: Array, dim: int = 1, is_target: bool = False
    ):
        super().__init__(dim, is_target)
        assert_axis_dimension(mean, 0, dim)
        assert_axis_dimension(scales, 0, dim)
        self._mean = mean
        self._scales = scales

    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        batched_sample_shape = (num_samples,) + (self.dim,)
        self._cov_matrix = jnp.diag(self._scales**2)
        samples = jax.random.multivariate_normal(
            key=key, mean=self._mean, cov=self._cov_matrix, shape=(num_samples,)
        )
        assert_shape(samples, batched_sample_shape)
        return samples

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        out = jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=self._mean, cov=jnp.diag(self._scales**2)
        )
        density_state += self.is_target * x.shape[0]
        return out, density_state


def NormalDistributionWrapper(
    mean: float, scale: float, dim: int = 1, is_target: bool = False
) -> Distribution:
    """Wraps the NormalDistribution class for easy initialisation from Hydra configs."""
    means = mean * jnp.ones((1, dim))
    return NormalDistribution(means, scale, dim, is_target)


class LogGaussianCoxPines(Distribution):
    """Log Gaussian Cox process posterior in 2D for pine saplings data.

    This follows Heng et al 2020 https://arxiv.org/abs/1708.08396 .

    config.file_path should point to a csv file of num_points columns
    and 2 rows containg the Finnish pines data.

    config.use_whitened is a boolean specifying whether or not to use a
    reparameterization in terms of the Cholesky decomposition of the prior.
    See Section G.4 of https://arxiv.org/abs/2102.07501 for more detail.
    The experiments in the paper have this set to False.

    num_dim should be the square of the lattice sites per dimension.
    So for a 40 x 40 grid num_dim should be 1600.

    Implementation from https://github.com/deepmind/annealed_flow_transport/tree/master
    """

    def __init__(
        self,
        file_path: str = "models/datasets/fpines.csv",
        use_whitened: bool = False,
        dim: int = 1600,
        is_target: bool = False,
    ):
        super().__init__(dim, is_target=is_target)

        # Discretization is as in Controlled Sequential Monte Carlo
        # by Heng et al 2017 https://arxiv.org/abs/1708.08396
        self._num_latents = dim
        self._num_grid_per_dim = int(np.sqrt(dim))

        bin_counts = jnp.array(
            cp_utils.get_bin_counts(
                self.get_pines_points(file_path), self._num_grid_per_dim
            )
        )

        self._flat_bin_counts = jnp.reshape(bin_counts, (self._num_latents))

        # This normalizes by the number of elements in the grid
        self._poisson_a = 1.0 / self._num_latents
        # Parameters for LGCP are as estimated in Moller et al, 1998
        # "Log Gaussian Cox processes" and are also used in Heng et al.

        self._signal_variance = 1.91
        self._beta = 1.0 / 33

        self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

        def short_kernel_func(x, y):
            return cp_utils.kernel_func(
                x, y, self._signal_variance, self._num_grid_per_dim, self._beta
            )

        self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
        self._cholesky_gram = jnp.linalg.cholesky(self._gram_matrix)
        self._white_gaussian_log_normalizer = (
            -0.5 * self._num_latents * jnp.log(2.0 * jnp.pi)
        )

        half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self._cholesky_gram))))
        self._unwhitened_gaussian_log_normalizer = (
            -0.5 * self._num_latents * jnp.log(2.0 * jnp.pi) - half_log_det_gram
        )
        # The mean function is a constant with value mu_zero.
        self._mu_zero = jnp.log(126.0) - 0.5 * self._signal_variance

        if use_whitened:
            self._posterior_log_density = self.whitened_posterior_log_density
        else:
            self._posterior_log_density = self.unwhitened_posterior_log_density

    def get_pines_points(self, file_path):
        """Get the pines data points."""
        with open(file_path, mode="rt") as input_file:
            # with open(file_path, "rt") as input_file:
            b = np.genfromtxt(input_file, delimiter=",", skip_header=1, usecols=(1, 2))
        return b

    def whitened_posterior_log_density(self, white: Array) -> Array:
        quadratic_term = -0.5 * jnp.sum(white**2)
        prior_log_density = self._white_gaussian_log_normalizer + quadratic_term
        latent_function = cp_utils.get_latents_from_white(
            white, self._mu_zero, self._cholesky_gram
        )
        log_likelihood = cp_utils.poisson_process_log_likelihood(
            latent_function, self._poisson_a, self._flat_bin_counts
        )
        return prior_log_density + log_likelihood

    def unwhitened_posterior_log_density(self, latents: Array) -> Array:
        white = cp_utils.get_white_from_latents(
            latents, self._mu_zero, self._cholesky_gram
        )
        prior_log_density = (
            -0.5 * jnp.sum(white * white) + self._unwhitened_gaussian_log_normalizer
        )
        log_likelihood = cp_utils.poisson_process_log_likelihood(
            latents, self._poisson_a, self._flat_bin_counts
        )
        return prior_log_density + log_likelihood

    # @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        # import pdb; pdb.set_trace()
        if len(x.shape) == 1:
            density_state += self.is_target
            return self._posterior_log_density(x), density_state
        else:
            density_state += self.is_target * x.shape[0]
            return jax.vmap(self._posterior_log_density)(x), density_state

    def sample(self, key: Key, num_samples: int) -> Array:
        return NotImplementedError("LGCP target cannot be sampled")


# target = LogGaussianCoxPines()

