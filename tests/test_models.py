"""Tests for modeling components."""
from absl.testing import parameterized
import numpy as np
from jax import numpy as jnp

from src.models.livae import make_η_bounded


class TransformParamTests(parameterized.TestCase):
    """Tests for the transformation parameters."""

    @parameterized.named_parameters(
        {"α": 0.0, "testcase_name": "α=0.0"},
        {"α": 1.0, "testcase_name": "α=1.0"},
        {"α": -1.0, "testcase_name": "α=-1.0"},
    )
    def test_equality(self, α):
        bounds = jnp.array([0.25, 0.25, jnp.pi, 0.25, 0.25, jnp.pi / 6, jnp.pi / 6])

        η = α * bounds
        η_bounded = make_η_bounded(η, bounds)

        np.testing.assert_allclose(η, η_bounded)

    @parameterized.named_parameters(
        {"α": 2.0, "testcase_name": "α=2.0"},
        {"α": -2.0, "testcase_name": "α=-2.0"},
        {"α": 3.0, "testcase_name": "α=3.0"},
        {"α": -3.0, "testcase_name": "α=-3.0"},
        {"α": 4.0, "testcase_name": "α=4.0"},
        {"α": -4.0, "testcase_name": "α=-4.0"},
        {"α": 50.0, "testcase_name": "α=50.0"},
        {"α": -50.0, "testcase_name": "α=-50.0"},
    )
    def test_multiples(self, α):
        bounds = jnp.array([0.25, 0.25, jnp.pi, 0.25, 0.25, jnp.pi / 6, jnp.pi / 6])

        η = α * bounds
        η_bounded = make_η_bounded(η, bounds)

        # np.testing.assert_allclose(bounds * jnp.imag(jnp.exp(1j * α * jnp.pi / 2)), η_bounded)
        np.testing.assert_allclose(bounds * jnp.sign(α), η_bounded)
