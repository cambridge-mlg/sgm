"""Tests for affine transformations."""
from pathlib import Path

from absl.testing import parameterized
import numpy as np
import jax
from jax import numpy as jnp
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
from PIL import Image

from src.transformations.affine import (
    affine_transform_image,
    gen_transform_mat,
    create_generator_matrices,
)


def _pytorch_transform_image(image, η):
    G = np.array(gen_transform_mat(η))
    image = torch.from_numpy(np.moveaxis(image[np.newaxis, :, :], -1, 1))
    flowgrid = F.affine_grid(
        torch.from_numpy(G[np.newaxis, :2, :]), size=image.size(), align_corners=True
    )
    output = F.grid_sample(
        image, flowgrid, align_corners=True, padding_mode="zeros", mode="nearest"
    )
    return np.moveaxis(output.numpy(), 1, -1)[0]


class AffineTransformTests(parameterized.TestCase):
    """Tests for affine transformations."""

    def setUp(self):
        testdata_filename = Path(__file__).parent / "checkerboard.png"
        input_image = Image.open(testdata_filename).convert("RGB")
        self.input_array = jnp.array(input_image, dtype=jnp.float32)

    def test_identity(self):
        η = jnp.zeros(7, dtype=jnp.float32)

        jax_output = affine_transform_image(self.input_array, η)
        pt_output = _pytorch_transform_image(np.array(self.input_array), η)

        self.assertLess(jnp.mean(jnp.abs(jax_output - self.input_array)) / 255, 0.02)
        self.assertLess(jnp.mean(jnp.abs(jax_output - pt_output)) / 255, 0.02)

    @parameterized.named_parameters(
        # small transformations
        {"testcase_name": "small_trans_x", "η": [0.1, 0, 0, 0, 0, 0, 0]},
        {"testcase_name": "small_trans_y", "η": [0, 0.1, 0, 0, 0, 0, 0]},
        {"testcase_name": "small_rot", "η": [0, 0, 0.1, 0, 0, 0, 0]},
        {"testcase_name": "small_scale_x", "η": [0, 0, 0, 0.1, 0, 0, 0]},
        {"testcase_name": "small_scale_y", "η": [0, 0, 0, 0, 0.1, 0, 0]},
        {"testcase_name": "small_shear_x", "η": [0, 0, 0, 0, 0, 0.1, 0]},
        {"testcase_name": "small_shear_y", "η": [0, 0, 0, 0, 0, 0, 1.0]},
        # larger transformations
        {"testcase_name": "big_trans_x", "η": [1.0, 0, 0, 0, 0, 0, 0]},
        {"testcase_name": "big_trans_y", "η": [0, 1.0, 0, 0, 0, 0, 0]},
        {"testcase_name": "big_rot", "η": [0, 0, 1.0, 0, 0, 0, 0]},
        {"testcase_name": "big_scale_x", "η": [0, 0, 0, 1.0, 0, 0, 0]},
        {"testcase_name": "big_scale_y", "η": [0, 0, 0, 0, 1.0, 0, 0]},
        {"testcase_name": "big_shear_x", "η": [0, 0, 0, 0, 0, 1.0, 0]},
        {"testcase_name": "big_shear_y", "η": [0, 0, 0, 0, 0, 0, 1.0]},
        # combos
        {"testcase_name": "trans_x_and_y", "η": [0.1, 0.1, 0, 0, 0, 0, 0]},
        {"testcase_name": "trans_x_and_rot", "η": [0.1, 0, 0.1, 0, 0, 0, 0]},
        {"testcase_name": "trans_y_and_scale_x", "η": [0, 0.1, 0, 0.1, 0, 0, 0]},
        {"testcase_name": "scale_x_and_y", "η": [0, 0, 0, 0.1, 0.1, 0, 0]},
        {"testcase_name": "trans_x_and_scale_y", "η": [0, 0.1, 0, 0, 0, 0.1, 0]},
        {"testcase_name": "all", "η": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
    )
    def test_vs_pytorch(self, η):
        jax_output = affine_transform_image(self.input_array, jnp.array(η))
        pt_output = _pytorch_transform_image(np.array(self.input_array), jnp.array(η))

        self.assertLess(jnp.mean(jnp.abs(jax_output - pt_output)) / 255, 0.02)


class AffineMatrixTests(parameterized.TestCase):
    """Tests for generating matrices for affine transformations."""

    @parameterized.parameters(
        {"θ": 0.0},
        {"θ": jnp.pi / 4},
        {"θ": jnp.pi / 2},
        {"θ": jnp.pi},
        {"θ": -jnp.pi / 4},
        {"θ": -jnp.pi / 2},
        {"θ": -jnp.pi},
    )
    def test_rotation(self, θ):
        η = jnp.array([0.0, 0.0, θ, 0.0, 0.0, 0.0, 0.0])
        T = gen_transform_mat(η)

        # pylint: disable=bad-whitespace
        T_rot = jnp.array(
            [[jnp.cos(θ), -jnp.sin(θ), 0.0], [jnp.sin(θ), jnp.cos(θ), 0.0], [0.0, 0.0, 1.0]]
        )
        # pylint: enable=bad-whitespace
        np.testing.assert_allclose(T, T_rot, rtol=1e-7, atol=1e-7)

    @parameterized.parameters(
        {"tx": 0.0, "ty": 0.0},
        {"tx": 1.0, "ty": 0.0},
        {"tx": 0.0, "ty": 1.0},
        {"tx": 1.0, "ty": 1.0},
        {"tx": 5.0, "ty": 5.0},
        {"tx": -5.0, "ty": -5.0},
    )
    def test_translation(self, tx, ty):
        η = jnp.array([tx, ty, 0.0, 0.0, 0.0, 0.0, 0.0])
        T = gen_transform_mat(η)

        T_trans = jnp.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(T, T_trans, rtol=1e-7, atol=1e-7)

    @parameterized.parameters(
        {"sx": 1.0, "sy": 1.0},
        {"sx": 2.0, "sy": 1.0},
        {"sx": 1.0, "sy": 2.0},
        {"sx": 2.0, "sy": 2.0},
        {"sx": 0.5, "sy": 0.5},
    )
    def test_scaling(self, sx, sy):
        η = jnp.array([0.0, 0.0, 0.0, sx, sy, 0.0, 0.0])
        T = gen_transform_mat(η)

        # pylint: disable=bad-whitespace
        T_trans = jnp.array([[jnp.exp(sx), 0.0, 0.0], [0.0, jnp.exp(sy), 0.0], [0.0, 0.0, 1.0]])
        # pylint: enable=bad-whitespace
        np.testing.assert_allclose(T, T_trans, rtol=1e-7, atol=1e-7)


def _pytorch_expm(A, rtol=1e-4):
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return odeint(lambda t, x: A @ x, I, torch.tensor([0.0, 1.0]).to(A.device, A.dtype), rtol=rtol)[
        -1
    ]


class MatrixExpTests(parameterized.TestCase):
    """Tests comparing the jax.scipy and PyTorch (ode-based) matrix exponentials."""

    @parameterized.named_parameters(
        {"testcase_name": "identity", "η": [0.0, 0, 0, 0, 0, 0, 0]},
        # small transformations
        {"testcase_name": "small_trans_x", "η": [0.1, 0, 0, 0, 0, 0, 0]},
        {"testcase_name": "small_trans_y", "η": [0, 0.1, 0, 0, 0, 0, 0]},
        {"testcase_name": "small_rot", "η": [0, 0, 0.1, 0, 0, 0, 0]},
        {"testcase_name": "small_scale_x", "η": [0, 0, 0, 0.1, 0, 0, 0]},
        {"testcase_name": "small_scale_y", "η": [0, 0, 0, 0, 0.1, 0, 0]},
        {"testcase_name": "small_shear_x", "η": [0, 0, 0, 0, 0, 0.1, 0]},
        {"testcase_name": "small_shear_y", "η": [0, 0, 0, 0, 0, 0, 1.0]},
        # larger transformations
        {"testcase_name": "big_trans_x", "η": [1.0, 0, 0, 0, 0, 0, 0]},
        {"testcase_name": "big_trans_y", "η": [0, 1.0, 0, 0, 0, 0, 0]},
        {"testcase_name": "big_rot", "η": [0, 0, 1.0, 0, 0, 0, 0]},
        {"testcase_name": "big_scale_x", "η": [0, 0, 0, 1.0, 0, 0, 0]},
        {"testcase_name": "big_scale_y", "η": [0, 0, 0, 0, 1.0, 0, 0]},
        {"testcase_name": "big_shear_x", "η": [0, 0, 0, 0, 0, 1.0, 0]},
        {"testcase_name": "big_shear_y", "η": [0, 0, 0, 0, 0, 0, 1.0]},
        # combos
        {"testcase_name": "trans_x_and_y", "η": [0.1, 0.1, 0, 0, 0, 0, 0]},
        {"testcase_name": "trans_x_and_rot", "η": [0.1, 0, 0.1, 0, 0, 0, 0]},
        {"testcase_name": "trans_y_and_scale_x", "η": [0, 0.1, 0, 0.1, 0, 0, 0]},
        {"testcase_name": "scale_x_and_y", "η": [0, 0, 0, 0.1, 0.1, 0, 0]},
        {"testcase_name": "trans_x_and_scale_y", "η": [0, 0.1, 0, 0, 0, 0.1, 0]},
        {"testcase_name": "all", "η": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
    )
    def test_vs_pytorch(self, η):
        Gs = create_generator_matrices()

        Gsum = (np.array(η)[:, np.newaxis, np.newaxis] * Gs).sum(axis=0)

        pt_T = _pytorch_expm(torch.from_numpy(np.array(Gsum)), rtol=1e-6)
        jax_T = jax.scipy.linalg.expm(Gsum)

        np.testing.assert_allclose(jax_T, pt_T, rtol=1e-6, atol=1e-6)
