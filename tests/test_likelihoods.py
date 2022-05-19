from math import prod
from absl.testing import parameterized
from jax import random
import numpy as np
from numpy.testing import assert_raises, assert_array_equal, assert_array_almost_equal
from chex import assert_shape, assert_rank, assert_equal_shape

from  src.models import enc_dec


class LikelihoodTests(parameterized.TestCase):
    """Tests for decoder likelihoods."""

    @parameterized.named_parameters(
        {'testcase_name': 'FC_iso-normal', 'likelihood': 'iso-normal', 'decoder': 'FC'},
        {'testcase_name': 'FC_unit-iso-normal', 'likelihood': 'unit-iso-normal', 'decoder': 'FC'},
        {'testcase_name': 'FC_diag-normal', 'likelihood': 'diag-normal', 'decoder': 'FC'},
        {'testcase_name': 'FC_hetero-diag-normal', 'likelihood': 'hetero-diag-normal', 'decoder': 'FC'},
        {'testcase_name': 'FC_hetero-iso-normal', 'likelihood': 'hetero-iso-normal', 'decoder': 'FC'},
        {'testcase_name': 'FC_bernoulli', 'likelihood': 'bernoulli', 'decoder': 'FC'},
        #
        {'testcase_name': 'Conv_iso-normal', 'likelihood': 'iso-normal', 'decoder': 'Conv'},
        {'testcase_name': 'Conv_unit-iso-normal', 'likelihood': 'unit-iso-normal', 'decoder': 'Conv'},
        {'testcase_name': 'Conv_diag-normal', 'likelihood': 'diag-normal', 'decoder': 'Conv'},
        {'testcase_name': 'Conv_hetero-diag-normal', 'likelihood': 'hetero-diag-normal', 'decoder': 'Conv'},
        {'testcase_name': 'Conv_hetero-iso-normal', 'likelihood': 'hetero-iso-normal', 'decoder': 'Conv'},
        {'testcase_name': 'Conv_bernoulli', 'likelihood': 'bernoulli', 'decoder': 'Conv'},
        #
        {'testcase_name': 'ConvNeXt_iso-normal', 'likelihood': 'iso-normal', 'decoder': 'ConvNeXt'},
        {'testcase_name': 'ConvNeXt_unit-iso-normal', 'likelihood': 'unit-iso-normal', 'decoder': 'ConvNeXt'},
        {'testcase_name': 'ConvNeXt_diag-normal', 'likelihood': 'diag-normal', 'decoder': 'ConvNeXt'},
        {'testcase_name': 'ConvNeXt_hetero-diag-normal', 'likelihood': 'hetero-diag-normal', 'decoder': 'ConvNeXt'},
        {'testcase_name': 'ConvNeXt_hetero-iso-normal', 'likelihood': 'hetero-iso-normal', 'decoder': 'ConvNeXt'},
        {'testcase_name': 'ConvNeXt_bernoulli', 'likelihood': 'bernoulli', 'decoder': 'ConvNeXt'},
    )
    def test_all_combos(self, likelihood, decoder):
        img_shape = (32, 32, 3)
        output_shape = (prod(img_shape),) if 'Conv' not in decoder else img_shape

        dec = getattr(enc_dec, decoder + 'Decoder')(
            image_shape=img_shape,
            likelihood=likelihood,
            hidden_dims=[64],
        )

        rng_z1, rng_z2, rng_init = random.split(random.PRNGKey(0), 3)

        z1 = random.normal(rng_z1, (32,))
        z2 = random.normal(rng_z2, (32,))

        params = dec.init(rng_init, z1)
        p_x_z1 = dec.apply(params, z1)
        p_x_z2 = dec.apply(params, z2)

        if likelihood == 'iso-normal':
            μ1, σ1 = p_x_z1.loc, p_x_z1.scale
            assert_shape(μ1, output_shape)
            assert_shape(σ1, output_shape)
            assert_array_almost_equal(σ1, np.ones_like(σ1) * σ1)

            μ2, σ2 = p_x_z2.loc, p_x_z2.scale
            assert_raises(AssertionError, assert_array_equal, μ1, μ2)
            assert_array_almost_equal(σ1, σ2)

        if likelihood == 'unit-iso-normal':
            μ1, σ1 = p_x_z1.loc, p_x_z1.scale
            assert_shape(μ1, output_shape)
            assert_shape(σ1, output_shape)
            assert_array_almost_equal(σ1, np.ones_like(σ1))

            μ2, σ2 = p_x_z2.loc, p_x_z2.scale
            assert_raises(AssertionError, assert_array_equal, μ1, μ2)
            assert_array_almost_equal(σ1, σ2)

        if likelihood == 'diag-normal':
            μ1, σ1 = p_x_z1.loc, p_x_z1.scale
            assert_shape(μ1, output_shape)
            assert_shape(σ1, output_shape)

            μ2, σ2 = p_x_z2.loc, p_x_z2.scale
            assert_raises(AssertionError, assert_array_equal, μ1, μ2)
            assert_array_almost_equal(σ1, σ2)

        if likelihood == 'hetero-diag-normal':
            μ1, σ1 = p_x_z1.loc, p_x_z1.scale
            assert_shape(μ1, output_shape)
            assert_shape(σ1, output_shape)

            μ2, σ2 = p_x_z2.loc, p_x_z2.scale
            assert_raises(AssertionError, assert_array_equal, μ1, μ2)
            assert_raises(AssertionError, assert_array_equal, σ1, σ2)

        if likelihood == 'hetero-iso-normal':
            μ1, σ1 = p_x_z1.loc, p_x_z1.scale
            assert_shape(μ1, output_shape)
            assert_shape(σ1, output_shape)
            assert_array_almost_equal(σ1, np.ones_like(σ1) * σ1)

            μ2, σ2 = p_x_z2.loc, p_x_z2.scale
            assert_raises(AssertionError, assert_array_equal, μ1, μ2)
            assert_raises(AssertionError, assert_array_equal, σ1, σ2)

        if likelihood == 'bernoulli':
            logits1 = p_x_z1.logits
            assert_shape(logits1, output_shape)

            logits2 = p_x_z2.logits
            assert_raises(AssertionError, assert_array_equal, logits1, logits2)
