from functools import partial
from typing import Optional

from chex import Array, assert_rank, assert_shape
from jax.scipy import linalg

from src.transformations.affine import (
    gen_affine_matrix,
    gen_affine_matrix_no_shear,
    transform_image_with_affine_matrix,
)
from src.transformations.color import (
    color_transform_image,
    gen_hsv_in_yiq_matrix,
)


class Transform:
    def __init__(
        self, n_aff_params: int = 0, n_color_params: int = 0, η: Optional[Array] = None
    ):
        self.n_aff_params = n_aff_params
        self.n_color_params = n_color_params

        if η is not None:
            assert_shape(η, (self.n_aff_params + self.n_color_params,))
            self.η = η
        self.aff_matrix = None
        self.color_matrix = None
        self.gen_aff_matrix = None
        self.gen_color_matrix = None
        self.transform_with_aff_matrix = None
        self.transform_with_color_matrix = None

    def _set_matrices(self):
        if self.gen_aff_matrix is not None:
            self.aff_matrix = self.gen_aff_matrix(self.η[: self.n_aff_params])

        if self.gen_color_matrix is not None:
            self.color_matrix = self.gen_color_matrix(self.η[self.n_aff_params :])

    @classmethod
    def create(
        cls,
        aff_matrix: Optional[Array],
        color_matrix: Optional[Array],
        η=None,
    ):
        transform = cls()
        transform.aff_matrix = aff_matrix
        transform.color_matrix = color_matrix
        if η is not None:
            transform.η = η
        return transform

    def apply(self, image: Array, **kwargs) -> Array:
        assert_rank(image, 3)

        if self.aff_matrix is not None:
            image = self.transform_with_aff_matrix(image, self.aff_matrix, **kwargs)

        if self.color_matrix is not None:
            image = self.transform_with_color_matrix(image, self.color_matrix, **kwargs)

        return image

    def inverse(self) -> Array:
        inv_aff_matrix = (
            linalg.inv(self.aff_matrix) if self.aff_matrix is not None else None
        )
        inv_color_matrix = (
            linalg.inv(self.color_matrix) if self.color_matrix is not None else None
        )

        return self.create(inv_aff_matrix, inv_color_matrix)

    def compose(self, other_transform) -> Array:
        new_aff_matrix = (
            self.aff_matrix @ other_transform.aff_matrix
            if self.aff_matrix is not None
            else None
        )
        new_color_matrix = (
            self.color_matrix @ other_transform.color_matrix
            if self.color_matrix is not None
            else None
        )

        return self.create(new_aff_matrix, new_color_matrix)

    def __lshift__(self, other: "Transform") -> "Transform":
        """
        Allows for calling `transform1.compose(transform2)` as `transform2 << transform1` (indicating that
        transform1) will be called first.
        """
        return other.compose(self)


class AffineTransform(Transform):
    def __init__(self, η: Optional[Array] = None):
        super().__init__(n_aff_params=6, η=η)
        self.gen_aff_matrix = gen_affine_matrix
        self.transform_with_aff_matrix = transform_image_with_affine_matrix

        if η is not None:
            self._set_matrices()


class AffineTransformWithoutShear(Transform):
    def __init__(self, η: Optional[Array] = None):
        super().__init__(n_aff_params=5, η=η)
        self.gen_aff_matrix = gen_affine_matrix_no_shear
        self.transform_with_aff_matrix = transform_image_with_affine_matrix

        if η is not None:
            self._set_matrices()


class HueTransform(Transform):
    def __init__(self, η: Optional[Array] = None):
        super().__init__(n_color_params=1, η=η)
        self.gen_color_matrix = partial(gen_hsv_in_yiq_matrix, only="hue")
        self.transform_with_color_matrix = color_transform_image

        if η is not None:
            self._set_matrices()

    def inverse(self) -> Array:
        inv_aff_matrix = None
        new_η = -self.η
        inv_color_matrix = self.gen_color_matrix(new_η[self.n_aff_params :])
        return self.create(inv_aff_matrix, inv_color_matrix, new_η)

    def compose(self, other_transform) -> Array:
        new_aff_matrix = None
        new_η = self.η + other_transform.η
        new_color_matrix = self.gen_color_matrix(new_η[self.n_aff_params :])

        return self.create(new_aff_matrix, new_color_matrix, new_η)

    def apply(self, image: Array, **kwargs) -> Array:
        assert_rank(image, 3)

        # NOTE: We always use the naive HSV transform here. The color_matrix is only used for the huber loss.
        image = self.transform_with_color_matrix(image, self.η)

        return image


class HueSaturationTransform(Transform):
    def __init__(self, η: Optional[Array] = None):
        super().__init__(n_color_params=2, η=η)
        self.gen_color_matrix = partial(gen_hsv_in_yiq_matrix, only="hue_sat")
        self.transform_with_color_matrix = partial(
            color_transform_image, transform="hue_sat"
        )

        if η is not None:
            self._set_matrices()

    def inverse(self) -> Array:
        inv_aff_matrix = None
        new_η = -self.η
        inv_color_matrix = self.gen_color_matrix(new_η[self.n_aff_params :])
        return self.create(inv_aff_matrix, inv_color_matrix, new_η)

    def compose(self, other_transform) -> Array:
        new_aff_matrix = None
        new_η = self.η + other_transform.η
        new_color_matrix = self.gen_color_matrix(new_η[self.n_aff_params :])

        return self.create(new_aff_matrix, new_color_matrix, new_η)

    def apply(self, image: Array, **kwargs) -> Array:
        assert_rank(image, 3)

        # NOTE: We always use the naive HSV transform here. The color_matrix is only used for the huber loss.
        image = self.transform_with_color_matrix(image, self.η)

        return image


class HSVTransform(Transform):
    def __init__(self, η: Optional[Array] = None):
        super().__init__(n_color_params=3, η=η)
        self.gen_color_matrix = partial(gen_hsv_in_yiq_matrix, only="hue_sat_val")
        self.transform_with_color_matrix = partial(
            color_transform_image, transform="hue_sat_val"
        )

        if η is not None:
            self._set_matrices()

    def inverse(self) -> Array:
        inv_aff_matrix = None
        new_η = -self.η
        inv_color_matrix = self.gen_color_matrix(new_η[self.n_aff_params :])
        return self.create(inv_aff_matrix, inv_color_matrix, new_η)

    def compose(self, other_transform) -> Array:
        new_aff_matrix = None
        new_η = self.η + other_transform.η
        new_color_matrix = self.gen_color_matrix(new_η[self.n_aff_params :])

        return self.create(new_aff_matrix, new_color_matrix, new_η)

    def apply(self, image: Array, **kwargs) -> Array:
        assert_rank(image, 3)

        # NOTE: We always use the naive HSV transform here. The color_matrix is only used for the huber loss.
        image = self.transform_with_color_matrix(image, self.η)

        return image
