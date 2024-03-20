"""galaxy_mnist dataset."""

from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

_IMAGE_DATA_URL = "https://dl.dropboxusercontent.com/s/lj89307kjx5plme9f66js/galaxy_mnist_images.tar.gz?rlkey=gvlwx2rl3hqpo3gplb9zlqrz0"
_TRAIN_DATA_URL = "https://dl.dropboxusercontent.com/s/6xlym0ney5q9aec14vvy1/galaxy_mnist_train_catalog.parquet?rlkey=fc5knqscwk2h6r4z5d3156vzr&dl=0"
_TEST_DATA_URL = "https://dl.dropboxusercontent.com/s/9assxy247i1nq8wjy7o0a/galaxy_mnist_test_catalog.parquet?rlkey=y1ns24xw0tcyntodi61rv14fu&dl=0"


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for galaxy_mnist dataset."""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "GalaxyMNIST Dataset: A dataset of galaxy images for classification."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "label": tfds.features.ClassLabel(
                        num_classes=4
                    ),  # Adjust num_classes as necessary
                }
            ),
            supervised_keys=("image", "label"),
            homepage="http://example.com/galaxymnist",
            citation=r"""@article{yourcitation,""",
        )

    def _split_generators(self, dl_manager):
        # Download the dataset files
        downloaded_files = dl_manager.download(
            {
                "train_data": _TRAIN_DATA_URL,
                "test_data": _TEST_DATA_URL,
                "images": _IMAGE_DATA_URL,
            }
        )

        # Extract the downloaded files
        extracted_path = dl_manager.extract(downloaded_files["images"])

        # Return split generators
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "images_dir": extracted_path,
                    "data_frame": pd.read_parquet(downloaded_files["train_data"]),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "images_dir": extracted_path,
                    "data_frame": pd.read_parquet(downloaded_files["test_data"]),
                },
            ),
        ]

    def _generate_examples(self, images_dir, data_frame):
        """Generates examples as (key, example) tuples."""
        for idx, row in data_frame.iterrows():
            file_name, label = row["filename"], row["label"]
            image_path = str(Path(images_dir) / "images" / file_name)
            # image = tf.io.decode_jpeg(tf.io.read_file(image_path))

            yield idx, {
                "image": image_path,
                "label": label,
            }
